# -*- coding: utf-8 -*-
"""
Main structure of STL

Tentative proposal by EA
"""

import matplotlib.pyplot as plt
import numpy as np
import torch as bk  # mean, zeros

###############################################################################
###############################################################################


class ST_Statistics:
    """
    Class whose instances correspond to an set of scattering statistics
    The set of statistics is built by the ST_operator method, which use the
    __init__ method. This class is DT-independent.

    This class contains methods that allow to deal with ST statistics in an
    unified manner. Most of these methods can be applied directly through the
    ST_operator implementation.

    When used in loss, a 1D array can be return using the for_loss method.
    It works for any type of ST_statistics. It can use a mask on the ST
    coefficients, which is option-dependent

    Parameters
    ----------
    # Data type and Wavelet Transform
    - DT : str
        Type of data (1d, 2d planar, HealPix, 3d)
    - N0 : tuple
        initial size of array (can be multiple dimensions)
    - J : int
        number of scales
    - L : int
        number of orientations
    - WType : str
        type of wavelets

    # Scattering Transform
    - SC : str
        type of ST coefficients ("ScatCov", "WPH")

    # Data array parameters
    - Nb : int
        size of batch
    - Nc : int
        number of channel

    Attributes
    ----------
    - parent parameters (DT,N0,J,L,WType,SC,Nb,Nc)

    # Additional transform/compression
    - norm : str
        type of norm (“self”, “from_ref”)
    - S2_ref_sqrt_chan_diag : array
        array of reference S2 coefficients (normalized by sqrt of diagonal over channels)
    - iso : bool
        keep only isotropic coefficients
    - angular_ft : bool
        perform angular fourier transform on the ST statistics
    - scale_ft : bool
        perform scale cosine transform on the ST statistics
    - flatten : bool
        only return a 1D-array and not a ST_Statistics instance
    - mask_st : list of position
        mask to be applied when flatten ST statistics

    # ST statistics
    - S1, S2, S2p, S3, S4 : array of relevant size to store the ST statistics

    # Power Spectrum
    - PS : bool
        whether power spectrum coefficients are computed

    """

    ########################################
    def __init__(
        self,
        DataClass,
        N0,  ######################################## not used?
        Nb,
        Nc,
        wavelet_op,
        SC,
        has_fewer_convolutions,
        compute_cross_matrix,
        compute_PS,
        n_bins,
        standardized,
        mean_pre_std,
        std_pre_std,
    ):
        """
        Constructor, see details above.
        """

        # Main parameters
        self.DataClass = DataClass
        self.N0 = N0  ######################################## not used?
        self.Nb = Nb
        self.Nc = Nc

        # Wavelet transform related parameters
        self.wavelet_op = wavelet_op
        self.J = self.wavelet_op.J
        self.L = self.wavelet_op.L
        self.WType = self.wavelet_op.WType
        self.mask_full_res = self.wavelet_op.mask_full_res
        self.device = self.wavelet_op.device
        self.dtype = self.wavelet_op.dtype

        # Scattering transform related parameters
        self.SC = SC
        self.S2_ref_sqrt_chan_diag = None
        self.has_fewer_convolutions = has_fewer_convolutions
        self.compute_cross_matrix = compute_cross_matrix

        # Power spectrum computation
        self.compute_PS = compute_PS
        self.n_bins = n_bins
        self.PS_ref_sqrt_chan_diag = None

        # Mean and variance of data if standardization is applied
        self.standardized = standardized
        self.mean_pre_std = mean_pre_std
        self.std_pre_std = std_pre_std

        # Additional transform/compression related parameters. While put to
        # False/None for the initialization, their value are modified if these
        # methods are called by the scattering operator, or independently.
        self.norm = False
        self.iso = False
        self.angular_ft = False
        self.scale_ft = False
        self.flatten = False
        self.mask_st = None  # Not used in flatten method for now

    @staticmethod
    def _get_sqrt_chan_diag(stat_ref, norm_batch_mean):
        """
        Prepare S2_ref that has shape [Nb,Nc,Nc,J,L] or PS_ref that has shape [Nb,Nc,Nc,n_bins]
        by keeping its diagonal, averaging over the batch dimension if norm_batch_mean is True, and applying sqrt.
        """
        stat_ref_chan_diag = stat_ref.diagonal(dim1=1, dim2=2).movedim(
            -1, 1
        )  # [Nb,Nc,J,L] or [Nb,Nc,n_bins] for S2 or PS
        stat_ref_sqrt_chan_diag = bk.sqrt(
            stat_ref_chan_diag.mean(dim=0, keepdim=True)
            if norm_batch_mean
            else stat_ref_chan_diag
        )  # [Nb,Nc,J,L] or [Nb,Nc,n_bins]
        return stat_ref_sqrt_chan_diag

    def _normalize_scatcov(self):
        """
        Normalize the ScatCov statistics S1,S2,S3,S4
        using self.S2_ref_sqrt_chan_diag
        """
        # self.S1 = self.S1 / self.S2_ref_sqrt_chan_diag  # [Nb,Nc,J1,L1]
        self.S1 = self.S1 / bk.sqrt(
            self.S2_ref_sqrt_chan_diag[:, :, None]
            * self.S2_ref_sqrt_chan_diag[:, None, :]
        )  # [Nb,Nc,Nc,J1,L1]
        self.S2 = self.S2 / (
            self.S2_ref_sqrt_chan_diag[:, :, None]
            * self.S2_ref_sqrt_chan_diag[:, None, :]
        )  # [Nb,Nc,Nc,J1,L1]
        self.S3 = self.S3 / (
            self.S2_ref_sqrt_chan_diag[:, :, None, :, None, :, None]
            * self.S2_ref_sqrt_chan_diag[:, None, :, None, :, None, :]
        )  # [Nb,Nc,Nc,J1,J2,L1,L2]
        self.S4 = self.S4 / (
            self.S2_ref_sqrt_chan_diag[:, :, None, :, None, None, :, None, None]
            * self.S2_ref_sqrt_chan_diag[:, None, :, None, :, None, None, :, None]
        )  # [Nb,Nc,Nc,J1,J2,J3,L1,L2,L3]

    ########################################
    def to_norm(
        self,
        norm_type=None,
        S2_ref_sqrt_chan_diag=None,
        PS_ref_sqrt_chan_diag=None,
        var_ref=None,
        norm_batch_mean=True,
    ):
        """
        Normalize the ST statistics.
        Parameters
        ----------
        - norm_type : str
            type of norm (“self”, “from_ref”)
        - S2_ref_sqrt_chan_diag : array
            if self.SC = "ScatCov"
            array of reference S2 coefficients if "from_ref" (normalized by sqrt of diagonal over channels)
        - PS_ref_sqrt_chan_diag : array
            if self.PS = True
            array of reference Power Spectrum coefficients if "from_ref"
        - var_ref : array
            array of reference variance if "from_ref"
        - norm_batch_mean : bool, default True
            Used with the "self" normalization type.
            If True, the reference coefficients are averaged over the batch dimension (dim=0).
        """
        # Check the proper ordering
        if self.iso:
            raise Exception("Normalization can only be done before isotropization")
        if self.angular_ft:
            raise Exception("Normalization can only be done before angular ft")
        if self.scale_ft:
            raise Exception("Normalization can only be done before scale_ft")
        # Leave the function if no normalization is required
        if norm_type is None:
            raise Exception("No normalization type specified")
        # Verifications
        if self.norm:
            raise Exception("ST statistics are already normalized")
        # Store_ref normalization
        elif norm_type == "self":
            var_ref = self.var * 1.0
            self.var_ref = (
                var_ref if not norm_batch_mean else var_ref.mean(dim=0, keepdim=True)
            )
            if self.SC == "ScatCov":
                if self.S2_ref_sqrt_chan_diag is None:
                    # prepare self.S2 that has shape [Nb,Nc,Nc,J,L] by keeping its diagonal and applying sqrt
                    # and store as reference
                    self.S2_ref_sqrt_chan_diag = self._get_sqrt_chan_diag(
                        self.S2, norm_batch_mean=norm_batch_mean
                    )

            if self.compute_PS:
                if self.PS_ref_sqrt_chan_diag is None:
                    # prepare self.PS that has shape [Nb,Nc,Nc,J,L] by keeping its diagonal, averaging over the batch dimension if norm_batch_mean is True, and applying sqrt
                    # and store as reference
                    self.PS_ref_sqrt_chan_diag = self._get_sqrt_chan_diag(
                        self.PS, norm_batch_mean=norm_batch_mean
                    )

        # Load_ref normalization
        elif norm_type == "from_ref":

            self.var_ref = var_ref

            if self.SC == "ScatCov":
                # store as reference
                self.S2_ref_sqrt_chan_diag = S2_ref_sqrt_chan_diag

            if self.compute_PS:
                self.PS_ref_sqrt_chan_diag = PS_ref_sqrt_chan_diag

        # Perform normalization and store reference
        self.var = self.var / self.var_ref

        if self.SC == "ScatCov":
            self._normalize_scatcov()

        if self.compute_PS:
            self.PS = self.PS / (
                self.PS_ref_sqrt_chan_diag[:, :, None, :]
                * self.PS_ref_sqrt_chan_diag[:, None, :, :]
            )  # [Nb, Nc, Nc, n_bins]

        # Store normalization parameters
        self.norm = True

        return self

    def to_iso(self):
        """
        Isotropize the set of ST statistics

        Note: S2_ref_sqrt_chan_diag is not isotropized since it is used before this step.
        Note: if self.PS = True, PS coefficients are already isotropized in PS_operator.

        EA: could probably be better vectorized, to be done.
        EA: to be done properly with the backend.
        EA: Sihao used .real for S3 and S4, to consider.

        """

        if self.angular_ft:
            raise Exception("Isotropization can only be done before angular ft")
        if self.scale_ft:
            raise Exception("Isotropization can only be done before scate_ft")

        Nb, Nc = self.Nb, self.Nc
        J, L = self.J, self.L

        if self.SC == "ScatCov":

            # S1 and S2
            # self.S1 = bk.mean(self.S1.mean, -1)  # (Nb,Nc,J,L) -> (Nb,Nc,J)
            # self.S1 = bk.mean(self.S2.mean, -1)  # (Nb,Nc,J,L) -> (Nb,Nc,J)
            #            self.S1 = bk.mean(self.S1, -1)  # (Nb,Nc,J,L) -> (Nb,Nc,J)
            self.S1 = bk.mean(self.S1, -1)  # (Nb,Nc,Nc,J,L) -> (Nb,Nc,Nc,J)
            self.S2 = bk.mean(self.S2, -1)  # (Nb,Nc,Nc,J,L) -> (Nb,Nc,Nc,J)

            # S3 and S4
            S3iso = bk.zeros(
                (Nb, Nc, Nc, J, J, L), device=self.S3.device, dtype=self.S3.dtype
            )
            S4iso = bk.zeros(
                (Nb, Nc, Nc, J, J, J, L, L), device=self.S4.device, dtype=self.S3.dtype
            )
            for l1 in range(L):
                for l2 in range(L):
                    # (Nb,Nc,Nc,J,J,L,L) -> (Nb,Nc,Nc,J,J,L)
                    S3iso[..., (l2 - l1) % L] += self.S3[..., l1, l2]
                    for l3 in range(L):
                        # (Nb,Nc,Nc,J,J,J,L,L,L) -> (Nb,Nc,Nc,J,J,J,L,L)
                        S4iso[..., (l2 - l1) % L, (l3 - l1) % L] += self.S4[
                            ..., l1, l2, l3
                        ]

            self.S3 = S3iso / L
            self.S4 = S4iso / L

        # store isotropy parameter
        self.iso = True

        return self

    ########################################
    def to_angular_ft(self, harmonics_angle=None):
        """
        Angular harmonic transform on the ST statistcs
        """
        Nb, Nc = self.Nb, self.Nc
        J, L = self.J, self.L
        if harmonics_angle is None:
            harmonics_angle = self.L

        if self.scale_ft:
            raise Exception("Angular_tf can only be done before scale_ft")

        # perform angular transform, to be done
        if self.SC == "ScatCov":
            if self.iso:
                S3f = bk.fft.fftn(
                    self.S3, norm="ortho", dim=(-1)
                )  # (Nb,Nc,Nc,J,J,L) -> (Nb,Nc,Nc,J,J,L)
                S4f = bk.fft.fftn(
                    self.S4, norm="ortho", dim=(-1, -2)
                )  # (Nb,Nc,Nc,J,J,L,L) -> (Nb,Nc,Nc,J,J,L,L)
                S3f = S3f[
                    ..., :harmonics_angle
                ]  # keep zeroth, first and second harmonic
                S4f = S4f[..., :harmonics_angle, :harmonics_angle]
            else:
                S3f = bk.fft.fftn(
                    self.S3, norm="ortho", dim=(-1, -2)
                )  # (Nb,Nc,Nc,J,J,L,L) -> (Nb,Nc,Nc,J,J,L,L)
                S4f = bk.fft.fftn(
                    self.S4, norm="ortho", dim=(-1, -2, -3)
                )  # (Nb,Nc,Nc,J,J,L,L,L) -> (Nb,Nc,Nc,J,J,L,L,L)
                S3f = S3f[..., :harmonics_angle, :harmonics_angle]
                S4f = S4f[..., :harmonics_angle, :harmonics_angle, :harmonics_angle]

        self.S3 = S3f
        self.S4 = S4f
        # store angular_ft parameter
        self.angular_ft = True

        return self

    ########################################
    def to_scale_ft(self, harmonics_scale=None, dj=None, harmonics_angle=None):
        """
        Angular scale transform on the ST statistcs
        """
        Nb, Nc = self.Nb, self.Nc
        if harmonics_scale is None:
            harmonics_scale = self.J
        if dj is None:
            dj = self.J + 1
        if harmonics_angle is None:
            harmonics_angle = self.L

        J, L = self.J, harmonics_angle

        def cosinus1(N, device, dtype):
            """
            The cosine basis cos(kpi (. + 0.5) / N) for 0 <= k < N.

            :param N:
            :return:
            """
            if N == 0:
                return bk.zeros((0, 0), device=device, dtype=dtype)

            ts = bk.linspace(0, bk.pi * (N - 1) / N, N, device=device, dtype=dtype) + (
                0.5 * bk.pi / N
            )
            indices = bk.stack([k * ts for k in range(N)], dim=0)

            F = bk.cos(indices)
            F[1:, :] *= bk.sqrt(bk.tensor(2 / N, device=device, dtype=dtype))
            F[0, :] *= bk.sqrt(bk.tensor(1 / N, device=device, dtype=dtype))

            return F

        if self.SC == "ScatCov":
            if self.iso:
                S3_reparam = bk.zeros(
                    (Nb, Nc, Nc, J, J, L), device=self.S3.device, dtype=self.S3.dtype
                )
                S4_reparam = bk.zeros(
                    (Nb, Nc, Nc, J, J, J, L, L),
                    device=self.S4.device,
                    dtype=self.S3.dtype,
                )
                nan_complex = bk.tensor(
                    complex(float("nan"), float("nan")),
                    dtype=self.S3.dtype,
                    device=self.S3.device,
                )

                for j1 in range(J):
                    for j2 in range(J):
                        dj2 = (j2 - j1) % J

                        # Set to NaN if dj2 > 3
                        if dj2 >= dj:
                            S3_reparam[..., j1, dj2, :] = nan_complex
                        else:
                            S3_reparam[..., j1, dj2, :] = self.S3[..., j1, j2, :]

                        for j3 in range(J):
                            dj3 = (j3 - j1) % J
                            dj32 = (j3 - j2) % J

                            # Set to NaN if dj2 > 3 or dj3 > 3 or dj32 > 3
                            if dj2 >= dj or dj3 >= dj or dj32 >= dj:
                                S4_reparam[..., j1, dj2, dj3, :, :] = nan_complex
                            else:
                                S4_reparam[..., j1, dj2, dj3, :, :] = self.S4[
                                    ..., j1, j2, j3, :, :
                                ]

                # # Apply DCT on the first J dimension
                F = cosinus1(J, device=self.S3.device, dtype=self.S3.real.dtype)

                S3_real = S3_reparam.real
                S3_imag = S3_reparam.imag

                # Create mask for NaN values
                nan_mask = bk.isnan(S3_real) | bk.isnan(S3_imag)

                # Replace NaN with 0 for computation
                S3_real_clean = bk.nan_to_num(S3_real, nan=0.0)
                S3_imag_clean = bk.nan_to_num(S3_imag, nan=0.0)

                S3_real_reshaped = S3_real_clean.reshape(-1, J, L)
                S3_imag_reshaped = S3_imag_clean.reshape(-1, J, L)

                S3_cos_real = bk.matmul(F, S3_real_reshaped)
                S3_cos_imag = bk.matmul(F, S3_imag_reshaped)

                S3_cos = bk.complex(S3_cos_real, S3_cos_imag).reshape(S3_reparam.shape)

                # Restore NaN where they were present
                S3_cos[nan_mask] = complex(float("nan"), float("nan"))

                S4_real = S4_reparam.real
                S4_imag = S4_reparam.imag

                # Create mask for NaN values
                nan_mask = bk.isnan(S4_real) | bk.isnan(S4_imag)

                # Replace NaN with 0 for computation
                S4_real_clean = bk.nan_to_num(S4_real, nan=0.0)
                S4_imag_clean = bk.nan_to_num(S4_imag, nan=0.0)

                S4_real_reshaped = S4_real_clean.reshape(-1, J, J, L, L)
                S4_imag_reshaped = S4_imag_clean.reshape(-1, J, J, L, L)

                S4_cos_real = bk.einsum("ij,bjklm->biklm", F, S4_real_reshaped)
                S4_cos_imag = bk.einsum("ij,bjklm->biklm", F, S4_imag_reshaped)

                S4_cos = bk.complex(S4_cos_real, S4_cos_imag).reshape(S4_reparam.shape)

                # Restore NaN where they were present
                S4_cos[nan_mask] = complex(float("nan"), float("nan"))
            else:
                # TODO
                pass
        else:
            pass

        self.S3 = S3_cos[:, :, :, 0:harmonics_scale]
        self.S4 = S4_cos[:, :, :, 0:harmonics_scale]
        # store scale_ft parameter
        self.scale_ft = True

        return self

    ########################################
    def to_flatten(
        self,
        keep_batch_dim=False,
        mask_st=None,
        mean_along_batch=False,
        keepnans=False,
        flatten_complex=False,
    ):
        """
        Produce either a 1d array that can be used for loss constructions or a 2d array, keeping the batch dimension, if keep_batch_dim is True.

        A mask can be used to select the coefficients from the initial 1d array.

        Parameters
        ----------
        - keep_batch_dim : bool, default False
            if True, the output will have shape [Nb, n_coeff] instead of [Nb * n_coeff]
        - mask_st : binary 1d array
            mask for st coefficients after initial flattening
        - flatten_complex : bool, default False
            if True, complex coefficients will be flattened into two separate real numbers (real and imaginary parts).
            Since S1, S2 and PS are real, their null imaginary part won't be included in the flattened statistics tensor.
        Output
        ----------
        - st_flatten : 1d array if keep_batch_dim is False, else 2d array with shape [Nb, n_coeff]

        """

        # Collect all statistics into a list
        stats = [self.mean, self.var]  # Always include mean and variance
        stats_names = ["mean", "var"]

        if self.SC == "ScatCov":
            stats += [self.S1, self.S2, self.S3, self.S4]
            stats_names += ["S1", "S2", "S3", "S4"]

        if self.compute_PS:
            stats += [self.PS]
            stats_names += ["PS"]

        if mean_along_batch:
            stats = [bk.mean(s, dim=0, keepdim=True) for s in stats]
        # Flatten each, remove NaNs, concat
        flattened_list = []
        for S, S_name in zip(stats, stats_names):
            if flatten_complex and bk.is_complex(S):

                S = bk.view_as_real(
                    S
                )  # [..., 2] with last dimension for real and imag parts

                if S_name in ["S1", "S2", "PS"]:
                    S = S[..., 0]  # Keep only real part

            S_flat = S.reshape(-1) if not keep_batch_dim else S.reshape(S.shape[0], -1)

            # Remove NaNs if specified
            if not keepnans:
                nan_mask = bk.isnan(S_flat)

                if not keep_batch_dim:
                    flattened_list.append(S_flat[~nan_mask])
                else:
                    # Check that all batch elements have NaNs in the same positions
                    assert bk.all(nan_mask == nan_mask[0]), (
                        "NaNs must be at the same indices across all batch elements "
                        "when keep_batch_dim=True and keepnans=False"
                    )
                    flattened_list.append(S_flat[:, ~nan_mask[0]])
            else:
                flattened_list.append(S_flat)

        # Concatenate all statistics into a single 1D vector (or 2D if keep_batch_dim is True)
        st_flatten = (
            bk.cat(flattened_list, dim=0)
            if not keep_batch_dim
            else bk.cat(flattened_list, dim=1)
        )

        # Optional mask after nan-removal
        if mask_st is not None:
            mask_st = bk.as_tensor(mask_st, dtype=bk.bool, device=st_flatten.device)

            if not keep_batch_dim:
                if mask_st.numel() != st_flatten.numel():
                    raise ValueError(
                        f"mask_st length ({mask_st.numel()}) does not match "
                        f"flattened statistic length ({st_flatten.numel()})."
                    )
                st_flatten = st_flatten[mask_st]

            else:
                assert bk.all(mask_st == mask_st[0]), (
                    "mask_st must be identical across all batch elements "
                    "when keep_batch_dim=True"
                )
                st_flatten = st_flatten[:, mask_st[0]]

        self.st_flatten = st_flatten

        return st_flatten

    ########################################
    def select(self, param):
        """
        Select and give tensor in output

        Parameters
        ----------
        -

        Output
        ----------
        -

        """

        output = 1

        return output

    ########################################
    def _to_np(self, x):
        if isinstance(x, bk.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    ########################################
    def plot_coeff(self, b: int = 0, c: int = 0, new_figure: bool = True):
        """
        Reproduce the classical S1/S2/S3/S4 scattering plot for a given (batch, channel).

        Parameters
        ----------
        b : int
            Batch index (0 <= b < Nb).
        c : int
            Channel index (0 <= c < Nc).
        new_figure : bool
            If True, create a new figure. If False, plot into the existing one
            (useful to overlay multiple ST_statistic objects on the same panels).
        """

        # ---- extract S1..S4 for one (b,c) and convert to numpy ----
        def to_np(x):
            if isinstance(x, bk.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        if self.S1 is None:
            raise ValueError("S1 is None; nothing to plot.")

        S1_bc = to_np(self.S1[b, c])  # (J, L)
        S2_bc = to_np(self.S2[b, c])  # (J, L)
        S3_bc = to_np(self.S3[b, c])  # (J, J, L, L)
        S4_bc = to_np(self.S4[b, c])  # (J, J, J, L, L, L)

        # Put a fake 'image' dimension of size 1 to match the old plotting code
        S1 = S1_bc[None, ...]  # (1, J, L)
        S2 = S2_bc[None, ...]  # (1, J, L)

        # ---- build the compact index arrays for S3 and S4 as in your old code ----
        J = S1.shape[1]
        N_orient = S1.shape[2]

        # count combinations for S3 and S4
        n_s3 = 0
        n_s4 = 0
        for j3 in range(J):
            for j2 in range(j3 + 1):
                n_s3 += 1
                for j1 in range(j2 + 1):
                    n_s4 += 1

        j1_s3 = np.zeros(n_s3, dtype=int)
        j2_s3 = np.zeros(n_s3, dtype=int)

        j1_s4 = np.zeros(n_s4, dtype=int)
        j2_s4 = np.zeros(n_s4, dtype=int)
        j3_s4 = np.zeros(n_s4, dtype=int)

        n_s3 = 0
        n_s4 = 0
        for j3 in range(J):
            for j2 in range(0, j3 + 1):
                j1_s3[n_s3] = j2
                j2_s3[n_s3] = j3
                n_s3 += 1
                for j1 in range(0, j2 + 1):
                    j1_s4[n_s4] = j1
                    j2_s4[n_s4] = j2
                    j3_s4[n_s4] = j3
                    n_s4 += 1

        # Now we build compact S3 and S4 arrays with shape
        #   S3: (1, n_s3, L, L)
        #   S4: (1, n_s4, L, L, L)
        S3 = np.zeros((1, len(j1_s3), N_orient, N_orient), dtype=S3_bc.dtype)
        S4 = np.zeros((1, len(j1_s4), N_orient, N_orient, N_orient), dtype=S4_bc.dtype)

        for idx in range(len(j1_s3)):
            j1 = j1_s3[idx]
            j2 = j2_s3[idx]
            S3[0, idx, :, :] = S3_bc[j1, j2, :, :]

        for idx in range(len(j1_s4)):
            j1 = j1_s4[idx]
            j2 = j2_s4[idx]
            j3 = j3_s4[idx]
            S4[0, idx, :, :, :] = S4_bc[j1, j2, j3, :, :, :]

        # ---- now we reproduce your original plot_scat(S1,S2,S3,S4) ----
        color = ["b", "r", "orange", "pink"]
        symbol = ["", ":", "-", "."]

        if new_figure:
            plt.figure(figsize=(16, 12))

        # ----- S1 -----
        plt.subplot(2, 2, 1)
        for k in range(min(4, N_orient)):
            plt.plot(S1[0, :, k], color=color[k % len(color)], label=rf"$\Theta = {k}$")
        plt.legend(frameon=False, ncol=2)
        plt.xlabel(r"$J_1$")
        plt.ylabel(r"$S_1$")
        plt.yscale("log")

        # ----- S2 -----
        plt.subplot(2, 2, 2)
        for k in range(min(4, N_orient)):
            plt.plot(S2[0, :, k], color=color[k % len(color)], label=rf"$\Theta = {k}$")
        plt.xlabel(r"$J_1$")
        plt.ylabel(r"$S_2$")
        plt.yscale("log")

        # ----- S3 -----
        plt.subplot(2, 2, 3)
        # nidx to separate groups of constant j1
        nidx = np.concatenate(
            [np.zeros([1], dtype=int), np.cumsum(np.bincount(j1_s3, minlength=J))],
            axis=0,
        )
        l_pos = []
        l_name = []
        for i in np.unique(j1_s3):
            idx = np.where(j1_s3 == i)[0]
            for k in range(min(4, N_orient)):
                for l in range(min(4, N_orient)):
                    if i == 0:
                        plt.plot(
                            j2_s3[idx] + nidx[i],
                            S3[0, idx, k, l],
                            symbol[l % len(symbol)],
                            color=color[k % len(color)],
                            label=rf"$\Theta = {k},{l}$",
                        )
                    else:
                        plt.plot(
                            j2_s3[idx] + nidx[i],
                            S3[0, idx, k, l],
                            symbol[l % len(symbol)],
                            color=color[k % len(color)],
                        )
            l_pos += list(j2_s3[idx] + nidx[i])
            l_name += [f"{j1_s3[m]},{j2_s3[m]}" for m in idx]

        plt.legend(frameon=False, ncol=2)
        plt.xticks(l_pos, l_name, fontsize=6)
        plt.xlabel(r"$j_{1},j_{2}$", fontsize=9)
        plt.ylabel(r"$S_{3}$", fontsize=9)

        # ----- S4 -----
        plt.subplot(2, 2, 4)
        nidx = 0
        l_pos = []
        l_name = []
        for i in np.unique(j1_s4):
            for j in np.unique(j2_s4):
                idx = np.where((j1_s4 == i) & (j2_s4 == j))[0]
                for k in range(min(4, N_orient)):
                    for l in range(min(4, N_orient)):
                        for m in range(min(4, N_orient)):
                            if i == 0 and j == 0 and m == 0:
                                plt.plot(
                                    j2_s4[idx] + j3_s4[idx] + nidx,
                                    S4[0, idx, k, l, m],
                                    symbol[l % len(symbol)],
                                    color=color[k % len(color)],
                                    label=rf"$\Theta = {k},{l},{m}$",
                                )
                            else:
                                plt.plot(
                                    j2_s4[idx] + j3_s4[idx] + nidx,
                                    S4[0, idx, k, l, m],
                                    symbol[l % len(symbol)],
                                    color=color[k % len(color)],
                                )
                l_pos += list(j2_s4[idx] + j3_s4[idx] + nidx)
                l_name += [f"{j1_s4[m]},{j2_s4[m]},{j3_s4[m]}" for m in idx]
            # increment nidx to separate groups of constant j1
            sel = j1_s4 == i
            if np.any(sel):
                span = j2_s4[sel] + j3_s4[sel]
                nidx += int(np.max(span) - np.min(span) + 1)

        plt.legend(frameon=False, ncol=2)
        plt.xticks(l_pos, l_name, fontsize=6, rotation=90)
        plt.xlabel(r"$j_{1},j_{2},j_{3}$", fontsize=9)
        plt.ylabel(r"$S_{4}$", fontsize=9)

        plt.tight_layout()
        plt.show()
