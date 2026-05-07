# -*- coding: utf-8 -*-
"""
Main structure of STL

Tentative proposal by EA
"""

import numpy as np
import torch

import STL_main.torch_backend as bk  # from_numpy, zeros, ones, dim, shape, nan, eye
from STL_main.ST_Statistics import ST_Statistics

###############################################################################
###############################################################################


class ST_Operator:
    """
    Class whose instances correspond to scattering transforms operators.
    The operator is built through __init__ method.
    The operator is applied through apply method.
    This operator is DT-independent, and call sub-functions with common
    I/O structure, which in turn rely on DT-dependent backend.

    When the ST operator is applied to some data, it creates an instance of the
    ST statistics where all necessary parameters are passed such that the ST
    operator that was used in the computation can be reconstructed from it if
    necessary.

    To allow that, a default setting for all parameters used for the apply
    method can be stored in the ST operator.

    A prescription is also given on the order of which the different
    normalizations/compression can be done:
        norm -> iso -> angular_ft -> scale_ft -> flatten (mask_st)
    Not every transform can be used, but the ordering should be respected.
    For instance:
        vanilla -> norm -> angular_ft -> flatten (mask_st)
        vanilla -> iso -> scale_ft
    This allow the operators to be defined in a unique way from these
    parameters.

    Mask can be stored in the operator.

    Parameters
    ----------
    # Data and Wavelet Transform
    - data : instance of some STL_Data_Class
        Data (1d, 2d planar, HealPix, 3d) ##################################################
    - J : int
        number of scales
    - L : int
        number of orientations
    - WType : str
        type of wavelets

    # Scattering Transform
    - SC : str
        type of ST coefficients ("ScatCov", "WPH")
    - has_fewer_convolutions : bool
        For "ScatCov" type, whether the S3 and S4 coefficients are computed with one convolution less (Sihao version)

    # Additional transform/compression
    - norm : str
        type of norm (“self”, “from_ref”)
    - S2_ref_sqrt_chan_diag : array
        array of reference S2 coefficients (square root of the diagonal over channels)
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

    # Power spectrum computation
    - PS : bool
        whether to compute power spectrum coefficients in addition to ST statistics
    - PS_ref : array
        array of reference PS coefficients

    Attributes
    ----------
    - parent parameters (see above)
    - wavelet_op : Wavelet_Transform class
        Wavelet Transform operator

    """

    ########################################
    def __init__(
        self,
        data_example,
        J=None,
        L=None,
        SC="ScatCov",
        has_fewer_convolutions=False,
        replace_nan_value=bk.nan,
        norm="store_ref",
        S2_ref_sqrt_chan_diag=None,
        iso=False,
        angular_ft=False,
        scale_ft=False,
        flatten=False,
        mask_st=None,
        dj=None,
        harmonics_angle=None,
        harmonics_scale=None,
        # Optional wavelet operator args
        WType=None,
        downsample_nan_weight_threshold=None,
        get_crop_border_size_method=None,
        # Power spectrum computation
        compute_PS=False,
        PS_ref=None,
    ):
        """
        Constructor, see details above.
        """
        # Main parameters
        self.DT = data_example.DT

        # Wavelet transform and related parameters
        wavelet_op_kwargs = {}
        if WType is not None:
            wavelet_op_kwargs["WType"] = WType
        if downsample_nan_weight_threshold is not None:
            wavelet_op_kwargs["downsample_nan_weight_threshold"] = (
                downsample_nan_weight_threshold
            )
        if get_crop_border_size_method is not None:
            wavelet_op_kwargs["get_crop_border_size_method"] = (
                get_crop_border_size_method
            )

        self.wavelet_op = data_example.get_wavelet_op(
            J=J, L=L, **wavelet_op_kwargs
        )  # Wavelet_Operator(DT, N0, J, L, WType)
        self.J = self.wavelet_op.J
        self.L = self.wavelet_op.L
        self.WType = self.wavelet_op.WType

        # ── Explicit kernel storage (no on-the-fly build during apply) ────────
        # Wavelet kernels: list[J] of [1, L, K, K] tensors, one per scale.
        if hasattr(self.wavelet_op, '_decimated_kernels'):
            self.kernels = list(self.wavelet_op._decimated_kernels)
            self.j_to_dg = list(self.wavelet_op.j_to_dg)
        else:
            self.kernels  = None
            self.j_to_dg  = None
        # Smooth (anti-aliasing) kernels for downsampling, both boundary modes.
        if hasattr(self.wavelet_op, 'smooth_kernel_pbc'):
            self.smooth_kernel_pbc   = self.wavelet_op.smooth_kernel_pbc
            self.smooth_kernel_nopbc = self.wavelet_op.smooth_kernel_nopbc
        else:
            self.smooth_kernel_pbc   = None
            self.smooth_kernel_nopbc = None

        # Scattering transform related parameters
        self.SC = SC
        self.has_fewer_convolutions = has_fewer_convolutions
        self.replace_nan_value = replace_nan_value

        # Additional transform/compression related parameters
        self.norm = norm
        self.S2_ref_sqrt_chan_diag = S2_ref_sqrt_chan_diag
        self.iso = iso
        self.angular_ft = angular_ft
        self.scale_ft = scale_ft
        self.flatten = flatten
        self.mask_st = mask_st

        self.dj = dj
        self.harmonics_angle = harmonics_angle
        self.harmonics_scale = harmonics_scale

        # Power spectrum computation
        if compute_PS:
            self.PS_op = data_example.get_PS_op()  # PowerSpectrum_Operator(N0)
        self.compute_PS = compute_PS
        self.PS_ref = PS_ref

    ########################################
    @classmethod
    def from_ST_Statistics(self, st_stat):
        """
        Alternative constructor, which generates the ST operator used to
        compute a given set of ST statistics.

        Parameters
        ----------
        - st_stat : ST_Statistics
            st_stat instance whose parameters have to be reproduced

        Remark and to do
        ----------
        - In fact, a ST_Statistics instance cannot transmit the flatten
        parameter, since it would have return a 1D array. This is not clear
        for me how to deal with this point.

        """
        raise NotImplementedError
        return ST_Operator(
            st_stat.DT,
            J=st_stat.J,
            L=st_stat.L,
            WType=st_stat.WType,
            SC=st_stat.SC,
            norm=st_stat.norm,
            S2_ref_sqrt_chan_diag=st_stat.S2_ref_sqrt_chan_diag,
            iso=st_stat.iso,
            angular_ft=st_stat.angular_ft,
            scale_ft=st_stat.scale_ft,
            flatten=st_stat.flatten,
            mask_st=st_stat.mask_st,
        )

    ########################################
    def apply(
        self,
        data,
        SC=None,
        has_fewer_convolutions=None,
        norm=None,
        S2_ref_sqrt_chan_diag=None,
        iso=None,
        angular_ft=None,
        scale_ft=None,
        flatten=None,
        mask_st=None,
        compute_PS=None,
        PS_ref=None,
        compute_cross_matrix=None,
    ):
        """
        Compute the Scattering Transform (ST) of data, which are either stored
        in an instance of the ST statistics class, or returned as a flatten
        array.

        This DT-independent methods calls sub-functions which have a common
        I/O structure, and in turn rely on DT-dependent backend.

        It outputs an instance of the Scattering Statistics class, whose
        additional methods can be called directly to get the desired output.

        Uses ST operator parameters unless explicitly overridden in apply.

        !!! Attention: I give an example in torch here, but we should consider
        how to include different backend !!!

        !!! Attention: I give here the version with standard scat cov !!!

        Parameters
        ----------
        # Data
        - data : StlData with MR=False, dim (N) or (Nc, N) or (Nb, Nc, N)
            data, Nc number of channel, Nb batch size. Should have dg=0.

        # Scattering Transform
        - SC : str
            type of ST coefficients ("ScatCov", "WPH")
        - has_fewer_convolutions : bool
            For "ScatCov" type, whether the S3 and S4 coefficients are computed with one convolution less (Sihao version)
        - pass_mask : bool
            Pass mask to ST statistics object if True

        # Additional transform/compression
        - norm : str
            type of norm ("self", "from_ref")
        - S2_ref_sqrt_chan_diag : array
            array of reference S2 coefficients (square root of the diagonal over channels)
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

        # Power spectrum computation
        - compute_PS : bool
            whether to compute power spectrum coefficients in addition to ST statistics

        # Cross statistics computation
        - compute_cross_matrix : ndarray of bool (Default: None which is auto-statistics only)
            Upper triangular matrix with shape (Nc,Nc), which determines pairs of channels for which to compute cross-statistics.
            More precisely:
                - computes S1(c1), S2(c1,c1), S3(c1,c1) and S4(c1,c1) if and only if compute_cross_matrix[c1,c1] == True
                - for c1 < c2, computes S2(c1,c2), S3(c1,c2), S3(c2,c1), S4(c1,c2) and S4(c2,c1) if and only if compute_cross_matrix[c1,c2] == True
                - for c1 > c2, compute_cross_matrix[c1,c2] is ignored and should not be specified
            If None, it is replaced by a boolean matrix full of True, so that all cross-statistics are computed.


        Output
        ----------
        - data_st : ST_Statistics instance, or 1D array
            ST statistics of I, as a flatten array if flatten=True
        """

        ########################################
        # General Initialization
        ########################################

        # Consistency checks
        if getattr(self.wavelet_op, "N0", data.N0) != data.N0:
            raise Exception(
                "Wavelet operator of the scattering operator and data should have same N0"
            )

        # Local value for the wavelet transform parameters
        N0 = data.N0
        J = self.J
        L = self.L
        WType = self.wavelet_op.WType

        # Local value for the scattering transform parameters
        SC = self.SC if SC is None else SC
        has_fewer_convolutions = (
            self.has_fewer_convolutions
            if has_fewer_convolutions is None
            else has_fewer_convolutions
        )

        # Local value for the additional transforms parameters
        norm = self.norm if norm is None else norm
        S2_ref_sqrt_chan_diag = (
            self.S2_ref_sqrt_chan_diag
            if S2_ref_sqrt_chan_diag is None
            else S2_ref_sqrt_chan_diag
        )
        iso = self.iso if iso is None else iso
        angular_ft = self.angular_ft if angular_ft is None else angular_ft
        scale_ft = self.scale_ft if scale_ft is None else scale_ft
        flatten = self.flatten if flatten is None else flatten
        mask_st = self.mask_st if mask_st is None else mask_st

        compute_PS = self.compute_PS if compute_PS is None else compute_PS
        PS_ref = self.PS_ref if PS_ref is None else PS_ref

        # Put in torch or relevant bk
        if type(data.array) == np.ndarray:
            data.array = bk.from_numpy(data.array)

        # Put in the expected size: (Nb, Nc, N)
        N_DT = len(N0)
        if data.array.dim() == N_DT:
            data.array = data.array[None, None, ...]  # (1,1,N)
        elif data.array.dim() == N_DT + 1:
            data.array = data.array[None, ...]  # (1,Nc,N)
        Nb, Nc = data.array.shape[0], data.array.shape[1]

        compute_cross_matrix = (
            bk.ones((Nc, Nc), dtype=bool, device=data.device)
            if compute_cross_matrix is None
            else compute_cross_matrix.to(device=data.device)
        )

        # Create a ST_statistics instance
        data_st = ST_Statistics(
            self.DT,
            N0,
            J,
            L,
            WType,
            SC,
            Nb,
            Nc,
            self.wavelet_op,
            compute_PS,
        )

        # Initialize ST statistics values
        # Add readability w.r.t. having it in the ST statistics initilization
        l_data = data.copy()

        data_st.mean = self.wavelet_op.mean(l_data)  # [Nb,Nc]
        data_st.var = self.wavelet_op.cov(l_data, l_data)  # [Nb,Nc]

        if compute_PS:
            data_st.PS = self.PS_op.apply(l_data)

        if SC == "ScatCov":
            data_st.S1 = bk.zeros((Nb, Nc, J, L)) + bk.nan
            data_st.S2 = bk.zeros((Nb, Nc, Nc, J, L)) + bk.nan
            data_st.S3 = (
                bk.zeros((Nb, Nc, Nc, J, J, L, L), dtype=bk._DEFAULT_COMPLEX_DTYPE)
                + bk.nan
            )
            import torch

            data_st.S4 = (
                bk.zeros(
                    (Nb, Nc, Nc, J, J, J, L, L, L), dtype=bk._DEFAULT_COMPLEX_DTYPE
                )
                + bk.nan
            )

            channels_with_auto_stats = compute_cross_matrix.diagonal()
            for channel in range(len(channels_with_auto_stats)):
                if not channels_with_auto_stats[channel]:
                    if not (
                        compute_cross_matrix[channel, channel + 1 :].any()
                        or compute_cross_matrix[:channel, channel].any()
                    ):
                        # If no auto-statistics are asked for this channel, we require it to appear in at least one cross-statistics
                        raise Exception(
                            f"Channel {channel} auto-statistics are not demanded and does not appear in any cross-statistics neither.\nPlease remove it or constrain at least its auto-statistics or one of its cross-statistics."
                        )

        ########################################
        # ST coefficients computation
        ########################################

        # Vanilla version uses the following form for S3 and S4
        # S3 = Cov(|I*psi1|*psi2, I*psi2)
        # S4 = Cov(|I*psi1|*psi3, |I*psi2|*psi3)

        # WARNING !! This is the version coded by JMD, that should be correct
        # for all DataTypes

        # See at the bottom of this file for the previous versions developped
        # for FFT

        data_l1m = {}
        ### Higher order computation ###

        for j3 in range(J):
            # Compute first convolution and modulus
            data_l1 = self.wavelet_op.apply(
                l_data, j=j3,
                kernel=self.kernels[j3] if self.kernels is not None else None,
            )  # (Nb,Nc,L,N3)
            data_l1m[j3] = data_l1.modulus(inplace=False)  # (Nb,Nc,L,N3)

            if False and self.wavelet_op.mask_full_res is not None:
                import torch

                assert torch.all(
                    data_l1m[j3].array.isnan() == self.layer1_mask[j3].array
                )  ###################### GOOD TO KEEP WHILE DEBUGGING with replace_nan_value=torch.nan

            ##############################################################################
            ########################## S1(j3) = Mean(|I*psi3|) ###########################
            ##############################################################################
            data_st.S1[:, channels_with_auto_stats, j3, :] = self.wavelet_op.mean(
                data_l1m[j3][:, channels_with_auto_stats, :, :],
            )  # (Nb,Nc,L3)

            ##############################################################################
            ######################### S2(j3) = Mean(|I*psi3|^2) ##########################
            ##############################################################################
            # auto S2 terms
            data_st.S2[:, channels_with_auto_stats, channels_with_auto_stats, j3, :] = (
                self.wavelet_op.square_mean(
                    data_l1m[j3][:, channels_with_auto_stats, :, :, :]
                )
            )  # (Nb, Nc, Nc, L3)

            # cross S2 terms (sub diagonal only)
            self.wavelet_op._compute_and_store_cross_cov(
                data_l1,
                data_l1,
                output=data_st.S2[:, :, :, j3, :],
                compute_cross_matrix=compute_cross_matrix
                * (
                    ~bk.eye(Nc, dtype=bool, device=data.device)
                ),  # remove diagonal wich was computed above with real mean square
                redundant_channels=True,  # S2(c1,c2) and S2(c2,c1) are conjugates
            )  # (Nb,Nc,Nc,L3)

            data_l1m_l2 = {}
            for j2 in range(j3 + 1):
                data_l1m_l2_j2 = self.wavelet_op.apply(
                    data_l1m[j2],
                    j=j3,
                    kernel=self.kernels[j3] if self.kernels is not None else None,
                )  # (Nb,Nc,L2,L3,N3)

                if False and self.wavelet_op.mask_full_res is not None:
                    assert torch.all(
                        data_l1m_l2_j2.array.isnan() == self.layer2_mask[j3][j2].array
                    )  ###################### GOOD TO KEEP WHILE DEBUGGING with replace_nan_value=torch.nan
                    assert torch.all(
                        self.layer2_mask[j3][j2].array >= self.layer1_mask[j3].array
                    )  ###################### sanity check to make sure mask for |I*psi2|*psi3 contains the one for I*psi3

                ##############################################################################
                ################### S3(j2,j3) = Cov(|I*psi2|*psi3, I*psi3) ###################
                ##############################################################################
                if not has_fewer_convolutions:
                    self.wavelet_op._compute_and_store_cross_cov(
                        data_l1m_l2_j2,
                        data_l1[:, :, None],
                        output=data_st.S3[:, :, :, j2, j3, :, :],
                        compute_cross_matrix=compute_cross_matrix,
                        redundant_channels=False,
                    )  # (Nb,Nc,Nc,L2,L3)

                else:
                    # Sihao S3 version : S3(j1,j2,j3) = Cov(I, |I*psi2|*psi3)
                    self.wavelet_op._compute_and_store_cross_cov(
                        l_data[:, :, None, None, :, :],  # [Nb,Nc,1,1,N3]
                        data_l1m_l2_j2,  # [Nb,Nc,L2,L3,N3]
                        output=data_st.S3[:, :, :, j2, j3, :, :],
                        compute_cross_matrix=compute_cross_matrix,
                        redundant_channels=False,
                    )  # [Nb, Nc, Nc, L2, L3]

                data_l1m_l2[j2] = data_l1m_l2_j2  # (Nb,Nc,L2,L3,N3)

                for j1 in range(j2 + 1):
                    ##############################################################################
                    ############## S4(j1,j2,j3) = Cov(|I*psi1|*psi3, |I*psi2|*psi3) ##############
                    ##############################################################################
                    if not has_fewer_convolutions:
                        self.wavelet_op._compute_and_store_cross_cov(
                            data_l1m_l2[j1][:, :, :, None],
                            data_l1m_l2[j2][:, :, None, :],
                            output=data_st.S4[:, :, :, j1, j2, j3, :, :, :],
                            compute_cross_matrix=compute_cross_matrix,
                            redundant_channels=False,
                        )  # (Nb,Nc,Nc,L1,L2,L3)

                    else:
                        # Sihao S4 version : S4(j1,j2,j3) = Cov(|I*psi1|, |I*psi2|*psi3)
                        self.wavelet_op._compute_and_store_cross_cov(
                            data_l1m[j1][
                                :, :, :, None, None, :, :
                            ],  # [Nb,Nc,L1,1,1,N3]
                            data_l1m_l2[j2][
                                :, :, None, :, :, :, :
                            ],  # [Nb,Nc,1,L2,L3,N3]
                            output=data_st.S4[:, :, :, j1, j2, j3, :, :, :],
                            compute_cross_matrix=compute_cross_matrix,
                            redundant_channels=False,
                        )  # (Nb,Nc,Nc,L1,L2,L3)

            # Downsample at Nj3
            if j3 < J - 1:
                _sk = (
                    (self.smooth_kernel_pbc if l_data.pbc else self.smooth_kernel_nopbc)
                    if self.smooth_kernel_pbc is not None else None
                )
                self.wavelet_op.downsample(
                    data=l_data,
                    dg_out=self.wavelet_op.j_to_dg[j3 + 1],
                    inplace=True,
                    replace_nan_value=self.replace_nan_value,
                    smooth_kernel=_sk,
                )  # (Nb,Nc,j3+1,L,N3)

                for j2 in range(j3 + 1):
                    self.wavelet_op.downsample(
                        data=data_l1m[j2],
                        dg_out=self.wavelet_op.j_to_dg[j3 + 1],
                        inplace=True,
                        replace_nan_value=self.replace_nan_value,
                        smooth_kernel=_sk,
                    )  # (Nb,Nc,j3+1,L,N3)

        """
        # Version to compute ST statistics for STL_FFT_Torch from fullJ mode 

        # --- Compute first convolution and modulus ---
        print(self.wavelet_op.wavelet_array.shape)
        data_l1 = self.wavelet_op.apply(data, target_fourier_status=False)  # (Nb,Nc,J,L,N)
        data_l1m = data_l1.modulus(inplace=True)  # (Nb,Nc,J,L,N)

        # --- Compute S1 and S2 ---
        data_st.S1 = self.wavelet_op.mean(data_l1m) # (Nb,Nc,J,L)
        data_st.S2 = self.wavelet_op.mean(data_l1m, square=True)  # (Nb,Nc,J,L)  

        for j3 in range(J):
            data_l1_tmp = data_l1.copy()  # (Nb,Nc,j3+1,L,N)
            data_l1m_tmp = data_l1m.copy()
            # (Nb,Nc,j3+1,L,N)
            data_l1_tmp.array = data_l1_tmp.array[:, :, : j3 + 1]
            data_l1m_tmp.array = data_l1m_tmp.array[:, :, : j3 + 1]

            # Downsample at Nj3
            self.wavelet_op.downsample(data_l1_tmp, j3)  # (Nb,Nc,j3+1,L,N3)
            self.wavelet_op.downsample(data_l1m_tmp, j3)  # (Nb,Nc,j3+1,L,N3)

            # Compute |I*psi2|*psi3                      #(Nb,Nc,j3+1,L2,L3,N3)
            data_l1m_l2 = self.wavelet_op.apply(data_l1m_tmp, j=j3)

            for j2 in range(j3 + 1):
                # S3(j2,j3) = Cov(|I*psi2|*psi3, I*psi3)
                data_st.S3[:, :, j2, j3, :, :] = self.wavelet_op.cov(
                    data_l1m_l2[:, :, j2],
                    data_l1_tmp[:, :, j3, None]
                )  # (Nb,Nc,L2,L3,N3) x (Nb,Nc,1,L3,N3)

                for j1 in range(j2 + 1):
                    # S4(j1,j2,j3) = Cov(|I*psi1|*psi3, |I*psi2|*psi3)
                    data_st.S4[:, :, j1, j2, j3, :, :, :] = self.wavelet_op.cov(
                        data_l1m_l2[:, :, j1, :, None],
                        data_l1m_l2[:, :, j2, None, :]
                    )  # (Nb,Nc,L1, 1,L3,N3) x (Nb,Nc, 1,L2,L3,N3)

        """

        ########################################
        # Additional transform/compression
        ########################################
        # Normalisation
        if norm == "vanilla":
            pass
        elif norm == "store_ref":
            if SC == "ScatCov" and self.S2_ref_sqrt_chan_diag is not None:
                print("Replacing existing S2_ref_sqrt_chan_diag in ST_Op")
            if compute_PS and self.PS_ref is not None:
                print("Replacing existing PS_ref in ST_Op")
            data_st.to_norm(norm_type="self")
            if SC == "ScatCov":
                self.S2_ref_sqrt_chan_diag = data_st.S2_ref_sqrt_chan_diag
            if compute_PS:
                self.PS_ref = data_st.PS_ref

        elif norm == "load_ref":
            if SC == "ScatCov" and S2_ref_sqrt_chan_diag is None:
                raise Exception(
                    "S2_ref_sqrt_chan_diag should be stored in the ST_Operator"
                )
            if compute_PS and PS_ref is None:
                raise Exception("PS_ref should be stored in the ST_Operator")

            kwargs = {}
            if SC == "ScatCov":
                kwargs["S2_ref_sqrt_chan_diag"] = self.S2_ref_sqrt_chan_diag
            if compute_PS:
                kwargs["PS_ref"] = self.PS_ref

            # Appel avec seulement les bons arguments
            data_st.to_norm(norm_type="from_ref", **kwargs)

        if iso:
            data_st.to_iso()

        if angular_ft:
            data_st.to_angular_ft(self.harmonics_angle)

        if scale_ft:
            data_st.to_scale_ft(self.harmonics_scale, self.dj, self.harmonics_angle)

        if flatten:
            data_st.to_flatten(mask_st)

        return data_st
