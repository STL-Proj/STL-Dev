# -*- coding: utf-8 -*-
"""
Main structure of STL

Tentative proposal by EA
"""

import warnings

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
        mask_full_res=None,
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
        compute_PS=True,
        PS_ref_sqrt_chan_diag=None,
        var_ref=None,
        # Optional wavelet operator args
        WType=None,
        downsample_nan_weight_threshold=None,
        get_crop_border_size_method=None,
        # Optional power spectrum args
        n_bins=None,
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
        if mask_full_res is not None:
            wavelet_op_kwargs["mask_full_res"] = mask_full_res
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

        # Scattering transform related parameters
        self.SC = SC
        self.has_fewer_convolutions = has_fewer_convolutions
        self.replace_nan_value = replace_nan_value

        # Additional transform/compression related parameters
        assert norm in [
            None,
            "vanilla",
            "store_ref",
            "load_ref",
        ], "Invalid norm type. Should be one of [None, 'vanilla', 'store_ref', 'load_ref']"
        self.norm = norm
        self.S2_ref_sqrt_chan_diag = S2_ref_sqrt_chan_diag
        self.var_ref = var_ref
        self.iso = iso
        self.angular_ft = angular_ft
        self.scale_ft = scale_ft
        self.flatten = flatten
        self.mask_st = mask_st

        self.dj = dj
        self.harmonics_angle = harmonics_angle
        self.harmonics_scale = harmonics_scale

        # Power spectrum computation
        self.compute_PS = compute_PS
        cross_spectrum_op_kwargs = {}
        if n_bins is not None:
            cross_spectrum_op_kwargs["n_bins"] = n_bins
        if J is not None:
            cross_spectrum_op_kwargs["J"] = J
        self.CS_op = data_example.get_CS_op(**cross_spectrum_op_kwargs)
        self.n_bins = self.CS_op.n_bins
        self.PS_ref_sqrt_chan_diag = PS_ref_sqrt_chan_diag

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
        standardize=False,
        SC=None,
        has_fewer_convolutions=None,
        norm=None,
        S2_ref_sqrt_chan_diag=None,
        norm_batch_mean=False,
        iso=None,
        angular_ft=None,
        scale_ft=None,
        flatten=None,
        mask_st=None,
        compute_PS=None,
        PS_ref_sqrt_chan_diag=None,
        var_ref=None,
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
        - compute_PS : bool, default True
            whether to compute power spectrum coefficients in addition to ST statistics
        - PS_ref_sqrt_chan_diag : array
            array of reference PS coefficients

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

        # Consistency check
        # TODO: find a way to retrieve J related to the data without loading the whole wavelet operator
        # data_J = data.get_wavelet_op().J
        # if self.J > data_J:
        #     raise ValueError(
        #         f"Incompatible J: ST operator initialized with J={self.J}, "
        #         f"but data only supports J up to {data_J}."
        #     )

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
        if norm is None and self.norm is None:
            raise Exception(
                f"Norm type should be one of ['vanilla', 'store_ref', 'load_ref'] but is not specified in the ST operator initialization nor in the apply method. If you don't want to apply any normalization, please set norm to 'vanilla'."
            )
        assert norm in [
            "vanilla",
            "store_ref",
            "load_ref",
        ], "Invalid norm type. Should be one of ['vanilla', 'store_ref', 'load_ref']."
        if norm == "store_ref":
            assert (
                var_ref is None
            ), "var_ref should not be provided when norm='store_ref'"
            if SC == "ScatCov":
                assert (
                    S2_ref_sqrt_chan_diag is None
                ), "S2_ref_sqrt_chan_diag should not be provided when norm='store_ref'"
            if compute_PS:
                assert (
                    PS_ref_sqrt_chan_diag is None
                ), "PS_ref_sqrt_chan_diag should not be provided when norm='store_ref'"
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
        PS_ref_sqrt_chan_diag = (
            self.PS_ref_sqrt_chan_diag
            if PS_ref_sqrt_chan_diag is None
            else PS_ref_sqrt_chan_diag
        )

        var_ref = self.var_ref if var_ref is None else var_ref

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
            torch.triu(bk.ones((Nc, Nc), dtype=bool, device=data.device))
            if compute_cross_matrix is None
            else compute_cross_matrix.to(device=data.device)
        )

        # Initialize ST statistics values
        # Add readability w.r.t. having it in the ST statistics initilization
        if standardize:
            standardized = True
            l_data, mean_pre_std, std_pre_std = self.wavelet_op.standardize(
                data, mean_field=False, inplace=False
            )
        else:
            l_data = data.copy()
            standardized = False
            mean_pre_std, std_pre_std = None, None

        # Create a ST_statistics instance
        data_st = ST_Statistics(
            data.__class__,
            N0,
            Nb,
            Nc,
            self.wavelet_op,
            SC,
            has_fewer_convolutions,
            compute_cross_matrix,
            compute_PS,
            self.n_bins,
            standardized,
            mean_pre_std,
            std_pre_std,
        )

        # Systematic statistics (data supposed to be real)
        assert (
            data.array.is_complex() == False
        ), "Data should be real for now, otherwise mean and var computation should be adapted"
        data_st.mean = self.wavelet_op.mean(l_data).real  # [Nb,Nc]
        data_st.var = self.wavelet_op.cov(l_data, l_data).real  # [Nb,Nc]

        if compute_PS:
            data_st.PS = self.CS_op.apply(
                l_data,
                compute_cross_spectrum_matrix=compute_cross_matrix,  # Same channel pairs as for ST cross-statistics
            )

        if SC == "ScatCov":
            # data_st.S1 = bk.zeros((Nb, Nc, J, L)) + bk.nan
            data_st.S1 = (
                bk.zeros((Nb, Nc, Nc, J, L), dtype=bk._DEFAULT_COMPLEX_DTYPE) + bk.nan
            )
            data_st.S2 = (
                bk.zeros((Nb, Nc, Nc, J, L), dtype=bk._DEFAULT_COMPLEX_DTYPE) + bk.nan
            )
            data_st.S3 = (
                bk.zeros((Nb, Nc, Nc, J, J, L, L), dtype=bk._DEFAULT_COMPLEX_DTYPE)
                + bk.nan
            )

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
            data_l1 = self.wavelet_op.apply(l_data, j=j3)  # (Nb,Nc,L,N3)
            data_l1m[j3] = data_l1.modulus(inplace=False)  # (Nb,Nc,L,N3)

            if False and self.wavelet_op.mask_full_res is not None:

                assert torch.all(
                    data_l1m[j3].array.isnan() == self.layer1_mask[j3].array
                )  ###################### GOOD TO KEEP WHILE DEBUGGING with replace_nan_value=torch.nan

            ##############################################################################
            ########################## S1(j3) = Mean(|I*psi3|) ###########################
            ##############################################################################
            #            data_st.S1[:, channels_with_auto_stats, j3, :] = self.wavelet_op.mean(
            #                data_l1m[j3][:, channels_with_auto_stats, :, :],
            #            )  # (Nb,Nc,L3)

            # auto S1 terms
            data_st.S1[
                :, channels_with_auto_stats, channels_with_auto_stats, j3, :
            ] = self.wavelet_op.mean(
                data_l1m[j3][:, channels_with_auto_stats, :, :, :],
            ).to(
                dtype=data_st.S1.dtype  # cast to complex if needed
            )  # (Nb,Nc,Nc,L3)

            # cross S1 terms (sub diagonal only)
            if (
                compute_cross_matrix * (~bk.eye(Nc, dtype=bool, device=data.device))
            ).any():
                data_l1_modulus_square_rooted = data_l1.divide(
                    data_l1m[j3], epsilon=1e-8, pow=0.5, inplace=False
                )  # [Nb,Nc,L,N3]

                self.wavelet_op._compute_and_store_cross_cov(
                    data_l1_modulus_square_rooted,
                    data_l1_modulus_square_rooted,
                    output=data_st.S1[:, :, :, j3, :],
                    compute_cross_matrix=compute_cross_matrix
                    * (
                        ~bk.eye(Nc, dtype=bool, device=data.device)
                    ),  # remove diagonal wich was computed above with real mean square
                    redundant_channels=True,  # S1(c1,c2) and S1(c2,c1) are conjugates
                )  # (Nb,Nc,Nc,L3)

            ##############################################################################
            ######################### S2(j3) = Mean(|I*psi3|^2) ##########################
            ##############################################################################
            # auto S2 terms
            data_st.S2[
                :, channels_with_auto_stats, channels_with_auto_stats, j3, :
            ] = self.wavelet_op.square_mean(
                data_l1m[j3][:, channels_with_auto_stats, :, :, :]
            ).to(
                dtype=data_st.S2.dtype  # cast to complex if needed
            )  # (Nb,Nc,Nc,L3)

            # cross S2 terms (sub diagonal only)
            if (
                compute_cross_matrix * (~bk.eye(Nc, dtype=bool, device=data.device))
            ).any():
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
                        data_l1[
                            :, :, None, :, :, :
                        ],  # (Nb,Nc,L2,L3,N3) x (Nb,Nc,1,L3,N3)
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
                            data_l1m_l2[j1][:, :, :, None, :, :],  # (Nb,Nc,L1,1,L3,N3)
                            data_l1m_l2[j2][:, :, None, :, :, :],  # (Nb,Nc,1,L2,L3,N3)
                            output=data_st.S4[
                                :, :, :, j1, j2, j3, :, :, :
                            ],  # (Nb,Nc,Nc,L1,L2,L3)
                            compute_cross_matrix=compute_cross_matrix,
                            redundant_channels=False,  # TODO: S4(c1,c2) and S4(c2,c1) are conjugates if j1==j2 and thetha1=theta2
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

                self.wavelet_op.downsample(
                    data=l_data,
                    dg_out=self.wavelet_op.j_to_dg[j3 + 1],
                    inplace=True,
                    replace_nan_value=self.replace_nan_value,
                )  # (Nb,Nc,j3+1,L,N3)

                for j2 in range(j3 + 1):
                    self.wavelet_op.downsample(
                        data=data_l1m[j2],
                        dg_out=self.wavelet_op.j_to_dg[j3 + 1],
                        inplace=True,
                        replace_nan_value=self.replace_nan_value,
                    )  # (Nb,Nc,j3+1,L,N3)

        ########################################
        # Additional transform/compression
        ########################################
        # Normalisation
        if norm == "vanilla":
            pass

        elif norm == "store_ref":
            if self.var_ref is not None:
                print("Replacing existing var_ref in ST_Op")
            if SC == "ScatCov" and self.S2_ref_sqrt_chan_diag is not None:
                print("Replacing existing S2_ref_sqrt_chan_diag in ST_Op")
            if compute_PS and self.PS_ref_sqrt_chan_diag is not None:
                print("Replacing existing PS_ref_sqrt_chan_diag in ST_Op")

            # Check if some auto-stats are not computed
            missing_auto = (~compute_cross_matrix.diagonal()).nonzero(as_tuple=True)[0]

            if len(missing_auto) > 0:
                warnings.warn(
                    f"S2 auto-stats are not computed for channels {(missing_auto + 1).tolist()}. "
                    "Using norm='store_ref' normalizes with sqrt(S2 auto-stats) and may generate NaNs "
                    "for cross-statistics involving these channels.",
                    UserWarning,
                    stacklevel=2,
                )

            data_st.to_norm(norm_type="self", norm_batch_mean=norm_batch_mean)

            self.var_ref = data_st.var_ref
            if SC == "ScatCov":
                self.S2_ref_sqrt_chan_diag = data_st.S2_ref_sqrt_chan_diag
            if compute_PS:
                self.PS_ref_sqrt_chan_diag = data_st.PS_ref_sqrt_chan_diag

        elif norm == "load_ref":
            if var_ref is None:
                raise Exception(
                    "var_ref should be stored in the ST_Operator or given in apply argument when norm='load_ref'"
                )
            if SC == "ScatCov" and S2_ref_sqrt_chan_diag is None:
                raise Exception(
                    "S2_ref_sqrt_chan_diag should be stored in the ST_Operator or given in apply argument when norm='load_ref'"
                )
            if compute_PS and PS_ref_sqrt_chan_diag is None:
                raise Exception(
                    "PS_ref_sqrt_chan_diag should be stored in the ST_Operator or given in apply argument when norm='load_ref'"
                )

            kwargs = {}
            kwargs["var_ref"] = var_ref
            if SC == "ScatCov":
                kwargs["S2_ref_sqrt_chan_diag"] = S2_ref_sqrt_chan_diag
            if compute_PS:
                kwargs["PS_ref_sqrt_chan_diag"] = PS_ref_sqrt_chan_diag

            # Appel avec seulement les bons arguments
            data_st.to_norm(
                norm_type="from_ref", norm_batch_mean=norm_batch_mean, **kwargs
            )

        else:
            raise Exception(f"Unknown norm type: {norm}.")

        if iso:
            data_st.to_iso()

        if angular_ft:
            data_st.to_angular_ft(self.harmonics_angle)

        if scale_ft:
            data_st.to_scale_ft(self.harmonics_scale, self.dj, self.harmonics_angle)

        if flatten:
            data_st.to_flatten(mask_st)

        return data_st
