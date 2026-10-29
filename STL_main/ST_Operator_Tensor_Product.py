import warnings
from itertools import product

import STL_main.torch_backend as bk  # from_numpy, zeros, ones, dim, shape, nan, eye
from STL_main.ST_Operator import ST_Operator


class ST_Operator_TensorProd(ST_Operator):
    """
    Scattering operator for tensor products of 1D and 2D blocks, with the parameters of ST_Operator and:

    - lowpass : bool, default True
        Add the low-pass filter phi_J of each block to its wavelets. Tensor products of wavelets alone do not
        see the frequencies close to 0 in any block (e.g. omega ~ 0 at any spatial frequency, the time average
        of a space-time field), since their Littlewood-Paley sum is the product of those of the blocks. In S1 and
        S2, the low-pass filter of block b is the scale j_b = J_b, after its wavelets, with a single orientation
        stored at index 0 (the other orientations are NaN). The combination with the low-pass filter in all
        blocks is not computed: the frequencies below the coarsest scale of every block are only described by the
        mean, as below the coarsest scale in 1D and 2D. S3 and S4 have no low-pass coefficients (see
        _compute_ST_loop).
        Blocks whose size is not divisible by 2**J_b (one more downsampling than the wavelets) get no low-pass
        filter.
    """

    def __init__(self, data_example, *args, lowpass=True, **kwargs):
        super().__init__(data_example, *args, **kwargs)

        # low-pass filter of each block, if its size allows one more downsampling than the wavelets
        self.lowpass = [
            lowpass
            and all(
                N % 2 ** self.wavelet_op.lowpass_dg(block_id) == 0
                for N in data_example.N0[block_id]
            )
            for block_id in range(self.wavelet_op.nblock)
        ]
        if lowpass and not all(self.lowpass):
            warnings.warn(
                f"No low-pass filter for blocks {[block_id for block_id, lowpass_block in enumerate(self.lowpass) if not lowpass_block]}, "
                f"whose size {data_example.N0} is not divisible by 2**J with J = {self.J}."
            )

    def _scales_S1_S2(self, J):
        # the low-pass filter of a block is its scale J_b, after the wavelets
        return [
            J_block + 1 if lowpass_block else J_block
            for J_block, lowpass_block in zip(J, self.lowpass)
        ]

    def _compute_ST_loop(
        self,
        l_data,
        data_st,
        J,
        Nc,
        device,
        channels_with_auto_stats,
        compute_cross_matrix,
        has_fewer_convolutions,
        verbose=False,
    ):

        ########################################
        # ST coefficients computation
        ########################################

        # Vanilla version uses the following form for S3 and S4
        # S3 = Cov(|I*psi1|*psi2, I*psi2)
        # S4 = Cov(|I*psi1|*psi3, |I*psi2|*psi3)

        # Low-pass filters (self.lowpass): the low-pass filter phi_J of block b is used as its scale j_b = J_b,
        # after its wavelets, and the multi-scales j3 run over the grid extended with these scales, except the
        # multi-scale with the low-pass filter in all blocks (not a zero-mean filter, it contains the mean).
        # S3 and S4 are expected to approximately vanish when their last filter psi3 includes a low-pass filter,
        # and are not computed. Since j1 <= j2 <= j3 componentwise and the low-pass filter is the coarsest scale of
        # its block, this excludes all S3 and S4 involving a low-pass filter: low-pass multi-scales only give S1
        # and S2. Hence:
        # - the first-layer path (l_data_multi_res, data_l1) follows all the multi-scales;
        # - the second-layer path (data_l1m, data_l1m_multi_res, data_l1m_l2), only used for S3 and S4, follows
        #   the wavelet multi-scales only: it goes from one wavelet multi-scale to the next one, skipping the
        #   low-pass multi-scales in between, exactly as without low-pass filters.

        def is_lowpass(j):
            # whether the multi-scale j includes a low-pass filter
            return any(j_block == J_block for j_block, J_block in zip(j, J))

        def dg_of(block_id, j_block):
            # resolution of the scale j_block of a block, the low-pass filter being one downsampling below the coarsest wavelet
            if j_block == J[block_id]:
                return self.wavelet_op.lowpass_dg(block_id)
            return self.wavelet_op.wavelet_ops[block_id].j_to_dg[j_block]

        l_data_multi_res = [
            None
        ] * l_data.nblock  # list of l_data for each block_id, to be used in the loop over j3
        l_data_multi_res[0] = (
            l_data  # first block_id is the original data which will be downsampled over j3x
        )
        data_l1 = {}
        data_l1m = {}
        data_l1m_multi_res = {}
        data_l1m_l2 = {}

        ### Higher order computation ###

        # J = (Jx, ..., Jz)
        J_range = [
            j3
            for j3 in product(
                *(
                    range(J_block + 1 if lowpass_block else J_block)
                    for J_block, lowpass_block in zip(J, self.lowpass)
                )
            )
            if not all(j3_block == J_block for j3_block, J_block in zip(j3, J))
        ]  # [(0, ..., 0), (0, ..., 1), ..., (Jx-1, ..., Jz-1)] lexicographic order, extended with the low-pass scales Jx, ..., Jz

        # next wavelet multi-scale after each multi-scale, for the second-layer path
        next_wavelet_j3 = {}
        next_j3_wavelet = (None,) * len(J)
        for j3 in reversed(J_range):
            next_wavelet_j3[j3] = next_j3_wavelet
            if not is_lowpass(j3):
                next_j3_wavelet = j3

        for j3, next_j3 in zip(J_range, J_range[1:] + [(None,) * len(J)]):
            (
                print(
                    f"\n****************** NEW ITERATION ******************\nj3={j3}{' (low-pass)' if is_lowpass(j3) else ''}, next_j3={next_j3}"
                )
                if verbose > 0
                else None
            )

            for block_id in range(1, l_data.nblock):
                (
                    print(
                        f"j3={j3} | block_id={block_id} | copying l_data_multi_res[{block_id - 1}] to l_data_multi_res[{block_id}]"
                    )
                    if verbose > 1
                    else None
                )
                if l_data_multi_res[block_id] is None:
                    l_data_multi_res[block_id] = l_data_multi_res[block_id - 1].copy(
                        empty=False
                    )
            l_data_to_use = l_data_multi_res[-1]  # last block_id

            # Compute first convolution and modulus
            for block_id, j3_block in enumerate(j3):
                if j3[: block_id + 1] not in data_l1:
                    (
                        print(
                            f"j3={j3} | block_id={block_id} | computing first convolution: data_l1[{j3[: block_id + 1]}]"
                        )
                        if verbose > 1
                        else None
                    )
                    if j3_block == J[block_id]:
                        # low-pass filter: the data are already at its resolution along this block
                        data_l1[j3[: block_id + 1]] = self.wavelet_op.lowpass(
                            l_data_to_use if block_id == 0 else data_l1[j3[:block_id]],
                            block_id=block_id,
                        )  # (Nb,Nc,1,Nx_3,...,Nblock_id_3,...,Nz)
                    else:
                        data_l1[j3[: block_id + 1]] = self.wavelet_op.apply(
                            l_data_to_use if block_id == 0 else data_l1[j3[:block_id]],
                            j=j3_block,
                            block_id=block_id,
                        )  # (Nb,Nc,L,Nx_3,...,Nblock_id_3,...,Nz)
                (
                    print(
                        f"j3={j3} | block_id={block_id} | data_l1[{j3[: block_id + 1]}].conv_history: {data_l1[j3[: block_id + 1]].conv_history} | dg: {data_l1[j3[: block_id + 1]].dg}"
                    )
                    if verbose > 1
                    else None
                )

            if is_lowpass(j3):
                # only used for S1 and S2, so not stored in the second-layer path
                data_l1m_j3 = data_l1[j3].modulus(inplace=False)
            else:
                data_l1m_multi_res[j3] = [
                    data_l1[j3].modulus(inplace=False)
                ]  # (Nb,Nc,L,Nx_3,...,Nz_3)
                data_l1m_multi_res[j3] += [None] * sum(
                    [
                        1 if j3[block_id] < J[block_id] - 1 else 0
                        for block_id in range(1, l_data.nblock)
                    ]
                )
                data_l1m[j3] = data_l1m_multi_res[j3][0]  # (Nb,Nc,L,Nx_3,...,Nz_3)
                data_l1m_j3 = data_l1m[j3]
            for key, val in data_l1m.items():
                (
                    print(
                        f"j3={j3} | data_l1m[{key}].conv_history: {val.conv_history} | dg: {val.dg}"
                    )
                    if verbose > 1
                    else None
                )

            # [REVIEW] dead debug code: torch is not imported in this module and self.layer1_mask does not exist
            if False and self.wavelet_op.mask_full_res is not None:

                assert torch.all(
                    data_l1m[j3].array.isnan() == self.layer1_mask[j3].array
                )  ###################### GOOD TO KEEP WHILE DEBUGGING with replace_nan_value=torch.nan

            # orientations of the multi-scale j3 in S1 and S2: a low-pass filter has a single one, stored at index 0
            orientations_j3 = tuple(
                slice(0, 1) if j3_block == J_block else slice(None)
                for j3_block, J_block in zip(j3, J)
            )

            ##############################################################################
            ########################## S1(j3) = Mean(|I*psi3|) ###########################
            ##############################################################################
            print(f"j3={j3} | computing S1") if verbose > 0 else None

            #            data_st.S1[:, channels_with_auto_stats, j3, :] = self.wavelet_op.mean(
            #                data_l1m[j3][:, channels_with_auto_stats, :, :],
            #            )  # (Nb,Nc,L3)

            # auto S1 terms
            data_st.S1[
                (
                    slice(None),
                    channels_with_auto_stats,
                    channels_with_auto_stats,
                    *j3,
                    *orientations_j3,
                )
            ] = self.wavelet_op.mean(
                data_l1m_j3[:, channels_with_auto_stats, :, ...],
            ).to(
                dtype=data_st.S1.dtype  # cast to complex if needed
            )  # (Nb,Nc,Nc,L3)

            # cross S1 terms (sub diagonal only)
            if (compute_cross_matrix * (~bk.eye(Nc, dtype=bool, device=device))).any():
                data_l1_modulus_square_rooted = data_l1[j3].divide(
                    data_l1m_j3, epsilon=1e-8, pow=0.5, inplace=False
                )  # [Nb,Nc,L,N3]

                self.wavelet_op._compute_and_store_cross_cov(
                    data_l1_modulus_square_rooted,
                    data_l1_modulus_square_rooted,
                    output=data_st.S1[
                        (slice(None), slice(None), slice(None), *j3, *orientations_j3)
                    ],
                    compute_cross_matrix=compute_cross_matrix
                    * (
                        ~bk.eye(Nc, dtype=bool, device=device)
                    ),  # remove diagonal wich was computed above with real mean square
                    redundant_channels=True,  # S1(c1,c2) and S1(c2,c1) are conjugates
                )  # (Nb,Nc,Nc,L3)

            ##############################################################################
            ######################### S2(j3) = Mean(|I*psi3|^2) ##########################
            ##############################################################################
            print(f"j3={j3} | computing S2") if verbose > 0 else None
            # auto S2 terms
            data_st.S2[
                (
                    slice(None),
                    channels_with_auto_stats,
                    channels_with_auto_stats,
                    *j3,
                    *orientations_j3,
                )
            ] = self.wavelet_op.square_mean(
                data_l1m_j3[:, channels_with_auto_stats, :, ...]
            ).to(
                dtype=data_st.S2.dtype  # cast to complex if needed
            )  # (Nb,Nc,Nc,L3)

            # cross S2 terms (sub diagonal only)
            if (compute_cross_matrix * (~bk.eye(Nc, dtype=bool, device=device))).any():
                self.wavelet_op._compute_and_store_cross_cov(
                    data_l1[j3],
                    data_l1[j3],
                    output=data_st.S2[
                        (slice(None), slice(None), slice(None), *j3, *orientations_j3)
                    ],
                    compute_cross_matrix=compute_cross_matrix
                    * (
                        ~bk.eye(Nc, dtype=bool, device=device)
                    ),  # remove diagonal wich was computed above with real mean square
                    redundant_channels=True,  # S2(c1,c2) and S2(c2,c1) are conjugates
                )  # (Nb,Nc,Nc,L3)

            # j3 = (j3x, ..., j3z)
            j3_range = (
                []  # low-pass multi-scale: no S3 and S4, and no update of the second-layer path
                if is_lowpass(j3)
                else list(product(*(range(j3_block + 1) for j3_block in j3)))
            )  # [(0, ..., 0), (0, ..., 1), ..., (j3x, ..., j3z)] lexicographic order

            for j2 in j3_range:

                for block_id, j3_block in enumerate(j3):

                    if (j3[: block_id + 1], j2) not in data_l1m_l2:
                        (
                            print(
                                f"j3={j3} | j2={j2} | block_id={block_id} | computing second convolution: data_l1m_l2[({j3[: block_id + 1]}, {j2})]"
                            )
                            if verbose > 2
                            else None
                        )
                        data_l1m_l2[(j3[: block_id + 1], j2)] = self.wavelet_op.apply(
                            (
                                data_l1m[j2]
                                if block_id == 0
                                else data_l1m_l2[(j3[:block_id], j2)]
                            ),
                            j=j3_block,
                            block_id=block_id,
                        )  # (Nb,Nc,L2,L3,Nx_3,...,Nblock_id_3,...,Nz_2)
                        for key, val in data_l1m_l2.items():
                            if key[1] == j2:
                                (
                                    print(
                                        f"j3={j3} | j2={j2} | block_id={block_id} | data_l1m_l2[{key}].conv_history: {val.conv_history} | dg: {val.dg}"
                                    )
                                    if verbose > 2
                                    else None
                                )

                # data_l1m_l2[(j3, j2)] of shape (Nb,Nc,L2,L3,Nx_3,...,Nz_3)

                # [REVIEW] dead debug code: torch, data_l1m_l2_j2, self.layer1_mask and self.layer2_mask are not defined here
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
                (
                    print(
                        f"j3={j3} | j2={j2} | computing S3 and S4 (over none printed j1)"
                    )
                    if verbose > 1
                    else None
                )
                if not has_fewer_convolutions:
                    self.wavelet_op._compute_and_store_cross_cov(
                        data_l1m_l2[(j3, j2)],
                        data_l1[j3][
                            (slice(None), slice(None)) + (None,) * len(j2)
                        ],  # (Nb,Nc,L2,L3,N3) x (Nb,Nc,1x,...,1z,L3,N3)
                        output=data_st.S3[
                            (slice(None), slice(None), slice(None), *j2, *j3)
                        ],
                        compute_cross_matrix=compute_cross_matrix,
                        redundant_channels=False,
                    )  # (Nb,Nc,Nc,L2,L3)

                else:
                    # [REVIEW] raised only after the first S1 and S2 have been computed: could be checked before the loop
                    raise NotImplementedError
                    # Sihao S3 version : S3(j1,j2,j3) = Cov(I, |I*psi2|*psi3)
                    self.wavelet_op._compute_and_store_cross_cov(
                        l_data_to_use[
                            (slice(None), slice(None))
                            + (None,) * len(j2)
                            + (None,) * len(j3)
                        ],  # [Nb,Nc,1x,...,1z,1x,...,1z,N3]
                        data_l1m_l2[(j3, j2)],  # [Nb,Nc,L2,L3,N3]
                        output=data_st.S3[
                            (slice(None), slice(None), slice(None), *j2, *j3)
                        ],
                        compute_cross_matrix=compute_cross_matrix,
                        redundant_channels=False,
                    )  # [Nb, Nc, Nc, L2, L3]

                # j2 = (j2x, ..., j2z)
                j2_range = list(
                    product(*(range(j2_block + 1) for j2_block in j2))
                )  # [(0, ..., 0), (0, ..., 1), ..., (j2x, ..., j2z)] lexicographic order

                for j1 in j2_range:
                    ##############################################################################
                    ############## S4(j1,j2,j3) = Cov(|I*psi1|*psi3, |I*psi2|*psi3) ##############
                    ##############################################################################
                    if not has_fewer_convolutions:
                        self.wavelet_op._compute_and_store_cross_cov(
                            data_l1m_l2[(j3, j1)][
                                (slice(None), slice(None), slice(None))
                                + (None,) * len(j2)
                            ],  # (Nb,Nc,L1,1x,...,1z,L3,N3)
                            data_l1m_l2[(j3, j2)][
                                (slice(None), slice(None)) + (None,) * len(j1)
                            ],  # (Nb,Nc,1x,...,1z,L2,L3,N3)
                            output=data_st.S4[
                                (slice(None), slice(None), slice(None), *j1, *j2, *j3)
                            ],  # (Nb,Nc,Nc,L1,L2,L3)
                            compute_cross_matrix=compute_cross_matrix,
                            redundant_channels=False,  # TODO: S4(c1,c2) and S4(c2,c1) are conjugates if j1==j2 and thetha1=theta2
                        )  # (Nb,Nc,Nc,L1,L2,L3)

                    else:
                        raise NotImplementedError
                        # Sihao S4 version : S4(j1,j2,j3) = Cov(|I*psi1|, |I*psi2|*psi3)
                        self.wavelet_op._compute_and_store_cross_cov(
                            data_l1m[j1][
                                (slice(None), slice(None), slice(None))
                                + (None,) * len(j2)
                                + (None,) * len(j3)
                            ],  # [Nb,Nc,L1,1x,...,1z,1x,...,1z,N3]
                            data_l1m_l2[(j3, j2)][
                                (slice(None), slice(None)) + (None,) * len(j1)
                            ],  # [Nb,Nc,1x,...,1z,L2,L3,N3]
                            output=data_st.S4[
                                (slice(None), slice(None), slice(None), *j1, *j2, *j3)
                            ],  # [Nb,Nc,Nc,L1,L2,L3]
                            compute_cross_matrix=compute_cross_matrix,
                            redundant_channels=False,
                        )  # (Nb,Nc,Nc,L1,L2,L3)

            # Downsample inplace at Nj3_block_id for all block_id in range(l_data.nblock)
            # First-layer path (l_data_multi_res, data_l1): from j3 to next_j3, low-pass multi-scales included
            for block_id in range(l_data.nblock):
                if next_j3[block_id] is not None and j3[block_id] < next_j3[block_id]:

                    (
                        print(
                            f"j3={j3} | block_id: {block_id} | downsampling l_data_multi_res[{block_id}]"
                        )
                        if verbose > 1
                        else None
                    )
                    self.wavelet_op.downsample(
                        data=l_data_multi_res[block_id],
                        dg_out=dg_of(block_id, j3[block_id] + 1),
                        block_id=block_id,
                        inplace=True,
                        replace_nan_value=self.replace_nan_value,
                    )  # (Nb,Nc,j3x+1,j3y+1,...,j3[block_id]+1,j3[block_id+1],...,j3z,L,N3)

                    # [REVIEW] fix: the copies of the following blocks were made at the previous resolution of this block.
                    # They were only deleted when their own scale goes back (elif branch below), which never happens for a
                    # block with a single scale (J=1, e.g. 8 time steps): the next convolution then got data at the wrong
                    # resolution ("j is not equal to dg").
                    for child_block_id in range(block_id + 1, l_data.nblock):
                        l_data_multi_res[child_block_id] = None

                    if block_id > 0:
                        (
                            print(
                                f"j3={j3} | block_id: {block_id} | downsampling data_l1[{j3[:block_id]}]",
                            )
                            if verbose > 1
                            else None
                        )
                        self.wavelet_op.downsample(
                            data=data_l1[j3[:block_id]],
                            dg_out=dg_of(block_id, j3[block_id] + 1),
                            block_id=block_id,
                            inplace=True,
                            replace_nan_value=self.replace_nan_value,
                        )  # (Nb,Nc,j3x+1,...,j3[block_id]+1,j3[block_id+1],...,j3z,L,N3)

                    # remove children that are not needed anymore
                    for child_block_id in range(block_id, l_data.nblock):
                        (
                            print(
                                f"j3={j3} | block_id: {block_id} | deleting data_l1[{j3[:child_block_id + 1]}]"
                            )
                            if verbose > 1
                            else None
                        )
                        del data_l1[j3[: child_block_id + 1]]
                        assert j3[: child_block_id + 1] not in data_l1
                elif (next_j3[block_id] is None) or (j3[block_id] > next_j3[block_id]):
                    to_del = l_data_multi_res[block_id]
                    l_data_multi_res[block_id] = None
                    del to_del  # free memory
                    assert l_data_multi_res[block_id] is None
                else:
                    pass  # case j3[block_id] == next_j3[block_id]

            # Second-layer path (data_l1m_multi_res, data_l1m_l2, data_l1m): from the wavelet multi-scale j3 to the
            # next wavelet multi-scale, skipping the low-pass multi-scales in between (j3_range is empty if j3 is low-pass)
            next_j3_wavelet = next_wavelet_j3[j3]
            for block_id in range(l_data.nblock):
                if (
                    next_j3_wavelet[block_id] is not None
                    and j3[block_id] < next_j3_wavelet[block_id]
                ):
                    for j2 in j3_range:

                        for ind, val in enumerate(data_l1m_multi_res[j2]):
                            if val is None:
                                (
                                    print(
                                        f"j3={j3} | block_id: {block_id} | j2={j2} | before data_l1m_multi_res[{j2}][{ind}] is None"
                                    )
                                    if verbose > 2
                                    else None
                                )
                            else:
                                (
                                    print(
                                        f"j3={j3} | block_id: {block_id} | j2={j2} | before data_l1m_multi_res[{j2}][{ind}].conv_history: {val.conv_history} | dg: {val.dg}"
                                    )
                                    if verbose > 2
                                    else None
                                )

                        if len(data_l1m_multi_res[j2]) > 1 or block_id == 0:

                            if block_id == 0:
                                block_id_loc_in_list = 0
                            else:
                                block_id_loc_in_list = 1 + sum(
                                    [
                                        (
                                            1
                                            if j3[test_block_id] < J[test_block_id] - 1
                                            else 0
                                        )
                                        for test_block_id in range(1, block_id)
                                    ]
                                )

                            # populating the None blocks from 0 (always populated) to block_id_loc_in_list included
                            for intermediate_block_id_loc_in_list in range(
                                1, block_id_loc_in_list + 1
                            ):
                                if (
                                    data_l1m_multi_res[j2][
                                        intermediate_block_id_loc_in_list
                                    ]
                                    is None
                                ):
                                    (
                                        print(
                                            f"j3={j3} | block_id: {block_id} | j2={j2} | copying data_l1m_multi_res[{j2}] block_id {intermediate_block_id_loc_in_list - 1} to {intermediate_block_id_loc_in_list}"
                                        )
                                        if verbose > 2
                                        else None
                                    )
                                    data_l1m_multi_res[j2][
                                        intermediate_block_id_loc_in_list
                                    ] = data_l1m_multi_res[j2][
                                        intermediate_block_id_loc_in_list - 1
                                    ].copy(
                                        empty=False
                                    )

                            (
                                print(
                                    f"j3={j3} | block_id: {block_id} | j2={j2} | downsampling data_l1m_multi_res[{j2}][{block_id_loc_in_list}]"
                                )
                                if verbose > 2
                                else None
                            )
                            self.wavelet_op.downsample(
                                data=data_l1m_multi_res[j2][block_id_loc_in_list],
                                dg_out=self.wavelet_op.wavelet_ops[block_id].j_to_dg[
                                    j3[block_id] + 1
                                ],
                                block_id=block_id,
                                inplace=True,
                                replace_nan_value=self.replace_nan_value,
                            )  # (Nb,Nc,j3x+1,...,j3[block_id]+1,j3[block_id+1],...,j3z,L,N3)

                            # kill descendants if there are any
                            for ind in range(
                                block_id_loc_in_list + 1, len(data_l1m_multi_res[j2])
                            ):
                                (
                                    print(
                                        f"j3={j3} | block_id: {block_id} | j2={j2} | deleting data_l1m_multi_res[{j2}][{ind}]"
                                    )
                                    if verbose > 2
                                    else None
                                )
                                obj = data_l1m_multi_res[j2][ind]
                                data_l1m_multi_res[j2][ind] = None
                                del obj
                                assert data_l1m_multi_res[j2][ind] is None

                            if block_id > 0:
                                (
                                    print(
                                        f"j3={j3} | block_id: {block_id} | j2={j2} | downsampling data_l1m_l2[({j3[:block_id]}, {j2})] of shape: {data_l1m_l2[(j3[:block_id], j2)].array.shape}"
                                    )
                                    if verbose > 2
                                    else None
                                )
                                self.wavelet_op.downsample(
                                    data=data_l1m_l2[(j3[:block_id], j2)],
                                    dg_out=self.wavelet_op.wavelet_ops[
                                        block_id
                                    ].j_to_dg[j3[block_id] + 1],
                                    block_id=block_id,
                                    inplace=True,
                                    replace_nan_value=self.replace_nan_value,
                                )  # (Nb,Nc,j3x+1,...,j3[block_id]+1,j3[block_id+1],...,j3z,L,N3) ############################################

                            # remove children that are not needed anymore
                            if block_id == 0:
                                # in this case the delete loop can be replaced by a single delete of the entire dict
                                # [REVIEW] done again for every j2 of j3_range (on an already empty dict after the first one): could be done once after the j2 loop
                                (
                                    print(
                                        f"j3={j3} | block_id: {block_id} | deleting entirely data_l1m_l2"
                                    )
                                    if verbose > 1
                                    else None
                                )
                                del data_l1m_l2  # j3x=j3[0] part of the loop done, free memory for next one: j3x=j3[0]+1
                                data_l1m_l2 = {}
                            else:
                                for child_block_id in range(block_id, l_data.nblock):
                                    (
                                        print(
                                            f"j3={j3} | block_id: {block_id} | j2={j2} | deleting data_l1m_l2[({j3[:child_block_id+1]}, {j2})]"
                                        )
                                        if verbose > 2
                                        else None
                                    )
                                    del data_l1m_l2[(j3[: child_block_id + 1], j2)]
                                    assert (
                                        j3[: child_block_id + 1],
                                        j2,
                                    ) not in data_l1m_l2

            # [REVIEW] the verbose prints below show block_id, a leftover of the loop above (always the last block)
            for j2 in j3_range:
                for ind, val in enumerate(data_l1m_multi_res[j2]):
                    if val is None:
                        (
                            print(
                                f"j3={j3} | block_id: {block_id} | j2={j2} | after data_l1m_multi_res[{j2}][{ind}] is None"
                            )
                            if verbose > 2
                            else None
                        )
                    else:
                        (
                            print(
                                f"j3={j3} | block_id: {block_id} | j2={j2} | after data_l1m_multi_res[{j2}][{ind}].conv_history: {val.conv_history} | dg: {val.dg}"
                            )
                            if verbose > 2
                            else None
                        )
                (
                    print(
                        f"j3={j3} | block_id: {block_id} | j2={j2} | SETTING data_l1m[{j2}] to last non-None value in data_l1m_multi_res[{j2}] at index {len([val for val in data_l1m_multi_res[j2] if val is not None]) - 1}"
                    )
                    if verbose > 2
                    else None
                )
                data_l1m[j2] = [
                    val for val in data_l1m_multi_res[j2] if val is not None
                ][-1]
                (
                    print(
                        f"j3={j3} | block_id: {block_id} | j2={j2} | data_l1m[{j2}].conv_history: {data_l1m[j2].conv_history} | dg: {data_l1m[j2].dg}"
                    )
                    if verbose > 2
                    else None
                )

        return data_st


# With low-pass filters, the low-pass multi-scales j3 only use l_data and data_l1 (S1 and S2), the low-pass filter
# of a block being applied by downsampling l_data_multi_res and data_l1[j3[:b]] once more along this block. The
# objects of the second layer (data_l1m, data_l1m_l2) are only used, and updated, at the wavelet multi-scales.

# l_data      used for data_l1[[j3x]] @j3 for apply at j3x
#          + used for S3 Sihao       @j3,j2

# data_l1     used as data_l1[j3[:b]] b>0 for data_l1[j3[:b+1]] @j3 for apply at j3[b]
#          + used as data_l1[j3]                               @j3
#          + used as data_l1[j3] for S3 not Sihao              @j3,j2

# data_l1m    used as data_l1m[j3]                                     @j3
#          + used as data_l1m[j2] for data_l1m_l2[([j3x], j2)]        @j3,j2 for apply at j3x
#          + used as data_l1m[j1] for S4 Sihao                        @j3,j2,j1

# data_l1m_l2 used as data_l1m_l2[(j3[:b], j2)] b>0 for data_l1m_l2[(j3[:b + 1], j2)] @j3,j2 for apply at j3[b]
#          + used as data_l1m_l2[(j3, j2)] for S3                                    @j3,j2
#          + used as data_l1m_l2[(j3, j2)] for S4 not Sihao (and Sihao)              @j3,j2,j1
#          + used as data_l1m_l2[(j3, j1)] for S4 not Sihao                          @j3,j2,j1


############################### Vanilla ST ###############################

# l_data      used for data_l1[[j3x]] @j3 for apply at j3x

# inside j3 loop: ok to downsample at j3x
# inside j3[1:] loop: ok to downsample at j3y -> MUST reset y resolution quiting the j3[1:] loop
#                 ...
# inside j3[nblock:] loop: ok to downsample at j3z  -> MUST reset y resolution quiting the j3[nblock:] loop


# data_l1     used as data_l1[j3[:b]] b>0 for data_l1[j3[:b+1]] @j3 for apply at j3[b]
#          + used as data_l1[j3]                               @j3
#          + used as data_l1[j3] for S3 not Sihao              @j3,j2

# inside j3 loop: ok to downsample at j3x
# inside j3[1:] loop: ok to downsample at j3y -> MUST reset y resolution quiting the j3[1:] loop
#                 ...
# inside j3[nblock:] loop: ok to downsample at j3z  -> MUST reset y resolution quiting the j3[nblock:] loop


# data_l1m    used as data_l1m[j3]                                     @j3
#          + used as data_l1m[j2] for data_l1m_l2[([j3x], j2)]        @j3,j2 for apply at j3x

# inside j3 loop: ok to downsample at j3x
# inside j3[1:] loop: ok to downsample at j3y -> MUST reset y resolution quiting the j3[1:] loop
#                 ...
# inside j3[nblock:] loop: ok to downsample at j3z  -> MUST reset y resolution quiting the j3[nblock:] loop


# data_l1m_l2 used as data_l1m_l2[(j3[:b], j2)] b>0 for data_l1m_l2[(j3[:b + 1], j2)] @j3,j2 for apply at j3[b]
#          + used as data_l1m_l2[(j3, j2)] for S3                                    @j3,j2
#          + used as data_l1m_l2[(j3, j2)] for S4 not Sihao (and Sihao)              @j3,j2,j1
#          + used as data_l1m_l2[(j3, j1)] for S4 not Sihao                          @j3,j2,j1

# NO MORE NEED: inside j3 loop: ok to downsample at j3x
# inside j3[1:] loop: ok to downsample at j3y -> MUST reset y resolution quiting the j3[1:] loop
#                 ...
# inside j3[nblock:] loop: ok to downsample at j3z  -> MUST reset y resolution quiting the j3[nblock:] loop
# IMPORTANT: data_l1m_l2 can be deleted when quiting the j3[1:] loop (i.e. when j3x is incremented to j3x + 1)
# ...
# IMPORTANT: for all b>=0 data_l1m_l2[(j3 s.t. len(j3) > b, whatever j2)] can be deleted when quiting the j3[b+1:] loop (i.e. when j3[b] (e.g. j3y for b=1) is incremented to j3[b] + 1)


############################### Sihao ST ###############################

# l_data      used for data_l1[[j3x]] @j3 for apply at j3x
#          + used for S3 Sihao       @j3,j2

# data_l1     used as data_l1[j3[:b]] b>0 for data_l1[j3[:b+1]] @j3 for apply at j3[b]
#          + used as data_l1[j3]                               @j3

# data_l1m    used as data_l1m[j3]                                     @j3
#          + used as data_l1m[j2] for data_l1m_l2[([j3x], j2)]        @j3,j2 for apply at j3x
#          + used as data_l1m[j1] for S4 Sihao                        @j3,j2,j1

# data_l1m_l2 used as data_l1m_l2[(j3[:b], j2)] b>0 for data_l1m_l2[(j3[:b + 1], j2)] @j3,j2 for apply at j3[b]
#          + used as data_l1m_l2[(j3, j2)] for S3                                    @j3,j2
#          + used as data_l1m_l2[(j3, j2)] for S4 not Sihao (and Sihao)              @j3,j2,j1
