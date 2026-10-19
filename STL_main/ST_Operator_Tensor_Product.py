from itertools import product

import STL_main.torch_backend as bk  # from_numpy, zeros, ones, dim, shape, nan, eye
from STL_main.ST_Operator import ST_Operator


class ST_Operator_TensorProd(ST_Operator):

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
    ):

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
        J_range = list(
            product(*(range(J_block) for J_block in J))
        )  # [(0, ..., 0), (0, ..., 1), ..., (Jx-1, ..., Jz-1)] lexicographic order

        for j3, next_j3 in zip(J_range, J_range[1:] + [(None,) * len(J)]):
            print(
                f"\n****************** NEW ITERATION ******************\nj3={j3}, next_j3={next_j3}"
            )

            for block_id in range(1, l_data.nblock):
                print(
                    f"j3={j3} | block_id={block_id} | copying l_data_multi_res[{block_id - 1}] to l_data_multi_res[{block_id}]"
                )
                if l_data_multi_res[block_id] is None:
                    l_data_multi_res[block_id] = l_data_multi_res[block_id - 1].copy(
                        empty=False
                    )
            l_data_to_use = l_data_multi_res[-1]  # last block_id

            # Compute first convolution and modulus
            for block_id, j3_block in enumerate(j3):
                if j3[: block_id + 1] not in data_l1:
                    print(
                        f"j3={j3} | block_id={block_id} | computing first convolution: data_l1[{j3[: block_id + 1]}]"
                    )
                    data_l1[j3[: block_id + 1]] = self.wavelet_op.apply(
                        l_data_to_use if block_id == 0 else data_l1[j3[:block_id]],
                        j=j3_block,
                        block_id=block_id,
                    )  # (Nb,Nc,L,Nx_3,...,Nblock_id_3,...,Nz)
                print(
                    f"j3={j3} | block_id={block_id} | data_l1[{j3[: block_id + 1]}].conv_history: {data_l1[j3[: block_id + 1]].conv_history} | dg: {data_l1[j3[: block_id + 1]].dg}"
                )

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
            for key, val in data_l1m.items():
                print(
                    f"j3={j3} | data_l1m[{key}].conv_history: {val.conv_history} | dg: {val.dg}"
                )

            if False and self.wavelet_op.mask_full_res is not None:

                assert torch.all(
                    data_l1m[j3].array.isnan() == self.layer1_mask[j3].array
                )  ###################### GOOD TO KEEP WHILE DEBUGGING with replace_nan_value=torch.nan

            ##############################################################################
            ########################## S1(j3) = Mean(|I*psi3|) ###########################
            ##############################################################################
            print(f"j3={j3} | computing S1")

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
                    slice(None),
                )
            ] = self.wavelet_op.mean(
                data_l1m[j3][:, channels_with_auto_stats, :, ...],
            ).to(
                dtype=data_st.S1.dtype  # cast to complex if needed
            )  # (Nb,Nc,Nc,L3)

            # cross S1 terms (sub diagonal only)
            if (compute_cross_matrix * (~bk.eye(Nc, dtype=bool, device=device))).any():
                data_l1_modulus_square_rooted = data_l1[j3].divide(
                    data_l1m[j3], epsilon=1e-8, pow=0.5, inplace=False
                )  # [Nb,Nc,L,N3]

                self.wavelet_op._compute_and_store_cross_cov(
                    data_l1_modulus_square_rooted,
                    data_l1_modulus_square_rooted,
                    output=data_st.S1[
                        (slice(None), slice(None), slice(None), *j3, slice(None))
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
            print(f"j3={j3} | computing S2")
            # auto S2 terms
            data_st.S2[
                (
                    slice(None),
                    channels_with_auto_stats,
                    channels_with_auto_stats,
                    *j3,
                    slice(None),
                )
            ] = self.wavelet_op.square_mean(
                data_l1m[j3][:, channels_with_auto_stats, :, ...]
            ).to(
                dtype=data_st.S2.dtype  # cast to complex if needed
            )  # (Nb,Nc,Nc,L3)

            # cross S2 terms (sub diagonal only)
            if (compute_cross_matrix * (~bk.eye(Nc, dtype=bool, device=device))).any():
                self.wavelet_op._compute_and_store_cross_cov(
                    data_l1[j3],
                    data_l1[j3],
                    output=data_st.S2[
                        (slice(None), slice(None), slice(None), *j3, slice(None))
                    ],
                    compute_cross_matrix=compute_cross_matrix
                    * (
                        ~bk.eye(Nc, dtype=bool, device=device)
                    ),  # remove diagonal wich was computed above with real mean square
                    redundant_channels=True,  # S2(c1,c2) and S2(c2,c1) are conjugates
                )  # (Nb,Nc,Nc,L3)

            # j3 = (j3x, ..., j3z)
            j3_range = list(
                product(*(range(j3_block + 1) for j3_block in j3))
            )  # [(0, ..., 0), (0, ..., 1), ..., (j3x, ..., j3z)] lexicographic order

            for j2 in j3_range:

                for block_id, j3_block in enumerate(j3):

                    if (j3[: block_id + 1], j2) not in data_l1m_l2:
                        print(
                            f"j3={j3} | j2={j2} | block_id={block_id} | computing second convolution: data_l1m_l2[({j3[: block_id + 1]}, {j2})]"
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
                                print(
                                    f"j3={j3} | j2={j2} | block_id={block_id} | data_l1m_l2[{key}].conv_history: {val.conv_history} | dg: {val.dg}"
                                )

                # data_l1m_l2[(j3, j2)] of shape (Nb,Nc,L2,L3,Nx_3,...,Nz_3)

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
                print(f"j3={j3} | j2={j2} | computing S3 and S4 (over none printed j1)")
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

            # Downsample inplace at Nj3x
            block_id = 0  # downsample first dimension

            if next_j3[block_id] is not None and j3[block_id] < next_j3[block_id]:

                print(
                    f"j3={j3} | block_id: {block_id} | downsampling l_data_multi_res[{block_id}]"
                )
                self.wavelet_op.downsample(
                    data=l_data_multi_res[block_id],
                    dg_out=self.wavelet_op.wavelet_ops[block_id].j_to_dg[
                        j3[block_id] + 1
                    ],
                    block_id=block_id,
                    inplace=True,
                    replace_nan_value=self.replace_nan_value,
                )  # (Nb,Nc,j3x+1,j3y,...,j3z,L,N3)

                # remove children that are not needed anymore
                for child_block_id in range(block_id, l_data.nblock):
                    print(
                        f"j3={j3} | block_id: {block_id} | deleting data_l1[{j3[:child_block_id + 1]}]"
                    )
                    del data_l1[j3[: child_block_id + 1]]
                    assert j3[: child_block_id + 1] not in data_l1

                for j2 in j3_range:

                    # for key_bloc_id in range(l_data.nblock):
                    #    self.wavelet_op.downsample(
                    #        data=data_l1[j2[:key_bloc_id + 1]],
                    #        dg_out=self.wavelet_op.wavelet_ops[block_id].j_to_dg[j3[block_id] + 1],
                    #        block_id=block_id,
                    #        inplace=True,
                    #        replace_nan_value=self.replace_nan_value,
                    #    )  # (Nb,Nc,j3x+1,j3y,...,j3z,L,N3)

                    for ind, val in enumerate(data_l1m_multi_res[j2]):
                        if val is None:
                            print(
                                f"j3={j3} | block_id: {block_id} | j2={j2} | before data_l1m_multi_res[{j2}][{ind}] is None"
                            )
                        else:
                            print(
                                f"j3={j3} | block_id: {block_id} | j2={j2} | before data_l1m_multi_res[{j2}][{ind}].conv_history: {val.conv_history} | dg: {val.dg}"
                            )

                    print(
                        f"j3={j3} | block_id: {block_id} | j2={j2} | downsampling data_l1m_multi_res[{j2}][{0}]"
                    )
                    self.wavelet_op.downsample(
                        data=data_l1m_multi_res[j2][0],
                        dg_out=self.wavelet_op.wavelet_ops[block_id].j_to_dg[
                            j3[block_id] + 1
                        ],
                        block_id=block_id,
                        inplace=True,
                        replace_nan_value=self.replace_nan_value,
                    )  # (Nb,Nc,j3x+1,j3y,...,j3z,L,N3)

                    for ind in range(1, len(data_l1m_multi_res[j2])):
                        print(
                            f"j3={j3} | block_id: {block_id} | j2={j2} | deleting data_l1m_multi_res[{j2}][{ind}]"
                        )
                        obj = data_l1m_multi_res[j2][ind]
                        data_l1m_multi_res[j2][ind] = None
                        del obj
                        assert data_l1m_multi_res[j2][ind] is None

                    for ind, val in enumerate(data_l1m_multi_res[j2]):
                        if val is None:
                            print(
                                f"j3={j3} | block_id: {block_id} | j2={j2} | before data_l1m_multi_res[{j2}][{ind}] is None"
                            )
                        else:
                            print(
                                f"j3={j3} | block_id: {block_id} | j2={j2} | before data_l1m_multi_res[{j2}][{ind}].conv_history: {val.conv_history} | dg: {val.dg}"
                            )

                print(f"j3={j3} | block_id: {block_id} | deleting entirely data_l1m_l2")
                del data_l1m_l2  # j3x=j3[0] part of the loop done,free memory for next one: j3x=j3[0]+1
                data_l1m_l2 = {}

            # Downsample inplace at Nj3_block_id for block_id in range(1, l_data.nblock)
            for block_id in range(1, l_data.nblock):
                if next_j3[block_id] is not None and j3[block_id] < next_j3[block_id]:

                    print(
                        f"j3={j3} | block_id: {block_id} | downsampling l_data_multi_res[{block_id}]"
                    )
                    self.wavelet_op.downsample(
                        data=l_data_multi_res[block_id],
                        dg_out=self.wavelet_op.wavelet_ops[block_id].j_to_dg[
                            j3[block_id] + 1
                        ],
                        block_id=block_id,
                        inplace=True,
                        replace_nan_value=self.replace_nan_value,
                    )  # (Nb,Nc,j3x+1,j3y+1,...,j3[block_id]+1,j3[block_id+1],...,j3z,L,N3)

                    print(
                        f"j3={j3} | block_id: {block_id} | downsampling data_l1[{j3[:block_id]}]",
                    )
                    self.wavelet_op.downsample(
                        data=data_l1[j3[:block_id]],
                        dg_out=self.wavelet_op.wavelet_ops[block_id].j_to_dg[
                            j3[block_id] + 1
                        ],
                        block_id=block_id,
                        inplace=True,
                        replace_nan_value=self.replace_nan_value,
                    )  # (Nb,Nc,j3x+1,...,j3[block_id]+1,j3[block_id+1],...,j3z,L,N3)

                    # remove children that are not needed anymore
                    for child_block_id in range(block_id, l_data.nblock):
                        print(
                            f"j3={j3} | block_id: {block_id} | deleting data_l1[{j3[:child_block_id + 1]}]"
                        )
                        del data_l1[j3[: child_block_id + 1]]
                        assert j3[: child_block_id + 1] not in data_l1

                    for j2 in j3_range:
                        print(
                            f"j3={j3} | block_id: {block_id} | j2={j2} | data_l1m_multi_res[j2]="
                        )
                        for ind, val in enumerate(data_l1m_multi_res[j2]):
                            if val is None:
                                print(
                                    f"j3={j3} | block_id: {block_id} | j2={j2} | before data_l1m_multi_res[{j2}][{ind}] is None"
                                )
                            else:
                                print(
                                    f"j3={j3} | block_id: {block_id} | j2={j2} | before data_l1m_multi_res[{j2}][{ind}].conv_history: {val.conv_history} | dg: {val.dg}"
                                )
                        if len(data_l1m_multi_res[j2]) > 1:
                            block_id_loc_in_list = 1 + sum(
                                [
                                    1 if j3[test_block_id] < J[test_block_id] - 1 else 0
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
                                    print(
                                        f"j3={j3} | block_id: {block_id} | j2={j2} | copying data_l1m_multi_res[{j2}] block_id {intermediate_block_id_loc_in_list - 1} to {intermediate_block_id_loc_in_list}"
                                    )
                                    data_l1m_multi_res[j2][
                                        intermediate_block_id_loc_in_list
                                    ] = data_l1m_multi_res[j2][
                                        intermediate_block_id_loc_in_list - 1
                                    ].copy(
                                        empty=False
                                    )
                            print(
                                f"j3={j3} | block_id: {block_id} | j2={j2} | downsampling data_l1m_multi_res[{j2}][{block_id_loc_in_list}]"
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
                                print(
                                    f"j3={j3} | block_id: {block_id} | j2={j2} | deleting data_l1m_multi_res[{j2}][{ind}]"
                                )
                                obj = data_l1m_multi_res[j2][ind]
                                data_l1m_multi_res[j2][ind] = None
                                del obj
                                assert data_l1m_multi_res[j2][ind] is None

                            # for ind, obj in enumerate(data_l1m_multi_res[j2][block_id_loc_in_list + 1:]):
                            #    data_l1m_multi_res[j2][block_id_loc_in_list + 1:][ind] = None
                            #    del obj
                            #    assert data_l1m_multi_res[j2][block_id_loc_in_list + 1:][ind] is None

                            print(
                                f"j3={j3} | block_id: {block_id} | j2={j2} | downsampling data_l1m_l2[({j3[:block_id]}, {j2})] of shape: {data_l1m_l2[(j3[:block_id], j2)].array.shape}"
                            )
                            self.wavelet_op.downsample(
                                data=data_l1m_l2[(j3[:block_id], j2)],
                                dg_out=self.wavelet_op.wavelet_ops[block_id].j_to_dg[
                                    j3[block_id] + 1
                                ],
                                block_id=block_id,
                                inplace=True,
                                replace_nan_value=self.replace_nan_value,
                            )  # (Nb,Nc,j3x+1,...,j3[block_id]+1,j3[block_id+1],...,j3z,L,N3) ############################################

                            # remove children that are not needed anymore
                            for child_block_id in range(block_id, l_data.nblock):
                                print(
                                    f"j3={j3} | block_id: {block_id} | j2={j2} | deleting data_l1m_l2[({j3[:child_block_id+1]}, {j2})]"
                                )
                                del data_l1m_l2[(j3[: child_block_id + 1], j2)]
                                assert (j3[: child_block_id + 1], j2) not in data_l1m_l2

                elif (next_j3[block_id] is None) or (j3[block_id] > next_j3[block_id]):
                    to_del = l_data_multi_res[block_id]
                    l_data_multi_res[block_id] = None
                    del to_del  # free memory
                    assert l_data_multi_res[block_id] is None
                else:
                    pass  # case j3[block_id] == next_j3[block_id]

            for j2 in j3_range:
                for ind, val in enumerate(data_l1m_multi_res[j2]):
                    if val is None:
                        print(
                            f"j3={j3} | block_id: {block_id} | j2={j2} | after data_l1m_multi_res[{j2}][{ind}] is None"
                        )
                    else:
                        print(
                            f"j3={j3} | block_id: {block_id} | j2={j2} | after data_l1m_multi_res[{j2}][{ind}].conv_history: {val.conv_history} | dg: {val.dg}"
                        )
                print(
                    f"j3={j3} | block_id: {block_id} | j2={j2} | SETTING data_l1m[{j2}] to last non-None value in data_l1m_multi_res[{j2}] at index {len([val for val in data_l1m_multi_res[j2] if val is not None]) - 1}"
                )
                data_l1m[j2] = [
                    val for val in data_l1m_multi_res[j2] if val is not None
                ][-1]
                print(
                    f"j3={j3} | block_id: {block_id} | j2={j2} | data_l1m[{j2}].conv_history: {data_l1m[j2].conv_history} | dg: {data_l1m[j2].dg}"
                )

        return data_st


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
