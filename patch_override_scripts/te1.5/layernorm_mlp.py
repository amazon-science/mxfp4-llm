# Original Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# See LICENSE for license information.

"""LayerNormMLP API"""
import os
import warnings
from typing import Union, Optional, Callable, Tuple, List, Dict, Any

import torch
from torch.nn.parameter import Parameter
from torch.nn import init

from .base import (
    get_workspace,
    _prepare_backward,
    get_ub,
    TransformerEngineBaseModule,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from ..fp8 import get_fp8_te_dtype, FP8GlobalStateManager
from ..jit import (
    bias_gelu_fused,
    bgrad_dgelu_fused,
    set_jit_fusion_options,
    warmup_jit_bias_gelu_all_dtypes,
)
from ..utils import (
    divide,
    get_default_init_method,
    init_method_constant,
    cast_if_needed,
    assert_dim_for_fp8_exec,
    clear_tensor_data,
)
from ..distributed import (
    set_tensor_model_parallel_attributes,
    get_distributed_world_size,
    allreduce,
    reduce_scatter_along_first_dim,
    gather_along_first_dim,
    is_fp8_activation_recompute_enabled,
    in_fp8_activation_recompute_phase,
    use_reentrant_activation_recompute,
)

from .. import cpp_extensions as tex

from ..constants import dist_group_type, TE_DType
from ..jit import no_torch_dynamo

from ..float8_tensor import Float8Tensor
from ._common import _apply_normalization

__all__ = ["LayerNormMLP"]


import mx
from scipy.linalg import hadamard
import os
from contextlib import contextmanager

mxfp4 = mx.MxSpecs()
mxfp4['scale_bits'] = 8
mxfp4['w_elem_format'] = 'fp4_e2m1'
mxfp4['a_elem_format'] = 'fp4_e2m1'
mxfp4['block_size'] = 32
mxfp4['bfloat'] = 0
mxfp4['custom_cuda'] = False

FW_EMULATE_FP8 = int(os.environ.get('FW_EMULATE_FP8', 0))
print(f'FW EMULATING FP8 WITH BF16: {FW_EMULATE_FP8}')
BW_USE_MXFP4 = int(os.environ.get('BW_USE_MXFP4', 0))
print(f'BW USING MXFP4: {BW_USE_MXFP4}')

# 0 = BF16 BW
# 1 = RHT + OCP MXFP4
# 2 = OCP MXFP4
# 3 = RHT + Scaled/SR MXFP4
# 4 = Scaled/SR MXFP4

if BW_USE_MXFP4 == 3 or BW_USE_MXFP4 == 4:
    mxfp4['round_mx_output'] = 'dither_scale'

HBS = int(os.environ.get('HBS', 64))

@contextmanager
def emulate_fp8():
    fp32_matmul_precision = torch.get_float32_matmul_precision()
    allow_bf16_reduced_precision_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    yield
    torch.set_float32_matmul_precision(fp32_matmul_precision)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = allow_bf16_reduced_precision_reduction


def _act_func(activation: str):
    funcs = {
            'gelu': (tex.gelu, tex.dgelu),
            'relu': (tex.relu, tex.drelu),
            'geglu': (tex.geglu, tex.dgeglu),
            'reglu': (tex.reglu, tex.dreglu),
            'swiglu': (tex.swiglu, tex.dswiglu),
            'qgelu': (tex.qgelu, tex.dqgelu)
    }
    if activation not in funcs:
        raise NotImplementedError("Activation type " + activation + " is not supported!")
    return funcs[activation]


class _LayerNormMLP(torch.autograd.Function):
    """LayerNormMLP semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc1_weight_fp8: Union[torch.Tensor, None],
        fc1_weight_t_fp8: Union[torch.Tensor, None],
        fc1_bias: torch.Tensor,
        use_fc1_bias: bool,
        fc2_weight: torch.Tensor,
        fc2_weight_fp8: Union[torch.Tensor, None],
        fc2_weight_t_fp8: Union[torch.Tensor, None],
        fc2_bias: torch.Tensor,
        use_fc2_bias: bool,
        eps: float,
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_calibration: bool,
        fp8_meta: Dict[str, Any],
        fuse_wgrad_accumulation: bool,
        cpu_offloading: bool,
        tp_group: Union[dist_group_type, None],
        tp_size: int,
        sequence_parallel: bool,
        tensor_parallel: bool,
        activation_dtype: torch.dtype,
        return_layernorm_output: bool,
        return_layernorm_output_gathered: bool,
        bias_gelu_nvfusion: bool,
        set_parallel_mode: bool,
        is_grad_enabled: bool,
        fwd_ln_sm_margin: int,
        bwd_ln_sm_margin: int,
        zero_centered_gamma: bool,
        activation: str,
        normalization: str,
        primary_weights_in_fp8: bool,
        ub_bulk_wgrad: bool,
        ub_bulk_dgrad: bool,
        ub_split_rs: bool,
        ub_atomic_gemm_rs: bool,
        ub_split_ag: bool,
        ub_atomic_gemm_ag: bool,
        gemm_gelu_fusion: bool,
        had: torch.Tensor,
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        # Make sure input dimensions are compatible
        in_features = ln_weight.numel()
        assert inp.shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.view((-1, in_features))
        if fp8:
            assert_dim_for_fp8_exec(inputmat)
            assert_dim_for_fp8_exec(fc1_weight)
            assert_dim_for_fp8_exec(fc2_weight)

        update_fp8_weights = is_first_microbatch is None or is_first_microbatch

        activation_func = _act_func(activation)[0]

        # Cast for native AMP
        inputmat = cast_if_needed(inputmat, activation_dtype)
        ln_weight = cast_if_needed(ln_weight, activation_dtype)
        if ln_bias is not None:
            ln_bias = cast_if_needed(ln_bias, activation_dtype)

        if ub_split_ag or ub_atomic_gemm_ag:
            tp_world_size = get_distributed_world_size(tp_group)
            if tp_world_size == 1 or (not is_grad_enabled) or return_layernorm_output:
                ub_split_ag = False
                ub_atomic_gemm_ag = False
        ub_overlap_ag = ub_split_ag or ub_atomic_gemm_ag
        if ub_overlap_ag:
            ub_obj_lnout = get_ub("fc1_fprop")
            ln_out = ub_obj_lnout.get_ubuf_output(0)
        else:
            ln_out_dtype = torch.uint8 if (fp8 and not return_layernorm_output) else inputmat.dtype
            ln_out = torch.empty_like(inputmat, dtype=ln_out_dtype)
        if ub_split_rs or ub_atomic_gemm_rs:
            tp_world_size = get_distributed_world_size(tp_group)
            if tp_world_size == 1:
                ub_split_rs = False
                ub_atomic_gemm_rs = False
        if ub_atomic_gemm_rs or ub_atomic_gemm_ag:
            assert fp8, "AtomicGemm overlap supported only for FP8 GEMM."

        fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)

        ln_out, mu, rsigma = _apply_normalization(inputmat,
                                                  ln_out,
                                                  ln_weight,
                                                  ln_bias,
                                                  eps,
                                                  fp8 and not return_layernorm_output,
                                                  fp8_meta,
                                                  normalization,
                                                  fwd_ln_sm_margin,
                                                  zero_centered_gamma,
                                                  is_grad_enabled)

        # Column Parallel Linear
        ln_out_gathered = False
        if ub_overlap_ag:
            ln_out_total = ub_obj_lnout.get_ubuf_output(1)
            ln_out = torch.empty_like(ln_out)
        elif set_parallel_mode and sequence_parallel:
            ln_out_gathered = True
            ln_out_total, _ = gather_along_first_dim(ln_out, tp_group)
        else:
            ln_out_total = ln_out

        # If residual connection is after LN, we need `ln_out`
        # tensor in higher precision, this comes at the cost
        # of an extra fp8 cast.
        if return_layernorm_output:
            ln_out_return = ln_out_total if return_layernorm_output_gathered else ln_out
            if fp8:
                ln_out = tex.cast_to_fp8(
                    ln_out,
                    fp8_meta["scaling_fwd"],
                    tex.FP8FwdTensors.GEMM1_INPUT,
                    fp8_dtype_forward,
                )

        if fp8:
            bias_dtype = (
                torch.bfloat16
                if activation_dtype == torch.float32
                else activation_dtype
            )
            fc1_bias = cast_if_needed(fc1_bias, bias_dtype) if use_fc1_bias else fc1_bias
            fc2_bias = cast_if_needed(fc2_bias, bias_dtype) if use_fc2_bias else fc2_bias

            if primary_weights_in_fp8:
                # Weights are already in FP8
                fc1_weight.reset_fp8_meta_scale_inv()
                fc2_weight.reset_fp8_meta_scale_inv()
                fc1_weight_fp8 = fc1_weight
                fc2_weight_fp8 = fc2_weight
                fc1_weight_t_fp8 = None
                fc2_weight_t_fp8 = None
            elif update_fp8_weights:
                # Need to cast weights to FP8
                fc1_weight_fp8 = Float8Tensor(
                    data=fc1_weight_fp8._data,
                    fp8_meta=fp8_meta,
                    fp8_meta_index=tex.FP8FwdTensors.GEMM1_WEIGHT,
                )
                fc2_weight_fp8 = Float8Tensor(
                    data=fc2_weight_fp8._data,
                    fp8_meta=fp8_meta,
                    fp8_meta_index=tex.FP8FwdTensors.GEMM2_WEIGHT,
                )
                if (is_grad_enabled
                    or (is_fp8_activation_recompute_enabled()
                        and not in_fp8_activation_recompute_phase())):
                    # Fused cast-transpose kernels
                    tex.fp8_cast_transpose_fused(
                        fc1_weight,
                        fp8_meta["scaling_fwd"],
                        tex.FP8FwdTensors.GEMM1_WEIGHT,
                        fp8_dtype_forward,
                        cast_out=fc1_weight_fp8._data,
                        transpose_out=fc1_weight_t_fp8._data,
                    )
                    tex.fp8_cast_transpose_fused(
                        fc2_weight,
                        fp8_meta["scaling_fwd"],
                        tex.FP8FwdTensors.GEMM2_WEIGHT,
                        fp8_dtype_forward,
                        cast_out=fc2_weight_fp8._data,
                        transpose_out=fc2_weight_t_fp8._data,
                    )
                else:
                    tex.cast_to_fp8(
                        fc1_weight,
                        fp8_meta["scaling_fwd"],
                        tex.FP8FwdTensors.GEMM1_WEIGHT,
                        fp8_dtype_forward,
                        out=fc1_weight_fp8._data,
                    )
                    fc1_weight_t_fp8 = None
                    tex.cast_to_fp8(
                        fc2_weight,
                        fp8_meta["scaling_fwd"],
                        tex.FP8FwdTensors.GEMM2_WEIGHT,
                        fp8_dtype_forward,
                        out=fc2_weight_fp8._data,
                    )
                    fc2_weight_t_fp8 = None

            ub_algo = tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG if ub_split_ag else None
            ub_algo = tex.UbufOverlapAlgo.ATOMIC_GEMM_AG if ub_atomic_gemm_ag else ub_algo

            # Perform FP8 GEMM
            fp8_gemm_args = [
                fc1_weight_fp8._data,
                fp8_meta["scaling_fwd"].scale_inv,
                tex.FP8FwdTensors.GEMM1_WEIGHT,
                fp8_dtype_forward,
                ln_out_total,
                fp8_meta["scaling_fwd"].scale_inv,
                tex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
                activation_dtype,
                get_workspace(),
            ]
            fp8_gemm_kwargs = dict(
                bias=fc1_bias,
                use_bias=use_fc1_bias,
                use_split_accumulator=_2X_ACC_FPROP,
                ub_algo=ub_algo,
                ub=ub_obj_lnout if ub_overlap_ag else None,
                extra_output_tensor=ln_out if ub_overlap_ag else None,
            )
            if gemm_gelu_fusion and FW_EMULATE_FP8 == 0:
                fp8_gemm_args[8] = torch.uint8  # out_dtype
                fp8_gemm_kwargs.update(
                    dict(
                        gelu=True,
                        out_index=tex.FP8FwdTensors.GEMM2_INPUT,
                        fp8_meta_tensor=fp8_meta["scaling_fwd"],
                        D_dtype=fp8_dtype_forward,
                    )
                )
                
            if FW_EMULATE_FP8 > 0:
                with emulate_fp8():
                    ln_out_tmp = torch.ops.tex_ts.cast_from_fp8_ts(
                        ln_out_total,
                        fp8_meta["scaling_fwd"].scale_inv,
                        tex.FP8FwdTensors.GEMM1_INPUT,
                        fp8_dtype_forward,
                        TE_DType[torch.bfloat16],
                    )
                    w_tmp = torch.ops.tex_ts.cast_from_fp8_ts(
                        fc1_weight_fp8._data,
                        fp8_meta["scaling_fwd"].scale_inv,
                        tex.FP8FwdTensors.GEMM1_WEIGHT,
                        fp8_dtype_forward,
                        TE_DType[torch.bfloat16],
                    )
                    if use_fc1_bias:
                        fp8_gemm_out = (ln_out_tmp @ w_tmp.T + fc1_bias, None)
                    else:
                        fp8_gemm_out = (ln_out_tmp @ w_tmp.T, None)
            else:
                fp8_gemm_out = tex.fp8_gemm(*fp8_gemm_args, **fp8_gemm_kwargs)
            if not is_grad_enabled:
                clear_tensor_data(ln_out_total)

            # Perform activation
            if gemm_gelu_fusion and FW_EMULATE_FP8 == 0:
                gelu_out, fc1_out = fp8_gemm_out
            else:
                fc1_out, _ = fp8_gemm_out
                gelu_out = activation_func(
                    fc1_out,
                    fp8_meta["scaling_fwd"],
                    tex.FP8FwdTensors.GEMM2_INPUT,
                    fp8_dtype_forward,
                )
            if not is_grad_enabled:
                clear_tensor_data(fc1_out)

            fc2_out_index, fc2_meta_tensor, fc2_te_type, out_type = (
                None, None, None, activation_dtype)
            if ub_split_rs or ub_atomic_gemm_rs:
                ub_obj_fc2out = get_ub("fc2_fprop")
                fc2_out = ub_obj_fc2out.get_ubuf_output(1)
                dim_size = list(gelu_out.size())
                dim_size[0] = dim_size[0] // tp_world_size
                dim_size[1] = fc2_weight.size(0)
                rs_out = torch.empty(dim_size, dtype=activation_dtype, device=gelu_out.device)

                if ub_obj_fc2out.is_fp8_ubuf():
                    fc2_out_index = tex.FP8FwdTensors.GEMM2_OUTPUT
                    fc2_meta_tensor = fp8_meta["scaling_fwd"]
                    fc2_te_type = fp8_dtype_forward
                    out_type = torch.uint8
                    ub_obj_fc2out.set_ubuf_scale_inv(fc2_meta_tensor.scale_inv[fc2_out_index])
            else:
                dim_size = list(gelu_out.size())
                dim_size[1] = fc2_weight.size(0)
                fc2_out = torch.empty(dim_size, dtype=activation_dtype, device=gelu_out.device)

            ub_algo=tex.UbufOverlapAlgo.ATOMIC_GEMM_RS if ub_atomic_gemm_rs else None
            ub_algo=tex.UbufOverlapAlgo.SPLIT_PIPELINED_RS if ub_split_rs else ub_algo
            if FW_EMULATE_FP8 > 0:
                with emulate_fp8():
                    gelu_out_tmp = torch.ops.tex_ts.cast_from_fp8_ts(
                        gelu_out,
                        fp8_meta["scaling_fwd"].scale_inv,
                        tex.FP8FwdTensors.GEMM2_INPUT,
                        fp8_dtype_forward,
                        TE_DType[torch.bfloat16],
                    )
                    w_tmp = torch.ops.tex_ts.cast_from_fp8_ts(
                        fc2_weight_fp8._data,
                        fp8_meta["scaling_fwd"].scale_inv,
                        tex.FP8FwdTensors.GEMM2_WEIGHT,
                        fp8_dtype_forward,
                        TE_DType[torch.bfloat16],
                    )
                    fc2_out = gelu_out_tmp @ w_tmp.T
                if use_fc2_bias:
                    fc2_out += fc2_bias
            else:
                _ = tex.fp8_gemm(
                    fc2_weight_fp8._data,
                    fp8_meta["scaling_fwd"].scale_inv,
                    tex.FP8FwdTensors.GEMM2_WEIGHT,
                    fp8_dtype_forward,
                    gelu_out,
                    fp8_meta["scaling_fwd"].scale_inv,
                    tex.FP8FwdTensors.GEMM2_INPUT,
                    fp8_dtype_forward,
                    out_type,
                    get_workspace(),
                    bias=fc2_bias,
                    use_bias=use_fc2_bias,
                    use_split_accumulator=_2X_ACC_FPROP,
                    out=fc2_out,
                    ub_algo=ub_algo,
                    ub=ub_obj_fc2out if ub_split_rs or ub_atomic_gemm_rs else None,
                    extra_output_tensor=rs_out if ub_split_rs or ub_atomic_gemm_rs else None,
                    out_index=fc2_out_index,
                    fp8_meta_tensor = fc2_meta_tensor,
                    D_dtype = fc2_te_type,
                )
            if not is_grad_enabled:
                clear_tensor_data(gelu_out)
        else:
            # Cast for native AMP
            fc1_weight = cast_if_needed(fc1_weight, activation_dtype)
            fc2_weight = cast_if_needed(fc2_weight, activation_dtype)
            fc1_bias = (
                cast_if_needed(fc1_bias, activation_dtype) if use_fc1_bias else fc1_bias
            )
            fc2_bias = (
                cast_if_needed(fc2_bias, activation_dtype) if use_fc2_bias else fc2_bias
            )

            if fp8_calibration:
                # amax of fc1 input
                amin, amax = ln_out_total.aminmax()
                fp8_meta["scaling_fwd"].amax_history[0][tex.FP8FwdTensors.GEMM1_INPUT] = \
                    torch.max(-amin, amax).float()
                # amax of fc1 weight
                amin, amax = fc1_weight.aminmax()
                fp8_meta["scaling_fwd"].amax_history[0][tex.FP8FwdTensors.GEMM1_WEIGHT] = \
                    torch.max(-amin, amax).float()

            fc1_outputs = tex.gemm(
                fc1_weight,
                ln_out_total,
                activation_dtype,
                get_workspace(),
                bias=fc1_bias,
                use_bias=(not bias_gelu_nvfusion) and use_fc1_bias,
                gelu=not bias_gelu_nvfusion and (activation == 'gelu'),
                ub_algo=tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG if ub_split_ag else None,
                ub=ub_obj_lnout if ub_split_ag else None,
                extra_output_tensor=ln_out if ub_split_ag else None,
            )
            if not is_grad_enabled:
                clear_tensor_data(ln_out_total)

            if bias_gelu_nvfusion:
                fc1_out, _, _ = fc1_outputs
                gelu_out = bias_gelu_fused(fc1_out, fc1_bias)
            else:
                if activation == 'gelu':
                    gelu_out, _, fc1_out = fc1_outputs
                else:
                    fc1_out, _, _ = fc1_outputs
                    gelu_out = activation_func(fc1_out,
                                               None,
                                               tex.FP8FwdTensors.GEMM2_INPUT,
                                               TE_DType[fc1_out.dtype])
            if not is_grad_enabled:
                clear_tensor_data(fc1_out)

            if fp8_calibration:
                # amax of fc2 input
                amin, amax = gelu_out.aminmax()
                fp8_meta["scaling_fwd"].amax_history[0][tex.FP8FwdTensors.GEMM2_INPUT] = \
                    torch.max(-amin, amax).float()
                # amax of fc2 weight
                amin, amax = fc2_weight.aminmax()
                fp8_meta["scaling_fwd"].amax_history[0][tex.FP8FwdTensors.GEMM2_WEIGHT] = \
                    torch.max(-amin, amax).float()

            if ub_split_rs:
                ub_obj_fc2out = get_ub("fc2_fprop")
                fc2_out = ub_obj_fc2out.get_ubuf_output(1)
                dim_size = list(gelu_out.size())
                dim_size[0] = dim_size[0] // tp_world_size
                dim_size[1] = fc2_weight.size(0)
                rs_out = torch.empty(dim_size, dtype=activation_dtype, device=gelu_out.device)
            else:
                dim_size = list(gelu_out.size())
                dim_size[1] = fc2_weight.size(0)
                fc2_out = torch.empty(dim_size, dtype=activation_dtype, device=gelu_out.device)
            _ = tex.gemm(
                fc2_weight,
                gelu_out,
                activation_dtype,
                get_workspace(),
                bias=fc2_bias,
                use_bias=use_fc2_bias,
                out=fc2_out,
                ub_algo=tex.UbufOverlapAlgo.SPLIT_PIPELINED_RS if ub_split_rs else None,
                ub=ub_obj_fc2out if ub_split_rs else None,
                extra_output_tensor=rs_out if ub_split_rs else None,
            )
            if not is_grad_enabled:
                clear_tensor_data(gelu_out)

        if is_grad_enabled:
            if cpu_offloading:
                if fuse_wgrad_accumulation:
                    fc1_weight.main_grad.weight_offloading = True
                    fc2_weight.main_grad.weight_offloading = True
                if fp8 and fc1_weight_t_fp8 is not None:
                    fc1_weight_t_fp8.weight_offloading = True
                if fp8 and fc2_weight_t_fp8 is not None:
                    fc2_weight_t_fp8.weight_offloading = True
                ln_weight.weight_offloading = True
                fc1_weight.weight_offloading = True
                fc2_weight.weight_offloading = True
                fc1_bias.weight_offloading = True

                inputmat.activation_offloading = True
                if normalization == "LayerNorm":
                    mu.activation_offloading = True
                rsigma.activation_offloading = True
                ln_out.activation_offloading = True
                fc1_out.activation_offloading = True
                gelu_out.activation_offloading = True

            ctx.save_for_backward(
                inputmat,
                ln_weight,
                mu,
                rsigma,
                ln_out,
                fc1_out,
                gelu_out,
                fc1_weight,
                fc1_weight.main_grad if (cpu_offloading and fuse_wgrad_accumulation) else None,
                fc1_weight_t_fp8,
                fc2_weight,
                fc2_weight.main_grad if (cpu_offloading and fuse_wgrad_accumulation) else None,
                fc2_weight_t_fp8,
                fc1_bias,
                fp8_meta["scaling_fwd"].scale_inv.clone() if fp8 else None,
                had,
            )
            ctx.activation_dtype = activation_dtype
            ctx.activation = activation
            ctx.fp8 = fp8
            ctx.fp8_meta = fp8_meta
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.cpu_offloading = cpu_offloading
            ctx.is_first_microbatch = is_first_microbatch
            ctx.use_fc1_bias = use_fc1_bias
            ctx.use_fc2_bias = use_fc2_bias
            ctx.sequence_parallel = sequence_parallel
            ctx.tensor_parallel = tensor_parallel
            ctx.inp_shape = inp.shape
            ctx.tp_group = tp_group
            ctx.tp_size = tp_size
            ctx.bias_gelu_nvfusion = bias_gelu_nvfusion
            ctx.return_layernorm_output = return_layernorm_output
            ctx.return_layernorm_output_gathered = return_layernorm_output_gathered \
                                                   and ln_out_gathered
            ctx.set_parallel_mode = set_parallel_mode
            ctx.bwd_ln_sm_margin = bwd_ln_sm_margin
            ctx.zero_centered_gamma = zero_centered_gamma
            ctx.ub_bulk_wgrad = ub_bulk_wgrad
            ctx.ub_bulk_dgrad = ub_bulk_dgrad
            ctx.ub_split_ag = ub_split_ag
            ctx.ub_atomic_gemm_ag = ub_atomic_gemm_ag
            ctx.requires_dgrad = inp.requires_grad
            ctx.normalization = normalization

        # Row Parallel Linear
        if ub_split_rs or ub_atomic_gemm_rs:
            fc2_out = rs_out
        elif set_parallel_mode and sequence_parallel:
            fc2_out, _ = reduce_scatter_along_first_dim(fc2_out, tp_group)
        elif set_parallel_mode and tensor_parallel:
            fc2_out, _ = allreduce(fc2_out, tp_group)

        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        fc2_out = fc2_out.view(-1, *inp.shape[1:-1], fc2_out.shape[-1])

        if return_layernorm_output:
            if return_layernorm_output_gathered:
                shape = list(inp.shape)
                shape[0] *= tp_size
                return fc2_out, ln_out_return.view(shape)
            return fc2_out, ln_out_return.view_as(inp)
        return fc2_out


    @staticmethod
    def backward(
        ctx, *grad_outputs: Tuple[torch.Tensor, ...]
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        with _prepare_backward(
            ctx.fp8, ctx.fp8_meta, ctx.tp_group, ctx.tp_size, name="_LayerNormMLP"
        ):
            (
                inputmat,
                ln_weight,
                mu,
                rsigma,
                ln_out,
                fc1_out,
                gelu_out,
                fc1_weight,
                fc1_weight_main_grad,
                fc1_weight_t_fp8,
                fc2_weight,
                fc2_weight_main_grad,
                fc2_weight_t_fp8,
                fc1_bias,
                fwd_scale_inverses,
                had,
            ) = ctx.saved_tensors

            if ctx.cpu_offloading and ctx.fuse_wgrad_accumulation:
                fc1_weight = Parameter(fc1_weight, False)
                fc2_weight = Parameter(fc2_weight, False)

                fc1_weight.main_grad = fc1_weight_main_grad
                fc2_weight.main_grad = fc2_weight_main_grad

            # Primary weights are in FP8.
            update_transpose_cache = "reuse_only" if ctx.is_first_microbatch is None else "lazy"
            if ctx.fp8 and fc1_weight_t_fp8 is None:
                fc1_weight_t_fp8 = fc1_weight.transpose(update_cache=update_transpose_cache)
            if ctx.fp8 and fc2_weight_t_fp8 is None:
                fc2_weight_t_fp8 = fc2_weight.transpose(update_cache=update_transpose_cache)

            activation_func = _act_func(ctx.activation)[1]

            if ctx.ub_bulk_dgrad:
                tp_world_size = get_distributed_world_size(ctx.tp_group)
                if tp_world_size == 1:
                    ctx.ub_bulk_dgrad = False
            if ctx.ub_bulk_dgrad:
                dim_size = list(ln_out.size())
                dim_size[0] = dim_size[0] * tp_world_size
                ub_obj_lnout = get_ub("fc1_dgrad")
                ub_obj_lnout.copy_input_to_ubuf(ln_out, 1)
            ub_overlap_ag = ctx.ub_split_ag or ctx.ub_atomic_gemm_ag
            if ub_overlap_ag:
                tp_world_size = get_distributed_world_size(ctx.tp_group)
                if tp_world_size == 1:
                    ctx.ub_split_ag = False
                    ctx.ub_overlap_ag = False
            ub_overlap_ag = ctx.ub_split_ag or ctx.ub_atomic_gemm_ag

            if ub_overlap_ag:
                dim_size = list(grad_outputs[0].size())
                dim_size[0] = dim_size[0] * tp_world_size
                ctx.ub_obj_gradout = get_ub("fc2_dgrad")

            ctx.use_bias = ctx.use_fc2_bias # For grad_output_preprocess
            (
                grad_output,
                grad_output_c,
                grad_output_t,
                fc2_bias_grad,
            ) = TransformerEngineBaseModule.grad_output_preprocess(
                ctx, grad_outputs[0], True
            )

            if ctx.ub_bulk_wgrad:
                tp_world_size = get_distributed_world_size(ctx.tp_group)
                if tp_world_size == 1 or not fc1_weight.requires_grad:
                    ctx.ub_bulk_wgrad = False
            # Column Parallel Linear
            # Overlap input AG with dgrad
            if (fc1_weight.requires_grad
                and (not ctx.ub_bulk_dgrad)
                and ctx.set_parallel_mode
                and ctx.sequence_parallel):
                ln_out_total, handle = gather_along_first_dim(
                    ln_out, ctx.tp_group, async_op=True
                )
            else:
                ln_out_total = ln_out
                handle = None

            if ctx.is_first_microbatch is not None:
                accumulate_wgrad_into_param_main_grad = (
                    ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
                )
            else:
                accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

            if ctx.fp8:
                fp8_dtype_forward = get_fp8_te_dtype(
                    ctx.fp8_meta["recipe"], fprop_tensor=True
                )
                fp8_dtype_backward = get_fp8_te_dtype(
                    ctx.fp8_meta["recipe"], fprop_tensor=False
                )

                ub_algo = tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG if ctx.ub_split_ag else None
                ub_algo = tex.UbufOverlapAlgo.ATOMIC_GEMM_AG if ctx.ub_atomic_gemm_ag else ub_algo
                # FC2 DGRAD; Unconditional
                if BW_USE_MXFP4 == 1 or BW_USE_MXFP4 == 3:
                    fc2_wt_tmp = torch.ops.tex_ts.cast_from_fp8_ts(
                        fc2_weight_t_fp8._data,
                        fwd_scale_inverses,
                        tex.FP8FwdTensors.GEMM2_WEIGHT,
                        fp8_dtype_forward,
                        TE_DType[had.dtype])
                    go_tmp = torch.ops.tex_ts.cast_from_fp8_ts(
                        grad_output_c,
                        ctx.fp8_meta["scaling_bwd"].scale_inv,
                        tex.FP8BwdTensors.GRAD_OUTPUT1,
                        fp8_dtype_backward,
                        TE_DType[had.dtype])
                    S = (had.T * ((torch.randn(HBS, device=had.device) > 0).to(had.dtype) * 2 - 1)).T
                    fc2_dgrad = mx.matmul(
                        (go_tmp.reshape(-1, HBS) @ S).reshape(go_tmp.shape),
                        (fc2_wt_tmp.reshape(-1, HBS) @ S).reshape(fc2_wt_tmp.shape).T,
                        mx_specs=mxfp4)
                elif BW_USE_MXFP4 == 2 or BW_USE_MXFP4 == 4:
                    fc2_wt_tmp = torch.ops.tex_ts.cast_from_fp8_ts(
                        fc2_weight_t_fp8._data,
                        fwd_scale_inverses,
                        tex.FP8FwdTensors.GEMM2_WEIGHT,
                        fp8_dtype_forward,
                        TE_DType[had.dtype])
                    go_tmp = torch.ops.tex_ts.cast_from_fp8_ts(
                        grad_output_c,
                        ctx.fp8_meta["scaling_bwd"].scale_inv,
                        tex.FP8BwdTensors.GRAD_OUTPUT1,
                        fp8_dtype_backward,
                        TE_DType[had.dtype])
                    dgrad = mx.matmul(go_tmp, fc2_wt_tmp.T, mx_specs=mxfp4)
                else:
                    fc2_dgrad, _ = tex.fp8_gemm(
                        fc2_weight_t_fp8._data,
                        fwd_scale_inverses,
                        tex.FP8FwdTensors.GEMM2_WEIGHT,
                        fp8_dtype_forward,
                        grad_output_c,
                        ctx.fp8_meta["scaling_bwd"].scale_inv,
                        tex.FP8BwdTensors.GRAD_OUTPUT1,
                        fp8_dtype_backward,
                        ctx.activation_dtype,
                        get_workspace(),
                        use_split_accumulator=_2X_ACC_DGRAD,
                        ub_algo=ub_algo,
                        ub=ctx.ub_obj_gradout if ub_overlap_ag else None,
                    )
                if ub_overlap_ag:
                    grad_output_t = tex.fp8_transpose(grad_output_c, fp8_dtype_backward)
                clear_tensor_data(grad_output_c)

                # FC2 WGRAD
                if not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                    if fc2_weight.requires_grad:
                        gelu_out_t = tex.fp8_transpose(gelu_out, fp8_dtype_forward)
                        clear_tensor_data(gelu_out)
                        if BW_USE_MXFP4 == 1 or BW_USE_MXFP4 == 3:
                            glot_tmp = torch.ops.tex_ts.cast_from_fp8_ts(
                                gelu_out_t,
                                fwd_scale_inverses,
                                tex.FP8FwdTensors.GEMM2_INPUT,
                                fp8_dtype_forward,
                                TE_DType[had.dtype])
                            got_tmp = torch.ops.tex_ts.cast_from_fp8_ts(
                                grad_output_t,
                                ctx.fp8_meta["scaling_bwd"].scale_inv,
                                tex.FP8BwdTensors.GRAD_OUTPUT1,
                                fp8_dtype_backward,
                                TE_DType[had.dtype])
                            S = (had.T * ((torch.randn(HBS, device=had.device) > 0).to(had.dtype) * 2 - 1)).T
                            fc2_wgrad = mx.matmul(
                                (got_tmp.reshape(-1, HBS) @ S).reshape(got_tmp.shape),
                                (glot_tmp.reshape(-1, HBS) @ S).reshape(glot_tmp.shape).T,
                                mx_specs=mxfp4)
                        elif BW_USE_MXFP4 == 2 or BW_USE_MXFP4 == 4:
                            glot_tmp = torch.ops.tex_ts.cast_from_fp8_ts(
                                gelu_out_t,
                                fwd_scale_inverses,
                                tex.FP8FwdTensors.GEMM2_INPUT,
                                fp8_dtype_forward,
                                TE_DType[had.dtype])
                            got_tmp = torch.ops.tex_ts.cast_from_fp8_ts(
                                grad_output_t,
                                ctx.fp8_meta["scaling_bwd"].scale_inv,
                                tex.FP8BwdTensors.GRAD_OUTPUT1,
                                fp8_dtype_backward,
                                TE_DType[had.dtype])
                            fc2_wgrad = mx.matmul(got_tmp, glot_tmp.T, mx_specs=mxfp4)
                        else:
                            fc2_wgrad, _ = tex.fp8_gemm(
                                gelu_out_t,
                                fwd_scale_inverses,
                                tex.FP8FwdTensors.GEMM2_INPUT,
                                fp8_dtype_forward,
                                grad_output_t,
                                ctx.fp8_meta["scaling_bwd"].scale_inv,
                                tex.FP8BwdTensors.GRAD_OUTPUT1,
                                fp8_dtype_backward,
                                ctx.activation_dtype,
                                get_workspace(),
                                accumulate=accumulate_wgrad_into_param_main_grad,
                                out=fc2_weight.main_grad
                                if ctx.fuse_wgrad_accumulation
                                else None,
                                use_split_accumulator=_2X_ACC_WGRAD,
                            )
                        clear_tensor_data(gelu_out_t, grad_output_t)

                    if ctx.activation == 'gelu':
                        fc1_bias_grad, dgelu, dgelu_t = tex.fp8_cast_transpose_bgrad_dgelu_fused(
                            fc2_dgrad,
                            fc1_out,
                            ctx.fp8_meta["scaling_bwd"],
                            tex.FP8BwdTensors.GRAD_OUTPUT2,
                            fp8_dtype_backward,
                        )
                    else:
                        dgelu = activation_func(fc2_dgrad, fc1_out,
                                                TE_DType[fc2_dgrad.dtype])
                        fc1_bias_grad, dgelu, dgelu_t = tex.fp8_cast_transpose_bgrad_fused(
                            dgelu,
                            ctx.fp8_meta["scaling_bwd"],
                            tex.FP8BwdTensors.GRAD_OUTPUT2,
                            fp8_dtype_backward,
                        )
                    clear_tensor_data(fc1_out)
                else:
                    if fc2_weight.requires_grad:
                        gelu_out_c = torch.ops.tex_ts.cast_from_fp8_ts(
                            gelu_out,
                            fwd_scale_inverses,
                            tex.FP8FwdTensors.GEMM2_INPUT,
                            fp8_dtype_forward,
                            TE_DType[ctx.activation_dtype],
                        )
                        clear_tensor_data(gelu_out)
                        fc2_wgrad, _, _ = tex.gemm(
                            gelu_out_c,
                            grad_output,
                            ctx.activation_dtype,
                            get_workspace(),
                            layout="NT",
                            grad=True,
                            use_bias=False,
                            accumulate=accumulate_wgrad_into_param_main_grad,
                            out=fc2_weight.main_grad
                            if ctx.fuse_wgrad_accumulation
                            else None,
                        )
                        clear_tensor_data(gelu_out_c)

                    if ctx.activation == 'gelu':
                        fc1_bias_grad, dgelu_no_fp8 = bgrad_dgelu_fused(
                            fc2_dgrad, fc1_out, fc1_bias
                        )
                    else:
                        dgelu_no_fp8 = activation_func(fc2_dgrad,
                                                       fc1_out,
                                                       TE_DType[fc2_dgrad.dtype])
                        fc1_bias_grad = dgelu_no_fp8.sum(dim=0)
                    clear_tensor_data(fc1_out)

                    dgelu = tex.cast_to_fp8(
                        dgelu_no_fp8,
                        ctx.fp8_meta["scaling_bwd"],
                        tex.FP8BwdTensors.GRAD_OUTPUT2,
                        fp8_dtype_backward,
                    )
                    dgelu_t = None

                out_index, meta_tensor, out_te_type, out_type = (
                    None, None, None, ctx.activation_dtype)
                fc1_dgrad_size = list(dgelu.size())
                fc1_dgrad_size[1] = fc1_weight.size(1)
                if ctx.ub_bulk_wgrad: # allocate dgrad output
                    ub_obj_dgrad = get_ub("fc1_wgrad")
                    fc1_dgrad = ub_obj_dgrad.get_ubuf_output(1) # AllGather output
                    if ub_obj_dgrad.is_fp8_ubuf():
                        out_index = tex.FP8BwdTensors.GRAD_INPUT2
                        meta_tensor = ctx.fp8_meta["scaling_bwd"]
                        out_te_type = fp8_dtype_backward
                        out_type = torch.uint8
                        ub_obj_dgrad.set_ubuf_scale_inv(meta_tensor.scale_inv[out_index])
                else:
                    fc1_dgrad = torch.empty(
                        fc1_dgrad_size, dtype=ctx.activation_dtype, device=fc1_weight.device
                    )
                if BW_USE_MXFP4 == 1 or BW_USE_MXFP4 == 3:
                    fc1_wt_tmp = torch.ops.tex_ts.cast_from_fp8_ts(
                        fc1_weight_t_fp8._data,
                        fwd_scale_inverses,
                        tex.FP8FwdTensors.GEMM1_WEIGHT,
                        fp8_dtype_forward,
                        TE_DType[had.dtype])
                    dgelu_tmp = torch.ops.tex_ts.cast_from_fp8_ts(
                        dgelu,
                        ctx.fp8_meta["scaling_bwd"].scale_inv,
                        tex.FP8BwdTensors.GRAD_OUTPUT2,
                        fp8_dtype_backward,
                        TE_DType[had.dtype])
                    S = (had.T * ((torch.randn(HBS, device=had.device) > 0).to(had.dtype) * 2 - 1)).T
                    fc1_dgrad = mx.matmul(
                        (dgelu_tmp.reshape(-1, HBS) @ S).reshape(dgelu_tmp.shape),
                        (fc1_wt_tmp.reshape(-1, HBS) @ S).reshape(fc1_wt_tmp.shape).T,
                        mx_specs=mxfp4)
                elif BW_USE_MXFP4 == 2 or BW_USE_MXFP4 == 4:
                    fc1_wt_tmp = torch.ops.tex_ts.cast_from_fp8_ts(
                        fc1_weight_t_fp8._data,
                        fwd_scale_inverses,
                        tex.FP8FwdTensors.GEMM1_WEIGHT,
                        fp8_dtype_forward,
                        TE_DType[had.dtype])
                    dgelu_tmp = torch.ops.tex_ts.cast_from_fp8_ts(
                        dgelu,
                        ctx.fp8_meta["scaling_bwd"].scale_inv,
                        tex.FP8BwdTensors.GRAD_OUTPUT2,
                        fp8_dtype_backward,
                        TE_DType[had.dtype])
                    fc1_dgrad = mx.matmul(dgelu_tmp, fc1_wt_tmp.T, mx_specs=mxfp4)
                else:
                    # FC1 DGRAD: Unconditional
                    _ = tex.fp8_gemm(
                        fc1_weight_t_fp8._data,
                        fwd_scale_inverses,
                        tex.FP8FwdTensors.GEMM1_WEIGHT,
                        fp8_dtype_forward,
                        dgelu,
                        ctx.fp8_meta["scaling_bwd"].scale_inv,
                        tex.FP8BwdTensors.GRAD_OUTPUT2,
                        fp8_dtype_backward,
                        out_type,
                        get_workspace(),
                        out=fc1_dgrad,
                        use_split_accumulator=_2X_ACC_DGRAD,
                        ub_algo=tex.UbufOverlapAlgo.BULK_OVERLAP_AG if ctx.ub_bulk_dgrad else None,
                        ub=ub_obj_lnout if ctx.ub_bulk_dgrad else None,
                        out_index=out_index,
                        fp8_meta_tensor = meta_tensor,
                        D_dtype = out_te_type,
                    )
            else:
                # FC2 DGRAD; Unconditional
                fc2_dgrad, _, _ = tex.gemm(
                    fc2_weight,
                    grad_output,
                    ctx.activation_dtype,
                    get_workspace(),
                    layout="NN",
                    gelu=(not ctx.bias_gelu_nvfusion) and (ctx.activation == 'gelu'),
                    grad=True,
                    gelu_input=fc1_out,
                    ub_algo=tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG if ctx.ub_split_ag else None,
                    ub=ctx.ub_obj_gradout if ctx.ub_split_ag else None,
                )

                # FC2 WGRAD
                if fc2_weight.requires_grad:
                    fc2_wgrad, fc2_bias_grad, _ = tex.gemm(
                        gelu_out,
                        grad_output,
                        ctx.activation_dtype,
                        get_workspace(),
                        layout="NT",
                        grad=True,
                        use_bias=ctx.use_fc2_bias,
                        accumulate=accumulate_wgrad_into_param_main_grad,
                        out=fc2_weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                    )
                clear_tensor_data(gelu_out)

                if ctx.bias_gelu_nvfusion and ctx.activation == 'gelu':
                    fc1_bias_grad, fc2_dgrad = bgrad_dgelu_fused(fc2_dgrad, fc1_out, fc1_bias)
                else:
                    if ctx.activation != 'gelu':
                        fc2_dgrad = activation_func(fc2_dgrad,
                                                    fc1_out,
                                                    TE_DType[fc2_dgrad.dtype])

                    # For non-fp8 execution, FC1 bias gradient is fused with FC1 wgrad GEMM
                    # and will not be calculated in case wgrad is not required.
                    if not fc1_weight.requires_grad:
                        fc1_bias_grad = fc2_dgrad.sum(dim=0)

                # Overwrite data. Deleting the tensor does not release underlying memory.
                clear_tensor_data(fc1_out)
                dgelu = fc2_dgrad

                fc1_dgrad_size = list(dgelu.size())
                fc1_dgrad_size[1] = fc1_weight.size(1)
                if ctx.ub_bulk_wgrad: # allocate dgrad output
                    ub_obj_dgrad = get_ub("fc1_wgrad")
                    fc1_dgrad = ub_obj_dgrad.get_ubuf_output(1) # AllGather output
                else:
                    fc1_dgrad = torch.empty(
                        fc1_dgrad_size, dtype=ctx.activation_dtype, device=fc1_weight.device
                    )

                # FC1 DGRAD: Unconditional
                _ = tex.gemm(
                    fc1_weight,
                    dgelu,
                    ctx.activation_dtype,
                    get_workspace(),
                    out=fc1_dgrad,
                    layout="NN",
                    grad=True,
                    ub_algo=tex.UbufOverlapAlgo.BULK_OVERLAP_AG if ctx.ub_bulk_dgrad else None,
                    ub=ub_obj_lnout if ctx.ub_bulk_dgrad else None
                )

            if ctx.ub_bulk_dgrad:
                ln_out_total = ub_obj_lnout.get_ubuf_output(1)
            # Overlap dgrad-RS/AR with wgrad
            if ctx.set_parallel_mode and ctx.sequence_parallel:
                if not ctx.ub_bulk_dgrad and handle is not None:
                    handle.wait()
                if not ctx.ub_bulk_wgrad:
                    if ctx.return_layernorm_output and ctx.return_layernorm_output_gathered:
                        fc1_dgrad = fc1_dgrad + grad_outputs[1].view_as(fc1_dgrad)
                    fc1_dgrad, handle = reduce_scatter_along_first_dim(
                        fc1_dgrad, ctx.tp_group, async_op=True
                    )
            elif ctx.set_parallel_mode and ctx.tensor_parallel:
                fc1_dgrad, handle = allreduce(fc1_dgrad, ctx.tp_group, async_op=True)

            if fc1_weight.requires_grad:
                if ctx.fp8:
                    # FC1 WGRAD
                    extra_output_tensor = None
                    if ctx.ub_bulk_wgrad:
                        if ub_obj_dgrad.is_fp8_ubuf():
                            dim_size = list(ub_obj_dgrad.get_ubuf_output(0).size()) # RS output
                            extra_output_tensor = torch.empty(
                                dim_size, dtype=ctx.activation_dtype, device=fc1_dgrad.device)
                            fc1_dgrad = extra_output_tensor
                        else:
                            fc1_dgrad = ub_obj_dgrad.get_ubuf_output(0)
                    if not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                        ln_out_total_t = tex.fp8_transpose(ln_out_total, fp8_dtype_forward)

                        if BW_USE_MXFP4 == 1 or BW_USE_MXFP4 == 3:
                            lnot_tmp = torch.ops.tex_ts.cast_from_fp8_ts(
                                ln_out_total_t,
                                fwd_scale_inverses,
                                tex.FP8FwdTensors.GEMM1_INPUT,
                                fp8_dtype_forward,
                                TE_DType[had.dtype])
                            dgelut_tmp = torch.ops.tex_ts.cast_from_fp8_ts(
                                dgelu_t,
                                ctx.fp8_meta["scaling_bwd"].scale_inv,
                                tex.FP8BwdTensors.GRAD_OUTPUT2,
                                fp8_dtype_backward,
                                TE_DType[had.dtype])
                            S = (had.T * ((torch.randn(HBS, device=had.device) > 0).to(had.dtype) * 2 - 1)).T
                            fc1_wgrad = mx.matmul(
                                (dgelut_tmp.reshape(-1, HBS) @ S).reshape(dgelut_tmp.shape),
                                (lnot_tmp.reshape(-1, HBS) @ S).reshape(lnot_tmp.shape).T,
                                mx_specs=mxfp4)
                        elif BW_USE_MXFP4 == 2 or BW_USE_MXFP4 == 4:
                            lnot_tmp = torch.ops.tex_ts.cast_from_fp8_ts(
                                ln_out_total_t,
                                fwd_scale_inverses,
                                tex.FP8FwdTensors.GEMM1_INPUT,
                                fp8_dtype_forward,
                                TE_DType[had.dtype])
                            dgelut_tmp = torch.ops.tex_ts.cast_from_fp8_ts(
                                dgelu_t,
                                ctx.fp8_meta["scaling_bwd"].scale_inv,
                                tex.FP8BwdTensors.GRAD_OUTPUT2,
                                fp8_dtype_backward,
                                TE_DType[had.dtype])
                            fc1_wgrad = mx.matmul(dgelut_tmp, lnot_tmp.T, mx_specs=mxfp4)
                        else:
                            fc1_wgrad, _ = tex.fp8_gemm(
                                ln_out_total_t,
                                fwd_scale_inverses,
                                tex.FP8FwdTensors.GEMM1_INPUT,
                                fp8_dtype_forward,
                                dgelu_t,
                                ctx.fp8_meta["scaling_bwd"].scale_inv,
                                tex.FP8BwdTensors.GRAD_OUTPUT2,
                                fp8_dtype_backward,
                                ctx.activation_dtype,
                                get_workspace(),
                                accumulate=accumulate_wgrad_into_param_main_grad,
                                out=fc1_weight.main_grad
                                if ctx.fuse_wgrad_accumulation
                                else None,
                                use_split_accumulator=_2X_ACC_WGRAD,
                                ub_algo=tex.UbufOverlapAlgo.BULK_OVERLAP_RS
                                if ctx.ub_bulk_wgrad else None,
                                ub=ub_obj_dgrad if ctx.ub_bulk_wgrad else None,
                                extra_output_tensor=extra_output_tensor,
                            )
                        clear_tensor_data(ln_out_total_t, dgelu_t)
                    else:
                        ln_out_total_c = torch.ops.tex_ts.cast_from_fp8_ts(
                            ln_out_total,
                            fwd_scale_inverses,
                            tex.FP8FwdTensors.GEMM1_INPUT,
                            fp8_dtype_forward,
                            TE_DType[ctx.activation_dtype],
                        )
                        fc1_wgrad, _, _ = tex.gemm(
                            ln_out_total_c,
                            dgelu_no_fp8,
                            ctx.activation_dtype,
                            get_workspace(),
                            layout="NT",
                            grad=True,
                            accumulate=accumulate_wgrad_into_param_main_grad,
                            out=fc1_weight.main_grad
                            if ctx.fuse_wgrad_accumulation
                            else None,
                            ub_algo=tex.UbufOverlapAlgo.BULK_OVERLAP_RS
                            if ctx.ub_bulk_wgrad else None,
                            ub=ub_obj_dgrad if ctx.ub_bulk_wgrad else None,
                            extra_output_tensor=extra_output_tensor,
                        )
                        clear_tensor_data(ln_out_total_c, dgelu_no_fp8)
                else:
                    # FC1 WGRAD
                    fc1_wgrad_outputs = tex.gemm(
                        ln_out_total,
                        dgelu,
                        ctx.activation_dtype,
                        get_workspace(),
                        layout="NT",
                        grad=True,
                        use_bias=not ctx.bias_gelu_nvfusion,
                        accumulate=accumulate_wgrad_into_param_main_grad,
                        out=fc1_weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                        ub_algo=tex.UbufOverlapAlgo.BULK_OVERLAP_RS if ctx.ub_bulk_wgrad else None,
                        ub=ub_obj_dgrad if ctx.ub_bulk_wgrad else None
                    )
                    clear_tensor_data(ln_out_total, dgelu)

                    if ctx.bias_gelu_nvfusion:
                        fc1_wgrad, _, _ = fc1_wgrad_outputs
                    else:
                        fc1_wgrad, fc1_bias_grad, _ = fc1_wgrad_outputs
                    if ctx.ub_bulk_wgrad:
                        fc1_dgrad = ub_obj_dgrad.get_ubuf_output(0) # Reduce-scatter output

            # Column Parallel Linear
            if ((not ctx.ub_bulk_wgrad)
                and ctx.set_parallel_mode
                and ctx.tensor_parallel
                and handle is not None):
                handle.wait()

            # LayerNorm gradient
            dgrad = fc1_dgrad.view(inputmat.shape)

            # Residual gradient
            if ctx.return_layernorm_output and not ctx.return_layernorm_output_gathered:
                dgrad = dgrad + grad_outputs[1].view_as(dgrad)

            if ctx.normalization == "LayerNorm":
                dgrad, dgamma, dbeta = tex.layernorm_bwd(
                    dgrad, inputmat, mu, rsigma, ln_weight,
                    ctx.bwd_ln_sm_margin, ctx.zero_centered_gamma
                )
            elif ctx.normalization == "RMSNorm":
                dgrad, dgamma = tex.rmsnorm_bwd(
                    dgrad, inputmat, rsigma, ln_weight,
                    ctx.bwd_ln_sm_margin, ctx.zero_centered_gamma
                )
                dbeta = None

        if fc1_weight.requires_grad:
            # Handle custom DDP from mcore.
            if ctx.fuse_wgrad_accumulation and hasattr(fc1_weight, 'grad_added_to_main_grad'):
                fc1_weight.grad_added_to_main_grad = True
                if getattr(fc1_weight, 'zero_out_wgrad', False):
                    fc1_wgrad = torch.zeros(fc1_weight.main_grad.shape,
                                            dtype=fc1_weight.dtype,
                                            device=torch.cuda.current_device(),
                                            requires_grad=False
                                           )
                else:
                    fc1_wgrad = torch.empty(fc1_weight.main_grad.shape,
                                            dtype=fc1_weight.dtype,
                                            device=torch.cuda.current_device(),
                                            requires_grad=False
                                           )
            elif ctx.fuse_wgrad_accumulation:
                fc1_wgrad = None
        else:
            fc1_wgrad = None

        if fc2_weight.requires_grad:
            # Handle custom DDP from mcore.
            if ctx.fuse_wgrad_accumulation and hasattr(fc2_weight, 'grad_added_to_main_grad'):
                fc2_weight.grad_added_to_main_grad = True
                if getattr(fc2_weight, 'zero_out_wgrad', False):
                    fc2_wgrad = torch.zeros(fc2_weight.main_grad.shape,
                                            dtype=fc2_weight.dtype,
                                            device=torch.cuda.current_device(),
                                            requires_grad=False
                                           )
                else:
                    fc2_wgrad = torch.empty(fc2_weight.main_grad.shape,
                                            dtype=fc2_weight.dtype,
                                            device=torch.cuda.current_device(),
                                            requires_grad=False
                                           )
            elif ctx.fuse_wgrad_accumulation:
                fc2_wgrad = None
        else:
            fc2_wgrad = None

        return (
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
            dgamma,
            dbeta,
            fc1_wgrad,
            None,
            None,
            fc1_bias_grad if ctx.use_fc1_bias else None,
            None,
            fc2_wgrad,
            None,
            None,
            fc2_bias_grad if ctx.use_fc2_bias else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class LayerNormMLP(TransformerEngineBaseModule):
    r"""
    Applies layer normalization on the input followed by the MLP module, consisting of
    2 successive linear transformations, separated by the GeLU activation.

    Parameters
    ----------
    hidden_size : int
                 size of each input sample.
    ffn_hidden_size : int
                     intermediate size to which input samples are projected.
    eps : float, default = 1e-5
         a value added to the denominator of layer normalization for numerical stability.
    bias : bool, default = `True`
          if set to `False`, the FC1 and FC2 layers will not learn an additive bias.
    normalization : { 'LayerNorm', 'RMSNorm' }, default = 'LayerNorm'
                   type of normalization applied.
    activation : str, default = 'gelu'
          activation function used.
          Options: 'gelu', 'geglu', 'relu', 'reglu', 'squared_relu', 'swiglu', 'qgelu'.
    init_method : Callable, default = `None`
                 used for initializing FC1 weights in the following way: `init_method(weight)`.
                 When set to `None`, defaults to `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    output_layer_init_method : Callable, default = `None`
                              used for initializing FC2 weights in the following way:
                              `output_layer_init_method(weight)`. When set to `None`, defaults to
                              `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    return_layernorm_output : bool, default = `False`
                             if set to `True`, output of layernorm is returned from the forward
                             together with the output of the linear transformation.
                             Example use case: residual connection for transformer module
                             is taken post layernorm.
    return_layernorm_output_gathered : bool, default = `False`
                             if set to `True`, output of layernorm is returned after the all
                             gather operation. Ignored if return_layernorm_output is False.
                             Example use case: with sequence parallel, input to residual connection
                             for transformer module (e.g. LoRA) will need to be gathered.
                             Returning layernorm output gathered will prevent a redundant gather.
    zero_centered_gamma : bool, default = 'False'
                         if set to 'True', gamma parameter in LayerNorm is initialized to 0 and
                         the LayerNorm formula changes to

                         .. math::
                            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \varepsilon}} *
                            (1 + \gamma) + \beta
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will allocated. It is the user's
          responsibility to ensure all parameters are moved to the GPU before running the
          forward pass.

    Parallelism parameters
    ----------------------
    set_parallel_mode : bool, default = `False`
                      if set to `True`, FC1 is used as Column Parallel and FC2 is used as Row
                      Parallel as described `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    tp_size : int, default = 1
             used as TP (tensor parallel) world size when TP groups are not formed during
             initialization. In this case, users must call the
             `set_tensor_parallel_group(tp_group)` method on the initialized module before the
             forward pass to supply the tensor parallel group needed for tensor and sequence
             parallel collectives.

    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = 'False'
                             if set to `True`, enables fusing of creation and accumulation of
                             the weight gradient. When enabled, it is assumed that the weights
                             have an additional `main_grad` attribute (used instead of the
                             regular `grad`) which is a pre-allocated buffer of the correct
                             size to accumulate gradients in.
    return_bias : bool, default = `False`
                 when set to `True`, this module will not apply the additive bias for FC2, but
                 instead return the bias value during the forward pass together with the
                 output of the linear transformation :math:`y = xA^T`. This is useful when
                 the bias addition can be fused to subsequent operations.
    params_dtype : torch.dtype, default = `torch.get_default_dtype()`
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.
    seq_length: int
               sequence length of input samples. Needed for JIT Warmup, a technique where jit fused
               functions are warmed up before training to ensure same kernels are used for forward
               propogation and activation recompute phase.
    micro_batch_size: int
                     batch size per training step. Needed for JIT Warmup, a technique where jit
                     fused functions are warmed up before training to ensure same kernels are
                     used for forward propogation and activation recompute phase.
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        eps: float = 1e-5,
        sequence_parallel: bool = False,
        return_bias: bool = False,
        get_rng_state_tracker: Optional[Callable] = None,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        normalization: str = 'LayerNorm',
        activation : str = "gelu",
        output_layer_init_method: Optional[Callable] = None,
        fuse_wgrad_accumulation: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        return_layernorm_output: bool = False,
        return_layernorm_output_gathered: bool = False,
        seq_length: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
        set_parallel_mode: bool = False,
        zero_centered_gamma: bool = False,
        device: Union[torch.device, str] = "cuda",
        ub_bulk_wgrad: bool = False,
        ub_bulk_dgrad: bool = False,
        ub_split_rs: bool = False,
        ub_atomic_gemm_rs: bool = False,
        ub_split_ag: bool = False,
        ub_atomic_gemm_ag: bool = False,
    ) -> None:
        super().__init__()

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.normalization = normalization
        assert normalization in ['LayerNorm', 'RMSNorm'], "Unsupported normalization type!"
        self.use_bias = bias
        self.activation = activation
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.return_layernorm_output = return_layernorm_output
        self.return_layernorm_output_gathered = return_layernorm_output_gathered
        self.bias_gelu_nvfusion = (bool(int(os.getenv("NVTE_BIAS_GELU_NVFUSION", "1"))) and
                                   self.activation == 'gelu')
        self.set_parallel_mode = set_parallel_mode
        self.zero_centered_gamma = zero_centered_gamma
        self.primary_weights_in_fp8 = FP8GlobalStateManager.with_fp8_parameters()
        self.ub_bulk_wgrad = ub_bulk_wgrad
        self.ub_bulk_dgrad = ub_bulk_dgrad
        self.ub_split_rs = ub_split_rs
        self.ub_split_ag = ub_split_ag
        self.ub_atomic_gemm_rs = ub_atomic_gemm_rs
        self.ub_atomic_gemm_ag = ub_atomic_gemm_ag
        # GEMM-GELU fusion is currently only supported with split GEMM-AG overlap
        self.gemm_gelu_fusion = (bool(int(os.getenv("NVTE_GEMM_GELU_FUSION", "0"))) and
                                 self.activation == 'gelu' and self.ub_split_ag)

        if (ub_bulk_wgrad # pylint: disable=too-many-boolean-expressions
            or ub_bulk_dgrad
            or ub_split_rs
            or ub_split_ag
            or ub_atomic_gemm_rs
            or ub_atomic_gemm_ag):
            assert (
                tex.userbuf_comm_available()
            ), "Userbuffer communication backend not available."

        if ub_atomic_gemm_rs or ub_atomic_gemm_ag:
            warnings.warn(
                "Atomic gemm uses a beta API from cublas and is not tested for all use cases."
            )

        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)
        self.set_nccl_overlap_warning_if_tp()

        if init_method is None:
            init_method = get_default_init_method()
        if output_layer_init_method is None:
            output_layer_init_method = get_default_init_method()

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel
        self.size_per_partition = divide(ffn_hidden_size, self.tp_size)

        # LN init
        self.eps = eps
        layer_norm_weight = Parameter(
            torch.empty(hidden_size, device=device, dtype=params_dtype)
        )
        self.register_parameter('layer_norm_weight', layer_norm_weight,
                                init_fn=init_method_constant(float(not self.zero_centered_gamma)))
        if self.normalization != "RMSNorm":
            layer_norm_bias = Parameter(
                torch.empty(hidden_size, device=device, dtype=params_dtype)
            )
            self.register_parameter('layer_norm_bias', layer_norm_bias,
                                    init_fn=init_method_constant(0.0))
        else:
            self.layer_norm_bias = None

        # FC1 init
        if self.activation in ['reglu', 'geglu', 'swiglu']:
            fc1_output_features = 2 * self.size_per_partition
        else:
            fc1_output_features = self.size_per_partition

        fc1_weight = Parameter(
            torch.empty(
                fc1_output_features, hidden_size, device=device, dtype=params_dtype
            )
        )
        self.register_parameter('fc1_weight', fc1_weight,
                                init_fn=init_method,
                                get_rng_state_tracker=get_rng_state_tracker,
                                fp8_meta_index=tex.FP8FwdTensors.GEMM1_WEIGHT)
        self.fp8_weight_shapes.append(self.fc1_weight.shape)

        if self.use_bias:
            fc1_bias = Parameter(
                torch.empty(fc1_output_features, device=device, dtype=params_dtype)
            )
            self.register_parameter('fc1_bias', fc1_bias,
                                    init_fn=init_method_constant(0.0))
        else:
            self.fc1_bias = torch.Tensor().to(dtype=params_dtype, device=device)

        # FC2 init
        fc2_weight = Parameter(
            torch.empty(hidden_size, self.size_per_partition, device=device, dtype=params_dtype)
        )
        self.register_parameter('fc2_weight', fc2_weight,
                                init_fn=output_layer_init_method,
                                get_rng_state_tracker=get_rng_state_tracker,
                                fp8_meta_index=tex.FP8FwdTensors.GEMM2_WEIGHT)
        self.fp8_weight_shapes.append(self.fc2_weight.shape)

        if self.use_bias:
            fc2_bias = Parameter(
                torch.empty(hidden_size, device=device, dtype=params_dtype)
            )
            self.register_parameter('fc2_bias', fc2_bias,
                                    init_fn=init_method_constant(0.0))
        else:
            self.fc2_bias = torch.Tensor().to(dtype=params_dtype, device=device)

        if self.primary_weights_in_fp8:
            self.init_fp8_metadata(num_gemms=2)
            self.fp8_meta["update_amax_and_scale_fwd"] = True

        self.reset_parameters(defer_init=(device == 'meta'))

        # For RPL, bias has to be added after TP collectives
        # So it cannot be fused with the GEMM
        if self.set_parallel_mode and self.apply_bias:
            self.gemm_bias_unfused_add = True
        else:
            self.gemm_bias_unfused_add = False

        if self.bias_gelu_nvfusion:
            set_jit_fusion_options()
            if seq_length and micro_batch_size:
                warmup_jit_bias_gelu_all_dtypes(
                    self.size_per_partition, seq_length, micro_batch_size
                )

        # These many SMs are subtracted from the total SM count when calling forward
        # and backward LayerNorm C APIs. These envvars can be used to prevent the LN
        # kernels from using all SMs in the device. This is useful for cases such as
        # communication overlap with LN.
        self.fwd_ln_sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))
        self.bwd_ln_sm_margin = int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", "0"))

        self.register_buffer('had',
                             torch.FloatTensor(hadamard(HBS) / (HBS ** 0.5)))
        
    def reset_layer_norm_parameters(self) -> None:
        """Init LN params"""
        warnings.warn(
            ("This method will be deprecated in an upcoming release. "
             "Update your code to use LayerNormMLP.reset_parameters() instead."),
            DeprecationWarning,
            stacklevel=2
        )
        if not self.zero_centered_gamma:
            init.ones_(self.layer_norm_weight)
        else:
            init.zeros_(self.layer_norm_weight)
        if self.layer_norm_bias is not None:
            init.zeros_(self.layer_norm_bias)

    def reset_parameters(self, defer_init=False):
        super().reset_parameters(defer_init=defer_init)

        if not defer_init:
            # Set parallel attributes for layer norm parameters
            setattr(self.layer_norm_weight, "sequence_parallel", self.sequence_parallel)
            if self.normalization != "RMSNorm":
                setattr(self.layer_norm_bias, "sequence_parallel", self.sequence_parallel)

            # Set parallel attributes for linear parameters
            set_tensor_model_parallel_attributes(self.fc1_weight, True, 0, 1)
            set_tensor_model_parallel_attributes(self.fc2_weight, True, 1, 1)
            if self.use_bias:
                set_tensor_model_parallel_attributes(self.fc1_bias, True, 0, 1)
                if self.set_parallel_mode:
                    setattr(self.fc2_bias, "sequence_parallel", self.sequence_parallel)

    def get_fp8_weights_scratchpad(
        self,
        is_first_microbatch: Union[bool, None],
    ) -> List[torch.Tensor]:
        """
        Fetch the fp8 weight tensor placeholders if they exist (when
        `is_first_microbatch` is not `None`) or return empty fp8 weight
        tensors (if `is_first_microbatch is None`)
        """
        if not self.fp8 or self.primary_weights_in_fp8:
            return [None, None, None, None]

        if is_first_microbatch is None:
            # Return empty weight placeholders for each fwd/bwd pass
            fp8_weight_tensors = self.get_fp8_weights_empty_tensors(
                is_first_microbatch
            )
        else:
            # These persistent weight placeholders should've been created in
            # `set_fp8_weights` method
            fp8_weight_tensors = [self.weight1_fp8, self.weight1_t_fp8,
                                  self.weight2_fp8, self.weight2_t_fp8]

        return fp8_weight_tensors

    @no_torch_dynamo()
    def forward(
        self, inp: torch.Tensor, is_first_microbatch: Optional[bool] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply layer normalization to the input followed by a feedforward network (MLP Block).

        Parameters
        ----------
        inp : torch.Tensor
             Input tensor.
        is_first_microbatch : {True, False, None}, default = None
                             During training using either gradient accumulation or
                             pipeline parallelism a minibatch of data is further split
                             into microbatches. Between the microbatches of the same minibatch
                             the model weights are not updated. Setting this parameter indicates
                             whether the current microbatch is the first in a minibatch or not.
                             When set, this parameter enables additional optimizations:

                             * during FP8 training, it allows caching of the FP8 versions of
                               the weights
                             * it also allows skipping gradient accumulation during the
                               first microbatch (since it is the first gradient being
                               produced)
        """

        with self.prepare_forward(inp, is_first_microbatch, num_gemms=2) as inp:
            assert self.fp8 or not self.primary_weights_in_fp8, \
                   "Need to run inside fp8_autocast region when weights are stored in FP8."
            # Fetch the fp8 weights placeholders (for linear/gemm)
            weight1_fp8, weight1_t_fp8, weight2_fp8, weight2_t_fp8 = \
                self.get_fp8_weights_scratchpad(
                        is_first_microbatch
                )

            # Disable bias_gelu_nvfusion for determinism checkpointing in non-reentrant mode
            if (self.bias_gelu_nvfusion
                and not use_reentrant_activation_recompute()):
                self.bias_gelu_nvfusion = False

            from ..cpu_offload import CPUOffloadEnabled

            if torch.is_grad_enabled():
                fwd_fn = _LayerNormMLP.apply
                args = []
            else:
                fwd_fn = _LayerNormMLP.forward
                args = [None]
            args += (
                inp,
                self.layer_norm_weight,
                self.layer_norm_bias,
                self.fc1_weight,
                weight1_fp8,
                weight1_t_fp8,
                self.fc1_bias,
                self.use_bias,
                self.fc2_weight,
                weight2_fp8,
                weight2_t_fp8,
                self.fc2_bias,
                self.apply_bias and not self.gemm_bias_unfused_add,
                self.eps,
                is_first_microbatch,
                self.fp8,
                self.fp8_calibration,
                self.fp8_meta,
                self.fuse_wgrad_accumulation,
                CPUOffloadEnabled,
                self.tp_group,
                self.tp_size,
                self.sequence_parallel,
                self.tp_size > 1,
                self.activation_dtype,
                self.return_layernorm_output,
                self.return_layernorm_output_gathered,
                self.bias_gelu_nvfusion,
                self.set_parallel_mode,
                torch.is_grad_enabled(),
                self.fwd_ln_sm_margin,
                self.bwd_ln_sm_margin,
                self.zero_centered_gamma,
                self.activation,
                self.normalization,
                self.primary_weights_in_fp8,
                self.ub_bulk_wgrad,
                self.ub_bulk_dgrad,
                self.ub_split_rs,
                self.ub_atomic_gemm_rs,
                self.ub_split_ag,
                self.ub_atomic_gemm_ag,
                self.gemm_gelu_fusion,
                self.had
            )
            out = fwd_fn(*args)

        if self.return_layernorm_output:
            out, ln_out = out

        if self.gemm_bias_unfused_add:
            out = out + cast_if_needed(self.fc2_bias, self.activation_dtype)

        if self.return_bias:
            if self.return_layernorm_output:
                return out, cast_if_needed(self.fc2_bias, self.activation_dtype), ln_out
            return out, cast_if_needed(self.fc2_bias, self.activation_dtype)
        if self.return_layernorm_output:
            return out, ln_out
        return out
