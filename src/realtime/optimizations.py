"""GVHMR model optimizations for real-time inference.

Applied as monkey-patches after model loading to avoid modifying third_party code.

Optimizations:
1. Flash Attention (F.scaled_dot_product_attention) — replaces manual einsum attention
2. torch.compile — compiles transformer blocks for kernel fusion
3. Skip postprocessing — disables IK and static joint refinement for real-time
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat


def _flash_attention_forward(self, x, attn_mask=None, key_padding_mask=None):
    """Drop-in replacement for RoPEAttention.forward using F.scaled_dot_product_attention.

    ~2-3x faster than manual einsum + softmax + dropout for sequence length 60-90.
    Requires float16/bfloat16 for Flash Attention backend (falls back to math otherwise).
    """
    B, L, _ = x.shape

    xq, xk, xv = self.query(x), self.key(x), self.value(x)
    xq = xq.reshape(B, L, self.num_heads, -1).transpose(1, 2)  # (B, H, L, D)
    xk = xk.reshape(B, L, self.num_heads, -1).transpose(1, 2)
    xv = xv.reshape(B, L, self.num_heads, -1).transpose(1, 2)

    # Apply RoPE
    xq = self.rope.rotate_queries_or_keys(xq)
    xk = self.rope.rotate_queries_or_keys(xk)

    # Build float attention bias from bool masks
    # GVHMR convention: True = mask out; SDPA convention: -inf = mask out
    attn_bias = None
    if attn_mask is not None or key_padding_mask is not None:
        attn_bias = torch.zeros(B, self.num_heads, L, L, dtype=xq.dtype, device=xq.device)
        if attn_mask is not None:
            # attn_mask: (L, L) bool, True = mask out
            attn_bias.masked_fill_(
                attn_mask.reshape(1, 1, L, L).expand(B, self.num_heads, -1, -1),
                float("-inf"),
            )
        if key_padding_mask is not None:
            # key_padding_mask: (B, L) bool, True = padding
            attn_bias.masked_fill_(
                key_padding_mask.reshape(B, 1, 1, L).expand(-1, self.num_heads, L, -1),
                float("-inf"),
            )

    output = F.scaled_dot_product_attention(
        xq, xk, xv,
        attn_mask=attn_bias,
        dropout_p=self.dropout.p if self.training else 0.0,
    )

    output = output.transpose(1, 2).reshape(B, L, -1)
    output = self.proj(output)
    return output


def patch_flash_attention(model):
    """Replace all RoPEAttention forward methods with Flash Attention version.

    Args:
        model: DemoPL model instance
    """
    from hmr4d.network.base_arch.transformer.encoder_rope import RoPEAttention

    patched = 0
    for module in model.modules():
        if isinstance(module, RoPEAttention):
            # Bind the method to this specific instance
            import types
            module.forward = types.MethodType(_flash_attention_forward, module)
            patched += 1

    print(f"[Optimize] Flash Attention patched: {patched} RoPEAttention layers")


def compile_model(model):
    """Apply torch.compile to the denoiser3d transformer.

    Uses reduce-overhead mode for best latency on small batch sizes.
    """
    try:
        model.pipeline.denoiser3d = torch.compile(
            model.pipeline.denoiser3d,
            mode="reduce-overhead",
            fullgraph=False,
        )
        print("[Optimize] torch.compile applied to denoiser3d (reduce-overhead)")
    except Exception as e:
        print(f"[Optimize] torch.compile failed (will use eager mode): {e}")


def predict_no_postproc(self, data, static_cam=False):
    """Replacement for DemoPL.predict() that skips expensive postprocessing.

    Skips:
    - pp_static_joint_cam / pp_static_joint (FK + sequential loop)
    - process_ik (CCD IK solver)

    Saves ~350-500ms per inference at the cost of slightly less refined output.
    """
    from hmr4d.utils.geo.hmr_cam import normalize_kp2d

    batch = {
        "length": data["length"][None],
        "obs": normalize_kp2d(data["kp2d"], data["bbx_xys"])[None],
        "bbx_xys": data["bbx_xys"][None],
        "K_fullimg": data["K_fullimg"][None],
        "cam_angvel": data["cam_angvel"][None],
        "f_imgseq": data["f_imgseq"][None],
    }
    batch = {k: v.cuda() for k, v in batch.items()}

    # postproc=False skips IK and static joint refinement
    outputs = self.pipeline.forward(batch, train=False, postproc=False, static_cam=static_cam)

    pred = {
        "smpl_params_global": {k: v[0] for k, v in outputs["pred_smpl_params_global"].items()},
        "smpl_params_incam": {k: v[0] for k, v in outputs["pred_smpl_params_incam"].items()},
        "K_fullimg": data["K_fullimg"],
    }
    return pred


def patch_skip_postproc(model):
    """Replace DemoPL.predict() with a version that skips postprocessing."""
    import types
    model.predict = types.MethodType(predict_no_postproc, model)
    print("[Optimize] Postprocessing disabled (skip IK + static joint refinement)")


def optimize_gvhmr_model(model, flash_attn=True, compile=False, skip_postproc=True):
    """Apply all optimizations to a loaded GVHMR DemoPL model.

    Args:
        model: DemoPL model instance (already on CUDA, eval mode)
        flash_attn: Replace manual attention with F.scaled_dot_product_attention
        compile: Apply torch.compile to transformer
        skip_postproc: Skip IK and static joint post-processing
    """
    if flash_attn:
        patch_flash_attention(model)

    if skip_postproc:
        patch_skip_postproc(model)

    # torch.compile should be applied AFTER other patches
    if compile:
        compile_model(model)

    return model
