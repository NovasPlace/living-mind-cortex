import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# ── Physical packing helpers ──────────────────────────────────────────────────
# direction_q: INT4 → two nibbles packed per int8 byte  (4× vs FP32)
# qjl_signs:  {+1,-1} → bit-packed as uint8 booleans stored in int8 (32× vs FP32)

def _pack_int4(x_fp: Tensor) -> Tensor:
    """
    Pack a FP32 int4-range tensor (values in [-7,7] after rounding) into int8.
    Two adjacent values are packed per byte: low nibble = even index, high nibble = odd.
    Input shape: [... , D]  — D must be even.
    Output shape: [... , D // 2]  dtype=torch.int8
    """
    x = x_fp.to(torch.int8)          # Each element is already in [-7,7]
    x_pos = (x & 0x0F)               # Keep only the low 4 bits (0-15 unsigned view)
    low  = x_pos[..., 0::2]          # Even indices → low nibble
    high = x_pos[..., 1::2]          # Odd  indices → high nibble
    return (low | (high << 4)).to(torch.int8)


def _unpack_int4(packed: Tensor, dim: int) -> Tensor:
    """
    Unpack int8-packed nibbles back to FP32 in range [-7, 7].
    Output shape: [..., dim]
    """
    # Extract unsigned nibbles from each packed byte.
    # Both must be promoted to int32 BEFORE torch.where runs the comparison.
    # If low stays int8, then (low - 16) executes as int8 arithmetic for
    # values 8-15, where the true result (-8 to -1) fits int8, but PyTorch's
    # type promotion for scalar operands can silently upcast inconsistently
    # depending on backend. Explicit int32 promotion makes the contract exact.
    low  = (packed & 0x0F).to(torch.int32)           # unsigned nibble [0,15] as int32
    high = ((packed >> 4) & 0x0F).to(torch.int32)    # unsigned nibble [0,15] as int32
    # Two's complement sign-extend: values > 7 are negative in 4-bit representation
    low  = torch.where(low  > 7, low  - 16, low).to(torch.int8)   # signed [-8, 7]
    high = torch.where(high > 7, high - 16, high).to(torch.int8)  # signed [-8, 7]
    # Interleave back to original order
    out = torch.empty(*packed.shape[:-1], dim, dtype=torch.int8, device=packed.device)
    out[..., 0::2] = low
    out[..., 1::2] = high
    return out.to(torch.float32)


def _pack_signs(signs_fp: Tensor) -> Tensor:
    """
    Pack {+1, -1} sign tensor into int8 bytes, 8 signs per byte.
    +1 → bit 1, -1 → bit 0.
    Input shape: [..., D]  — D must be divisible by 8.
    Output shape: [..., D // 8]  dtype=torch.int8
    """
    bits = (signs_fp > 0).to(torch.uint8)  # +1 → 1, -1 → 0
    n = bits.shape[-1]
    padded_n = ((n + 7) // 8) * 8
    if padded_n > n:
        # Dead branch in practice: TurboQuantKV.__init__ enforces
        # jl_dim = (jl_dim + 7) // 8 * 8, guaranteeing D % 8 == 0.
        # Kept for standalone use outside of TurboQuantKV.
        pad = torch.zeros(*bits.shape[:-1], padded_n - n, dtype=torch.uint8, device=bits.device)
        bits = torch.cat([bits, pad], dim=-1)
    bits = bits.reshape(*bits.shape[:-1], -1, 8)
    weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=bits.device)
    packed = (bits * weights).sum(dim=-1).to(torch.int8)
    return packed


def _unpack_signs(packed: Tensor, dim: int) -> Tensor:
    """
    Unpack int8 bit-packed signs back to FP32 {+1.0, -1.0}.
    Output shape: [..., dim]
    """
    weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=packed.device)
    p = packed.to(torch.uint8).unsqueeze(-1)       # [..., D//8, 1]
    bits = ((p & weights) > 0).to(torch.float32)   # [..., D//8, 8]
    flat = bits.reshape(*bits.shape[:-2], -1)       # [..., D//8 * 8]
    return flat[..., :dim] * 2.0 - 1.0             # {0,1} → {-1,+1}

class TurboQuantKV(nn.Module):
    """
    TurboQuant: Training-Free High-Efficiency KV Cache Compression (Google Research).
    
    This module implements a twin-stage approach:
    1. PolarQuant: Random orthogonal preconditioning followed by polar coordinate 
       quantization, eliminating per-channel scale constraints.
    2. Quantized Johnson-Lindenstrauss (QJL): A 1-bit stochastic projection of the 
       residual error to form an unbiased attention dot-product estimator.
       
    Architecture:
    Input (D-dim) -> Rotate(R) -> Norm + Direction(INT4) [PolarQuant]
                  -> Residual Error -> Project(P) -> Stochastic 1-Bit [QJL]
    """
    
    def __init__(self, dim: int, jl_dim: Optional[int] = None, pq_bits: int = 4):
        """
        Args:
            dim (int): Embedding dimensionality of the key/value vectors.
            jl_dim (int): Dimensionality for the JL projection. Default is dim // 4.
            pq_bits (int): Bit-width for the PolarQuant directional quantization.
        """
        super().__init__()
        self.dim = dim
        self.jl_dim = jl_dim if jl_dim is not None else max(dim // 4, 64)
        # jl_dim must be divisible by 8 so sign bit-packing is lossless
        self.jl_dim = (self.jl_dim + 7) // 8 * 8
        # dim must be even for nibble-packing
        assert dim % 2 == 0, f"dim must be even for INT4 nibble packing, got {dim}"
        self.pq_bits = pq_bits

        # STAGE 1: Random Orthogonal Preconditioning Matrix (R)
        # Scatters sparse vector mass uniformly across all dimensions.
        R = torch.empty(self.dim, self.dim)
        nn.init.orthogonal_(R)
        self.register_buffer('R', R)

        # STAGE 2: Johnson-Lindenstrauss Projection Matrix (P)
        # Scaled to ensure exactly unbiased inner products: E[P P^T] = I
        P = torch.randn(self.dim, self.jl_dim) / math.sqrt(self.jl_dim)
        self.register_buffer('P', P)

        # Pre-compute pseudo-inverse of P — registered as a buffer so it lives on the
        # correct device and is never recomputed in the hot decompress() path.
        P_pinv = torch.linalg.pinv(P)   # shape: [jl_dim, dim]
        self.register_buffer('P_pinv', P_pinv)

    def compress(self, kv: Tensor) -> Dict[str, Tensor]:
        """
        Compresses a high-precision Key or Value cache tensor.
        
        Args:
            kv (Tensor): Shape [..., dim], standard FP32 or FP16 tensor.
            
        Returns:
            Dict: The structured mathematical compression state.
        """
        # ==========================================
        # STAGE 1: PolarQuant
        # ==========================================
        
        # 1.1 Rotate to dense space
        kv_rot = torch.matmul(kv, self.R)
        
        # 1.2 Polar Transformation 
        norm = torch.norm(kv_rot, p=2, dim=-1, keepdim=True)
        # Add epsilon to prevent division by zero for padded/empty tokens
        direction = kv_rot / (norm + 1e-8)
        
        # 1.3 Uniform Quantization of the Directional Tensor
        # High dimensional unit spheres have coordinates heavily concentrated around 0 (variance 1/D).
        # We standardize by multiplying by sqrt(D) so it behaves like N(0, 1), clip to [-3, 3], and quantize.
        std_direction = direction * math.sqrt(self.dim)
        
        bins = (1 << (self.pq_bits - 1)) - 1   # 7 for 4-bit
        clip_val = 3.0

        # Normalize to [-1, 1] relative to our clip bound
        normalized = torch.clamp(std_direction / clip_val, min=-1.0, max=1.0)

        # Quantize to integer values in [-bins, bins] = [-7, 7] for pq_bits=4
        quantized_int = torch.round(normalized * bins)  # FP32, values in [-7,7]

        # ── Physical INT4 packing: two nibbles per int8 byte ──────────────
        # Saves 4× VRAM vs FP32.  Unpacked to FP32 only in decompress/attention.
        direction_q_packed = _pack_int4(quantized_int)  # [..., dim // 2] int8

        # For the hot-path attention we keep the FP32 scale so attention math stays clean
        # Rescale factor: recovers the original direction_q = (q_int/bins * clip_val) / sqrt(dim)
        dq_scale = (clip_val / bins) / math.sqrt(self.dim)  # scalar float

        # Reconstruct FP32 direction_q for kv_hat (used only in compress to compute residual)
        direction_q_fp = quantized_int * dq_scale
        kv_hat = direction_q_fp * norm

        # ==========================================
        # STAGE 2: Quantized Johnson-Lindenstrauss
        # ==========================================

        # 2.1 Residual Error
        error = kv_rot - kv_hat

        # 2.2 Project error to lower dimension (JL Lemma)
        error_proj = torch.matmul(error, self.P)

        # 2.3 Stochastic 1-bit Quantization (Unbiased Estimator)
        #   P(sign = +1) = (x + 1) / 2    →  E[sign] = x  ✓
        error_scale = torch.max(torch.abs(error_proj), dim=-1, keepdim=True).values.clamp_min(1e-8)
        error_normalized = error_proj / error_scale
        rand_noise = torch.rand_like(error_normalized) * 2 - 1  # Uniform[-1, 1]
        qjl_signs_fp = torch.sign(error_normalized - rand_noise)  # {+1, -1} FP32

        # ── Physical INT8 sign packing: 8 signs per byte (32× vs FP32) ───
        qjl_signs_packed = _pack_signs(qjl_signs_fp)  # [..., jl_dim // 8] int8

        return {
            'norm':           norm,               # FP16/FP32  [..., 1]  — scalar per token
            'direction_q':    direction_q_packed, # INT8 nibble-packed  [..., dim // 2]
            'dq_scale':       torch.tensor(dq_scale, dtype=norm.dtype, device=norm.device),
            'error_scale':    error_scale,        # FP16/FP32  [..., 1]  — scalar per token
            'qjl_signs':      qjl_signs_packed,   # INT8 bit-packed  [..., jl_dim // 8]
        }

    def decompress(self, compressed_kv: Dict[str, Tensor]) -> Tensor:
        """
        Constructs the high-fidelity approximate tensor.
        Note: The true power of TurboQuant is computing attention directly in
        compressed space (see `attention_dot_product`), bypassing decompression.
        This path is for debugging / offline analysis only — not the inference hot path.
        """
        # ── Unpack direction_q nibbles → FP32 ────────────────────────────
        dq_int  = _unpack_int4(compressed_kv['direction_q'], self.dim)  # [..., dim]
        dq_fp   = dq_int * compressed_kv['dq_scale'].float()
        kv_hat  = dq_fp * compressed_kv['norm'].float()

        # ── Unpack qjl_signs bits → FP32 {+1,-1} ─────────────────────────
        signs_fp = _unpack_signs(compressed_kv['qjl_signs'], self.jl_dim)  # [..., jl_dim]
        error_proj_approx = signs_fp * compressed_kv['error_scale'].float()

        # Back-project using pre-computed P_pinv buffer — zero runtime cost ✓
        error_approx  = torch.matmul(error_proj_approx, self.P_pinv)

        # Combine and inverse-rotate
        kv_rot_approx = kv_hat + error_approx
        return torch.matmul(kv_rot_approx, self.R.T)

    def score_sequence(
        self,
        query: Tensor,
        compressed_keys: Dict[str, Tensor],
    ) -> Tensor:
        """
        Score one query position against every key in the compressed KV cache.
        Fully vectorized — NO per-token loop.

        Shape contract (multi-head attention layout):
          query:          [batch, heads, q_len,  dim]       FP32/BF16
          direction_q:    [batch, heads, kv_len, dim // 2]  INT8 packed
          norm:           [batch, heads, kv_len, 1]         FP32
          dq_scale:       scalar tensor                     FP32
          qjl_signs:      [batch, heads, kv_len, jl_dim//8] INT8 packed
          error_scale:    [batch, heads, kv_len, 1]         FP32

        Returns:
          scores: [batch, heads, q_len, kv_len]  — raw dot products, not softmaxed

        The unpack operations each run once over the full kv_len axis, then
        a single batched matmul scores all q × k pairs simultaneously.
        At kv_len = 8192, this is ~100x faster than a per-token loop.
        """
        # ── Rotate query into PolarQuant space ────────────────────────────
        # q_rot: [B, H, q_len, dim]
        q_rot = torch.matmul(query.float(), self.R)

        # ── Term 1: PolarQuant scores over full sequence ──────────────────
        # Unpack all kv_len keys at once: [B, H, kv_len, dim]
        dq_int  = _unpack_int4(compressed_keys['direction_q'], self.dim)
        dq_fp   = dq_int * compressed_keys['dq_scale'].float()
        k_hat   = dq_fp * compressed_keys['norm'].float()          # [B, H, kv_len, dim]
        # Batched matmul: [B, H, q_len, dim] × [B, H, dim, kv_len] → [B, H, q_len, kv_len]
        dot_pq  = torch.matmul(q_rot, k_hat.transpose(-1, -2))

        # ── Term 2: QJL residual correction over full sequence ────────────
        # Project query into JL space: [B, H, q_len, jl_dim]
        q_proj = torch.matmul(q_rot, self.P)
        # Unpack all kv_len error vectors at once: [B, H, kv_len, jl_dim]
        signs_fp          = _unpack_signs(compressed_keys['qjl_signs'], self.jl_dim)
        error_proj_approx = signs_fp * compressed_keys['error_scale'].float()
        # Batched matmul: [B, H, q_len, jl_dim] × [B, H, jl_dim, kv_len] → [B, H, q_len, kv_len]
        dot_qjl = torch.matmul(q_proj, error_proj_approx.transpose(-1, -2))

        # ── Unbiased fusion ───────────────────────────────────────────────
        return dot_pq + dot_qjl

    def attention_dot_product(self, query: Tensor, compressed_key: Dict[str, Tensor]) -> Tensor:
        """
        Single-query, single-key scoring. Convenience alias for score_sequence()
        when q_len == kv_len == 1. Use score_sequence() for full-sequence scoring.

        For scoring one query against a full KV cache, do NOT call this in a loop.
        Call score_sequence() instead — it batches the entire kv_len dimension
        into a single matmul, giving ~100x speedup at kv_len=8192.
        """
        if query.shape[-2] > 1:
            raise ValueError(
                f"attention_dot_product() called with q_len={query.shape[-2]}. "
                "This method is for single-token scoring only. "
                "Use score_sequence() for q_len > 1 — it runs a single batched matmul "
                "instead of requiring a per-token loop."
            )
        # Precondition the high-precision query
        q_rot = torch.matmul(query.float(), self.R)

        # Term 1: PolarQuant
        dq_int  = _unpack_int4(compressed_key['direction_q'], self.dim)
        dq_fp   = dq_int * compressed_key['dq_scale'].float()
        k_hat   = dq_fp * compressed_key['norm'].float()
        dot_pq  = torch.sum(q_rot * k_hat, dim=-1)

        # Term 2: QJL Residual
        q_proj            = torch.matmul(q_rot, self.P)
        signs_fp          = _unpack_signs(compressed_key['qjl_signs'], self.jl_dim)
        error_proj_approx = signs_fp * compressed_key['error_scale'].float()
        dot_qjl           = torch.sum(q_proj * error_proj_approx, dim=-1)

        return dot_pq + dot_qjl
