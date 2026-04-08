import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

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
        
        bins = (1 << (self.pq_bits - 1)) - 1
        clip_val = 3.0
        
        # Normalize to [-1, 1] relative to our clip bound
        normalized = torch.clamp(std_direction / clip_val, min=-1.0, max=1.0)
        
        quantized_normalized = torch.round(normalized * bins) / bins
        
        # Re-scale back to the original Directional space
        direction_q = (quantized_normalized * clip_val) / math.sqrt(self.dim)
        
        # Reconstruct the PolarQuant estimation
        kv_hat = direction_q * norm
        
        # ==========================================
        # STAGE 2: Quantized Johnson-Lindenstrauss
        # ==========================================
        
        # 2.1 Calculate Residual Error
        error = kv_rot - kv_hat
        
        # 2.2 Project error to lower dimension (JL Lemma)
        error_proj = torch.matmul(error, self.P)
        
        # 2.3 Stochastic 1-bit Quantization (Unbiased Estimator)
        # We find the max scale of the projection so we can map it between [-1, 1]
        error_scale = torch.max(torch.abs(error_proj), dim=-1, keepdim=True).values.clamp_min(1e-8)
        error_normalized = error_proj / error_scale
        
        # Stochastic rounding technique:
        # P(val = +1) = (x + 1) / 2
        # P(val = -1) = (1 - x) / 2
        # E[sign] = x
        rand_noise = torch.rand_like(error_normalized) * 2 - 1  # Uniform[-1, 1]
        qjl_signs = torch.sign(error_normalized - rand_noise)
        
        return {
            'norm': norm,            # FP32/FP16 (scalar)
            'direction_q': direction_q, # INT4 simulated as FP32 (shape: [..., dim])
            'error_scale': error_scale, # FP32/FP16 (scalar)
            'qjl_signs': qjl_signs   # INT1 simulated as FP32 (shape: [..., jl_dim])
        }

    def decompress(self, compressed_kv: Dict[str, Tensor]) -> Tensor:
        """
        Constructs the high-fidelity approximate tensor. 
        Note: The true power of TurboQuant is computing attention directly in 
        compressed space (see `attention_dot_product`), bypassing decompression.
        """
        # Reconstruct PolarQuant core
        kv_hat = compressed_kv['direction_q'] * compressed_kv['norm']
        
        # Estimate the error projection using the stochastic bits
        error_proj_approx = compressed_kv['qjl_signs'] * compressed_kv['error_scale']
        
        # Pseudo-inverse back-projection of the JL space
        P_pinv = torch.linalg.pinv(self.P)
        error_approx = torch.matmul(error_proj_approx, P_pinv)
        
        # Combine and inverse-rotate
        kv_rot_approx = kv_hat + error_approx
        kv_approx = torch.matmul(kv_rot_approx, self.R.T)
        
        return kv_approx

    def attention_dot_product(self, query: Tensor, compressed_key: Dict[str, Tensor]) -> Tensor:
        """
        Computes the unbiased Attention Dot Product matching Google's algorithm.
        
        score(Q, K) = Q^T(K_pq + K_err) = (Q^T K_pq) + (Q_proj^T K_proj_err)
        
        Args:
            query (Tensor): Uncompressed query tensor [..., dim]
            compressed_key (Dict): TurboQuant compressed key state
            
        Returns:
            Tensor: Attention logits [..., 1]
        """
        # Precondition the high-precision query
        q_rot = torch.matmul(query, self.R)
        
        # Term 1: High-Speed PolarQuant Dot Product
        k_hat = compressed_key['direction_q'] * compressed_key['norm']
        dot_pq = torch.sum(q_rot * k_hat, dim=-1)
        
        # Term 2: QJL Residual Correction Dot Product
        q_proj = torch.matmul(q_rot, self.P)
        error_proj_approx = compressed_key['qjl_signs'] * compressed_key['error_scale']
        dot_qjl = torch.sum(q_proj * error_proj_approx, dim=-1)
        
        # Unbiased Fusion
        return dot_pq + dot_qjl
