import numpy as np
from typing import Tuple

class VirtualTransformerBlock:
    """Pure-NumPy Multi-Head Attention block, normalized for memory rules processing."""
    def __init__(self, d_model: int = 768, n_heads: int = 4, d_ff: int = 2048, dropout: float = 0.1, n_layers: int = 12):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale_residual = 1.0 / (2.0 * n_layers) ** 0.5

        self.W_q = self._init_weights(d_model, d_model, "q")
        self.W_k = self._init_weights(d_model, d_model, "k")
        self.W_v = self._init_weights(d_model, d_model, "v")
        self.W_o = self._init_weights(d_model, d_model, "o")
        self.W1 = self._init_weights(d_model, d_ff, "ff1")
        self.W2 = self._init_weights(d_ff, d_model, "ff2")
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)

    def _init_weights(self, rows: int, cols: int, name: str) -> np.ndarray:
        seed = sum(ord(c) for c in name) + rows + cols
        rng = np.random.default_rng(seed)
        return rng.standard_normal((rows, cols)) * np.sqrt(2. / rows)

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + eps)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        L, D = x.shape
        q = np.dot(x, self.W_q).reshape(L, self.n_heads, self.d_k).transpose(1, 0, 2)
        k = np.dot(x, self.W_k).reshape(L, self.n_heads, self.d_k).transpose(1, 0, 2)
        v = np.dot(x, self.W_v).reshape(L, self.n_heads, self.d_k).transpose(1, 0, 2)
        
        scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(self.d_k)
        attn_weights = self._softmax(scores)

        context = np.matmul(attn_weights, v).transpose(1, 0, 2).reshape(L, D)
        mha_out = self._layer_norm(x + self.scale_residual * np.dot(context, self.W_o))

        ffn_h = np.maximum(0, np.dot(mha_out, self.W1) + self.b1)
        block_out = self._layer_norm(mha_out + self.scale_residual * (np.dot(ffn_h, self.W2) + self.b2))

        return block_out, attn_weights.mean(axis=0)


class DualVirtualTransformerBlock:
    """
    Dual-stream cross-attention core. 
    Stream A: Current episodic memory sequence.
    Stream B: Semantic Rules / Identity Floors.
    """
    def __init__(self, d_model: int = 768, n_heads: int = 4, d_ff: int = 2048, n_layers: int = 12):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale_residual = 1.0 / (2.0 * n_layers) ** 0.5

        self.episodic_self_attn = VirtualTransformerBlock(d_model, n_heads, d_ff, n_layers=n_layers)
        self.semantic_self_attn = VirtualTransformerBlock(d_model, n_heads, d_ff, n_layers=n_layers)

        self.W_xq = self.episodic_self_attn._init_weights(d_model, d_model, "xq")
        self.W_sk = self.episodic_self_attn._init_weights(d_model, d_model, "sk")
        self.W_sv = self.episodic_self_attn._init_weights(d_model, d_model, "sv")
        self.W_xo = self.episodic_self_attn._init_weights(d_model, d_model, "xo")

    def forward(self, x_episodic: np.ndarray, s_semantic: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        L_e, D = x_episodic.shape
        L_s, _ = s_semantic.shape

        x_refined, _ = self.episodic_self_attn.forward(x_episodic)
        s_refined, _ = self.semantic_self_attn.forward(s_semantic)

        q = np.dot(x_refined, self.W_xq).reshape(L_e, self.n_heads, self.d_k).transpose(1, 0, 2)
        k = np.dot(s_refined, self.W_sk).reshape(L_s, self.n_heads, self.d_k).transpose(1, 0, 2)
        v = np.dot(s_refined, self.W_sv).reshape(L_s, self.n_heads, self.d_k).transpose(1, 0, 2)

        scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(self.d_k)
        cross_attn_weights = self.episodic_self_attn._softmax(scores)

        context = np.matmul(cross_attn_weights, v).transpose(1, 0, 2).reshape(L_e, D)
        conditioned_x = x_refined + self.scale_residual * np.dot(context, self.W_xo)

        return conditioned_x, cross_attn_weights.mean(axis=0), s_refined


class MoVEFilter:
    """
    Bridges the 256-dim Phase HRR vectors into the 768-dim MoVE transformer,
    cross-attends against Identity floors, and projects back down.
    """
    def __init__(self, dim_in: int = 256, d_model: int = 768):
        self.dim_in = dim_in
        self.d_model = d_model
        
        # Projection matrices
        self.W_proj_in  = np.random.default_rng(0).standard_normal((dim_in, d_model)) * np.sqrt(2/dim_in)
        self.W_proj_out = np.random.default_rng(1).standard_normal((d_model, dim_in)) * np.sqrt(2/d_model)
        
        self.dual_transformer = DualVirtualTransformerBlock(d_model=d_model)

    def filter(self, hvec_256: np.ndarray, identity_floors: np.ndarray) -> np.ndarray:
        """
        Conditions an incoming memory candidate vector against the crystallized identity floors.
        hvec_256: shape (256,) or (1, 256)
        identity_floors: shape (N, 256)
        """
        # Ensure correct shape
        x_episodic = hvec_256.reshape(1, self.dim_in)
        if len(identity_floors.shape) == 1:
            s_semantic = identity_floors.reshape(1, self.dim_in)
        else:
            s_semantic = identity_floors
            
        # 1. Project UP to 768
        x_episodic_proj = np.dot(x_episodic, self.W_proj_in)   # -> (1, 768)
        s_semantic_proj = np.dot(s_semantic, self.W_proj_in)   # -> (N, 768)
        
        # 2. Dual Cross-Attention
        conditioned, weights, _ = self.dual_transformer.forward(x_episodic_proj, s_semantic_proj)
        
        # 3. Project DOWN to 256
        filtered_hvec = np.dot(conditioned, self.W_proj_out)   # -> (1, 256)
        
        # Ensure Phase format [0, 2pi) after neural processing
        TWO_PI = 2.0 * np.pi
        filtered_hvec = np.mod(filtered_hvec, TWO_PI)
        
        return filtered_hvec.flatten()

# Global Singleton
move_guard = MoVEFilter()
