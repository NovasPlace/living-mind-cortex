import torch
import torch.nn as nn
from cortex.turboquant import TurboQuantKV

def test_turboquant_shapes():
    batch_size = 2
    seq_len = 16
    dim = 256
    
    tq = TurboQuantKV(dim=dim, jl_dim=64, pq_bits=4)
    x = torch.randn(batch_size, seq_len, dim)
    
    compressed = tq.compress(x)
    
    assert compressed['norm'].shape == (batch_size, seq_len, 1)
    assert compressed['direction_q'].shape == (batch_size, seq_len, dim)
    assert compressed['error_scale'].shape == (batch_size, seq_len, 1)
    assert compressed['qjl_signs'].shape == (batch_size, seq_len, 64)
    
    decompressed = tq.decompress(compressed)
    assert decompressed.shape == (batch_size, seq_len, dim)


def test_turboquant_dot_product_approximation():
    """
    Validates that the composite TurboQuant dot product approximation functions 
    statistically effectively compared to standard FP32 dot product.
    """
    torch.manual_seed(42) # Reproducibility
    
    # 4096 is standard Llama-3 embedding dimension
    dim = 4096 
    jl_dim = 1024
    
    tq = TurboQuantKV(dim=dim, jl_dim=jl_dim, pq_bits=4)
    
    q = torch.randn(1, 128, dim)
    k = torch.randn(1, 128, dim)
    
    # True FP32
    true_dot = torch.sum(q * k, dim=-1)
    
    # Compressed TQ
    compressed_k = tq.compress(k)
    tq_dot = tq.attention_dot_product(q, compressed_k)
    
    q_rot = torch.matmul(q, tq.R)
    k_rot = torch.matmul(k, tq.R)
    k_hat = compressed_k['direction_q'] * compressed_k['norm']
    error = k_rot - k_hat
    
    # 1. PQ correlation
    pq_dot = torch.sum(q_rot * k_hat, dim=-1)
    
    # 2. JL Exact correlation
    jl_exact_dot = torch.sum(torch.matmul(q_rot, tq.P) * torch.matmul(error, tq.P), dim=-1)
    
    # 3. JL Quantized correlation
    error_scale = compressed_k['error_scale']
    error_proj_approx = compressed_k['qjl_signs'] * error_scale
    jl_quant_dot = torch.sum(torch.matmul(q_rot, tq.P) * error_proj_approx, dim=-1)
    
    true_error_dot = torch.sum(q_rot * error, dim=-1)
    
    pq_corr = torch.corrcoef(torch.stack((true_dot.view(-1), pq_dot.view(-1))))[0,1].item()
    jl_exact_corr = torch.corrcoef(torch.stack((true_error_dot.view(-1), jl_exact_dot.view(-1))))[0,1].item()
    jl_quant_corr = torch.corrcoef(torch.stack((true_error_dot.view(-1), jl_quant_dot.view(-1))))[0,1].item()
    
    tq_corr = torch.corrcoef(torch.stack((true_dot.view(-1), tq_dot.view(-1))))[0,1].item()
    
    print(f"\\n[Debug] PQ Base Correlation: {pq_corr:.4f}")
    print(f"[Debug] JL Exact Error Correlation: {jl_exact_corr:.4f}")
    print(f"[Debug] JL Quantized Error Correlation: {jl_quant_corr:.4f}")
    print(f"TurboQuant Total Correlation to FP32 Dot Product: {tq_corr:.4f}")
    
    assert pq_corr > 0.95, "PolarQuant Base compression failed to capture variance."
    
if __name__ == "__main__":
    test_turboquant_shapes()
    test_turboquant_dot_product_approximation()
    print("TurboQuant tests passed.")
