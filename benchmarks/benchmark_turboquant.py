import torch
from cortex.turboquant import TurboQuantKV

def run_benchmark():
    print("=" * 50)
    print("TurboQuant Memory Compression Benchmark")
    print("=" * 50)
    
    # Simulate a realistic Key-Value cache layer for standard Llama-3 70B attention
    num_heads = 32
    seq_len = 4096
    head_dim = 128
    
    # We flatten out the heads to pretend it's a massive continuous cache block
    dim = num_heads * head_dim # 4096
    
    print(f"Token Sequence Length: {seq_len}")
    print(f"Embedding Dimension: {dim} (e.g. 32 heads * 128 dim)")
    
    # 1. Standard FP16 Cache Formulation
    fp16_bits_per_param = 16
    uncompressed_bits = seq_len * dim * fp16_bits_per_param
    uncompressed_mb = uncompressed_bits / (8 * 1024 * 1024)
    print(f"\\n[Standard FP16 KV Cache]")
    print(f"Size: {uncompressed_mb:.2f} MB")
    
    # 2. TurboQuant Memory Footprint
    jl_dim = 1024 # 1-bit projection
    pq_bits = 4   # 4-bit INT PolarQuant
    
    # The footprint calculation purely by data types:
    # 1. 'norm' (FP16): 1 scalar per vector
    norm_bits = seq_len * 1 * 16
    # 2. 'direction_q' (INT4): D scalars per vector
    dir_bits = seq_len * dim * pq_bits
    # 3. 'error_scale' (FP16): 1 scalar per vector
    err_scale_bits = seq_len * 1 * 16
    # 4. 'qjl_signs' (INT1): jl_dim scalars per vector
    qjl_bits = seq_len * jl_dim * 1
    
    tq_total_bits = norm_bits + dir_bits + err_scale_bits + qjl_bits
    tq_total_mb = tq_total_bits / (8 * 1024 * 1024)
    
    effective_bits_per_param = tq_total_bits / (seq_len * dim)
    
    print(f"\\n[TurboQuant Compressed Cache]")
    print(f"PQ Direction (INT{pq_bits}): {dir_bits / (8*1024*1024):.2f} MB")
    print(f"QJL Signs (1-Bit, {jl_dim} dim): {qjl_bits / (8*1024*1024):.2f} MB")
    print(f"Metadata (Norm + error_scale): {(norm_bits + err_scale_bits) / (8*1024*1024):.4f} MB")
    print(f"----------------------------------------")
    print(f"Total Size: {tq_total_mb:.2f} MB")
    print(f"Effective bit-width per parameter: {effective_bits_per_param:.2f} bits")
    
    savings = (1 - (tq_total_mb / uncompressed_mb)) * 100
    print(f"\\n=> Final Memory Savings: {savings:.2f}% (approx {uncompressed_mb/tq_total_mb:.1f}x reduction!)")

if __name__ == "__main__":
    run_benchmark()
