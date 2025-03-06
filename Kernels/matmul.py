#!POPCORN leaderboard matmul



import triton
import triton.language as tl
import torch
from task import input_t, output_t

# Tunable tiling parameters
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program computes one tile of C of size (BLOCK_M x BLOCK_N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the row and column indices for the C tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    for k in range(0, K, BLOCK_K):
        # Compute pointers for the current A tile
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k + tl.arange(0, BLOCK_K))[None, :] * stride_ak
        # Load A tile with boundary check (if out-of-bound, load 0.0)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + tl.arange(0, BLOCK_K))[None, :] < K), other=0.0)

        # Compute pointers for the current B tile
        b_ptrs = b_ptr + (k + tl.arange(0, BLOCK_K))[:, None] * stride_bk + offs_n[None, :] * stride_bn
        # Load B tile with boundary check
        b = tl.load(b_ptrs, mask=((k + tl.arange(0, BLOCK_K))[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # Perform the dot-product of A tile and B tile, accumulating results in registers
        acc += tl.dot(a, b)

    # Write the computed C tile to global memory (with boundary checks)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def custom_kernel(data: input_t) -> output_t:
    # Unpack the tuple of input matrices A and B.
    A, B = data
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match for multiplication."

    # Allocate output tensor C.
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)

    # Define grid dimensions (one program per C tile)
    grid = lambda meta: (
        tl.cdiv(M, meta['BLOCK_M']),
        tl.cdiv(N, meta['BLOCK_N']),
    )

    # Launch the Triton kernel.
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return C


# Optional self-test
if __name__ == "__main__":
    # Create random matrices A (M x K) and B (K x N)
    M, K, N = 256, 256, 256
    A = torch.randn((M, K), device="cuda", dtype=torch.float32)
    B = torch.randn((K, N), device="cuda", dtype=torch.float32)
    C = custom_kernel((A, B))
    torch.cuda.synchronize()
    ref = A @ B
    assert torch.allclose(C, ref, atol=1e-3), "Mismatch detected!"
