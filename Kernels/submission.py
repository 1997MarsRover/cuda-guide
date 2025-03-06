#!POPCORN leaderboard grayscale
#!POPCORN gpu A100


import triton
import triton.language as tl
import torch
from task import input_t, output_t

COEFF_R = tl.constexpr(0.2989)
COEFF_G = tl.constexpr(0.5870)
COEFF_B = tl.constexpr(0.1140)

@triton.jit
def rgb_to_gray_kernel(
    input_ptr, output_ptr, H, W, stride0, stride1, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < H * W
    x = offs % W
    y = offs // W

    base = y * stride0 + x * stride1  # (BLOCK_SIZE,)
    rgb_ptrs = input_ptr + base[:, None] + tl.arange(0, 3)[None, :]  # (BLOCK_SIZE, 3)
    rgbs = tl.load(rgb_ptrs, mask=mask[:, None], other=0.0)
    r, g, b = rgbs[:, 0], rgbs[:, 1], rgbs[:, 2]
    gray = COEFF_R * r + COEFF_G * g + COEFF_B * b
    output_offs = y * W + x
    tl.store(output_ptr + output_offs, gray, mask=mask)

def custom_kernel(data: input_t) -> output_t:
    data = data.contiguous()
    H, W, _ = data.shape
    out = torch.empty((H, W), device=data.device, dtype=data.dtype)
    BLOCK_SIZE = 128  # Optimized for coalescing
    grid = lambda meta: (triton.cdiv(H * W, BLOCK_SIZE),)
    rgb_to_gray_kernel[grid](data, out, H, W, data.stride(0), data.stride(1), BLOCK_SIZE=BLOCK_SIZE)
    return out

if __name__ == "__main__":
    for _ in range(3):
        inp = torch.rand(4096, 4096, 3, device="cuda")
        out = custom_kernel(inp)
        torch.cuda.synchronize()
        ref = inp[..., 0] * 0.2989 + inp[..., 1] * 0.5870 + inp[..., 2] * 0.1140
        assert torch.allclose(out, ref, atol=1e-5), "Mismatch found!"