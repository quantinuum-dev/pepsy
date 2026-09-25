"""Optional bounded-memory kernels for structured dense exact gates."""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _cpu_kernels():
    try:
        from numba import njit
    except ImportError:
        return None

    @njit(nogil=True)
    def parity(inp, out, low, high, mask0, mask1, coeff, start, stop):
        low_mask = (1 << low) - 1
        high_mask = (1 << high) - 1
        for i in range(start, stop):
            base = (i & low_mask) | ((i & ~low_mask) << 1)
            base = (base & high_mask) | ((base & ~high_mask) << 1)
            i00 = base
            i01 = base | mask1
            i10 = base | mask0
            i11 = base | mask0 | mask1
            v00, v01, v10, v11 = inp[i00], inp[i01], inp[i10], inp[i11]
            out[i00] = coeff[0] * v00 + coeff[1] * v11
            out[i01] = coeff[2] * v01 + coeff[3] * v10
            out[i10] = coeff[4] * v01 + coeff[5] * v10
            out[i11] = coeff[6] * v00 + coeff[7] * v11

    @njit(inline="always")
    def popcount(value):
        value = value - ((value >> np.uint64(1)) & np.uint64(0x5555555555555555))
        value = (value & np.uint64(0x3333333333333333)) + (
            (value >> np.uint64(2)) & np.uint64(0x3333333333333333)
        )
        value = (value + (value >> np.uint64(4))) & np.uint64(0x0f0f0f0f0f0f0f0f)
        return (value * np.uint64(0x0101010101010101)) >> np.uint64(56)

    @njit(nogil=True)
    def grouped_phase(inp, out, offsets, masks, table, start, stop):
        for i in range(start, stop):
            basis = np.uint64(i)
            disagreements = np.uint64(0)
            for group in range(offsets.size):
                bits = (basis ^ (basis >> offsets[group])) & masks[group]
                disagreements += popcount(bits)
            out[i] = inp[i] * table[disagreements]

    return parity, grouped_phase


def _run_cpu(kernel, total, *args):
    """Use four independent chunks without changing Numba's global thread count."""
    workers = min(4, max(1, total // (1 << 16)))
    if workers == 1:
        kernel(*args, 0, total)
        return
    # Compile this dtype on the caller thread before concurrent launches.
    kernel(*args, 0, 0)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [
            pool.submit(kernel, *args, total * worker // workers,
                        total * (worker + 1) // workers)
            for worker in range(workers)
        ]
        for job in jobs:
            job.result()


_CUDA = r"""
typedef {scalar} Real;
typedef {vector} Value;

__device__ __forceinline__ Value multiply(Value a, Value b) {{
    return make_{vector}(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}}

__device__ __forceinline__ Value add(Value a, Value b) {{
    return make_{vector}(a.x + b.x, a.y + b.y);
}}

extern "C" __global__ void grouped_phase(
    const Value* inp, Value* out, const int* offsets,
    const unsigned long long* masks, const Value* table,
    unsigned long long size, int groups
) {{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= size) return;
    int disagreements = 0;
    for (int group = 0; group < groups; ++group) {{
        disagreements += __popcll((i ^ (i >> offsets[group])) & masks[group]);
    }}
    out[i] = multiply(inp[i], table[disagreements]);
}}

extern "C" __global__ void parity(
    const Value* inp, Value* out, int low, int high,
    unsigned long long mask0, unsigned long long mask1,
    const Value* coeff, unsigned long long groups
) {{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= groups) return;
    unsigned long long low_mask = (1ULL << low) - 1ULL;
    unsigned long long high_mask = (1ULL << high) - 1ULL;
    unsigned long long base = (i & low_mask) | ((i & ~low_mask) << 1);
    base = (base & high_mask) | ((base & ~high_mask) << 1);
    unsigned long long i00 = base;
    unsigned long long i01 = base | mask1;
    unsigned long long i10 = base | mask0;
    unsigned long long i11 = base | mask0 | mask1;
    Value v00 = inp[i00], v01 = inp[i01], v10 = inp[i10], v11 = inp[i11];
    out[i00] = add(multiply(coeff[0], v00), multiply(coeff[1], v11));
    out[i01] = add(multiply(coeff[2], v01), multiply(coeff[3], v10));
    out[i10] = add(multiply(coeff[4], v01), multiply(coeff[5], v10));
    out[i11] = add(multiply(coeff[6], v00), multiply(coeff[7], v11));
}}
"""


@lru_cache(maxsize=4)
def _cuda_kernel(kind, dtype):
    import cupy as cp

    scalar, vector = ("float", "float2") if dtype == "complex64" else ("double", "double2")
    return cp.RawKernel(_CUDA.format(scalar=scalar, vector=vector), kind, options=("--std=c++11",))


def _supported(data, coeff_dtype):
    return (
        data.dtype in (np.dtype("complex64"), np.dtype("complex128"))
        and data.dtype == coeff_dtype
        and data.flags.c_contiguous
    )


def apply_parity(data, mask0, mask1, coeff):
    """Apply two parity-sector 2x2 matrices, or return None."""
    if not _supported(data, coeff.dtype):
        return None
    low, high = sorted((mask0.bit_length() - 1, mask1.bit_length() - 1))
    if isinstance(data, np.ndarray):
        kernels = _cpu_kernels()
        if kernels is None:
            return None
        out = np.empty_like(data)
        _run_cpu(kernels[0], data.size // 4, data.reshape(-1), out.reshape(-1), low, high, mask0, mask1, coeff)
        return out

    import cupy as cp

    out = cp.empty_like(data)
    device_coeff = cp.asarray(coeff)
    groups = int(data.size // 4)
    _cuda_kernel("parity", data.dtype.name)(
        ((groups + 255) // 256,), (256,),
        (data, out, low, high, mask0, mask1, device_coeff, groups),
    )
    return out


def apply_grouped_phase(data, offsets, masks, table):
    """Apply an equal-value ZZ layer using grouped XOR population counts."""
    if not _supported(data, table.dtype):
        return None
    if isinstance(data, np.ndarray):
        kernels = _cpu_kernels()
        if kernels is None:
            return None
        out = np.empty_like(data)
        _run_cpu(kernels[1], data.size, data.reshape(-1), out.reshape(-1), offsets, masks, table)
        return out

    import cupy as cp

    out = cp.empty_like(data)
    device_offsets = cp.asarray(offsets)
    device_masks = cp.asarray(masks)
    device_table = cp.asarray(table)
    size = int(data.size)
    _cuda_kernel("grouped_phase", data.dtype.name)(
        ((size + 255) // 256,), (256,),
        (data, out, device_offsets, device_masks, device_table, size, len(offsets)),
    )
    return out
