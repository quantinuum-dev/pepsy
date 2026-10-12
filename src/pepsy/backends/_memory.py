"""Optional allocator queries without changing device or allocator policy."""

import autoray as ar


def available_device_memory(array):
    """Return usable allocator bytes, or None when no public query is available.

    Count free device memory and reusable cached blocks on the array's device.
    This is a snapshot, not a reservation against other processes.
    """
    blocks = getattr(array, "blocks", None)
    if blocks is not None:
        array = next(iter(blocks.values()), None)
    backend = ar.infer_backend(array)
    try:
        if backend == "torch" and array.device.type == "cuda":
            import torch

            device = array.device
            free, total = torch.cuda.mem_get_info(device)
            reserved = torch.cuda.memory_reserved(device)
            allocated = torch.cuda.memory_allocated(device)
            fraction = torch.cuda.get_per_process_memory_fraction(device)
            return max(0, min(free + reserved - allocated,
                              int(total * fraction) - allocated))
        if backend == "cupy":
            import cupy

            with array.device:
                free, _ = cupy.cuda.runtime.memGetInfo()
                pool = cupy.get_default_memory_pool()
                # A custom allocator might not be able to reuse this pool.
                if cupy.cuda.get_allocator() == pool.malloc:
                    available = free + pool.free_bytes()
                    limit = pool.get_limit()
                    return max(0, min(available, limit - pool.used_bytes())
                               if limit else available)
                return int(free)
        if backend == "jax":
            devices = array.devices()
            if len(devices) == 1:
                stats = next(iter(devices)).memory_stats()
                if stats and "bytes_limit" in stats and "bytes_in_use" in stats:
                    return max(0, int(stats["bytes_limit"] - stats["bytes_in_use"]))
    except (AttributeError, RuntimeError, ValueError):
        # Optional runtimes may lack a memory query even when execution works.
        return None
    return None
