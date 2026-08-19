# Import-weight measurements

Use the repository profiler to measure clean-process import behavior:

```bash
source /Users/rezah/envs/genpy/bin/activate
NUMBA_DISABLE_JIT=1 python tools/measure_imports.py --repeat 3
```

Each sample starts a fresh interpreter and uses only `src/`, so a previously
imported profile cannot warm a later one. The profiler reports median wall time,
loaded Pepsy module count, and any optional dependency roots imported by the
profile.

Reference run: 2026-08-18, local `genpy` environment, three samples per
profile:

```text
profile       median_ms  pepsy_modules  optional_modules
------------- ---------- -------------- ----------------
root               22.55              2 -
core               23.22             11 -
sampling           20.86              3 -
optimizers         21.35              3 -
experimental       22.60              3 -
```

The exact timings are machine-dependent. The durable boundary is that root,
core, sampling, optimizer, and experimental discovery imports do not load the
optional roots checked by `tools/measure_imports.py`. Domain implementations
are measured separately when their public symbols are resolved.
