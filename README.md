# PePsY

<p align="center">
  <img src="assets/pepsy-icon.svg" alt="PePsY icon" width="180" />
</p>

Simulate quantum circuits with MPS, PEPS, tree, and stabilizer tensor networks.
Includes optimization, sampling, noise, and measurements.

**0.6.0** · **Python 3.12+** · [Changelog](CHANGELOG.md)

## Install

From a checkout:

```bash
python -m pip install .
```

[Setup and optional dependencies](docs/installation.md)

## Get started

Run the [MPS simulation notebook](examples/MpsMagnetization/mps_simulator.ipynb),
or read the [MPS](docs/api/optimizers/mps.md),
[PEPS](docs/api/optimizers/peps.md), [tree](docs/api/optimizers/tree_replay.md),
or [stabilizer](docs/api/optimizers/stabilizer_tn.md) guide.

Reuse and batch independent PEPO gate schedules with
[prepared simple update](docs/api/operators/gates.md#prepared-dense-simple-update).

[Documentation](docs/index.md) · [Examples](docs/examples.md) ·
[Contributing](CONTRIBUTING.md)
