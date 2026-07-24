import json
import os
import shutil
import stat
from pathlib import Path


path = Path("../pepsy_examples/fermi_hubbard/fh_peps.ipynb").resolve()
backup = Path("/tmp/fh_peps_before_vmc_api.ipynb")
tmp_path = path.with_suffix(path.suffix + ".tmp")

with path.open("r", encoding="utf-8") as f:
    notebook = json.load(f)

new_source = [
    "# VMC energy estimate on the post-imaginary-time PEPS.\n",
    "from pepsy.vmc import (\n",
    "    FermionSiteEncoding,\n",
    "    TorchPEPSBoundaryAmplitude,\n",
    "    TorchSquareLattice,\n",
    "    TorchVMCDriver,\n",
    ")\n",
    "\n",
    "vmc_encoding = FermionSiteEncoding.vmc_torch()\n",
    "vmc_device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "vmc_sites = tuple(native_peps.sites)\n",
    "vmc_seed_config = torch.tensor(\n",
    "    [[\n",
    "        vmc_encoding.up\n",
    "        if half_filled_occupations[site] == (1, 0)\n",
    "        else vmc_encoding.down\n",
    "        for site in vmc_sites\n",
    "    ]],\n",
    "    dtype=torch.long,\n",
    "    device=vmc_device,\n",
    ")\n",
    "n_vmc_walkers = 100\n",
    "vmc_configs = vmc_seed_config.repeat(n_vmc_walkers, 1)\n",
    "\n",
    "vmc_model = TorchPEPSBoundaryAmplitude(\n",
    "    native_peps,\n",
    "    contraction=\"boundary\",\n",
    "    chi=32,\n",
    "    cutoff=1.0e-10,\n",
    "    contraction_opts={\"compress_opts\": {\"method\": \"cholesky\", \"absorb\": \"both\"}},\n",
    "    dtype=torch.complex128,\n",
    "    device=vmc_device,\n",
    ")\n",
    "\n",
    "# The supplied native terms define the local energy; no t/U guessing.\n",
    "vmc_driver = TorchVMCDriver(\n",
    "    vmc_model,\n",
    "    TorchSquareLattice(Lx, Ly),\n",
    "    vmc_configs,\n",
    "    terms=ham.terms,\n",
    "    site_order=vmc_sites,\n",
    "    proposal=\"spinful\",\n",
    "    encoding=vmc_encoding,\n",
    "    chunk_size=16,\n",
    "    generator=torch.Generator(device=vmc_device).manual_seed(7),\n",
    ")\n",
    "\n",
    "# 100 measured samples, with burn-in and extra mixing before measurement.\n",
    "with torch.inference_mode():\n",
    "    vmc_estimate = vmc_driver.estimate_energy(\n",
    "        burn_in=10,\n",
    "        n_measurements=1,\n",
    "        sweeps_between=3,\n",
    "        progress=True,\n",
    "    )\n",
    "\n",
    "vmc_energy_total = float(torch.real(vmc_estimate.energy_mean).cpu())\n",
    "vmc_energy_density = vmc_energy_total / num_sites\n",
    "vmc_stderr_density = float(vmc_estimate.energy_stderr.cpu()) / num_sites\n",
    "print(\n",
    "    f\"VMC boundary-MPS: E={vmc_energy_total:+.8f}, \"\n",
    "    f\"E/N={vmc_energy_density:+.8f} +/- {vmc_stderr_density:.3e} \"\n",
    "    f\"({vmc_estimate.n_samples} samples, \"\n",
    "    f\"acceptance={vmc_estimate.acceptance_rate:.3f}, \"\n",
    "    f\"samples/s={vmc_estimate.samples_per_second:.2f})\"\n",
    ")\n",
    "print(\n",
    "    f\"deterministic boundary-MPS check: E/N=\"\n",
    "    f\"{native_energy_density(native_peps, max_bond=32):+.8f}\"\n",
    ")\n",
]

found = False
for cell in notebook["cells"]:
    if cell.get("id") == "vmc-energy-estimate-code":
        cell["source"] = new_source
        found = True
        break
if not found:
    raise SystemExit("VMC code cell not found; refusing to edit.")

if not any(
    cell.get("id") == "vmc-bp-importance" for cell in notebook["cells"]
):
    index = next(
        i for i, cell in enumerate(notebook["cells"])
        if cell.get("id") == "vmc-energy-estimate-code"
    ) + 1
    notebook["cells"][index:index] = [
        {
            "cell_type": "markdown",
            "id": "vmc-bp-importance",
            "metadata": {},
            "source": [
                "### Optional BP importance proposal\n",
                "\n",
                "Set `use_bp_importance=True` to draw proposals with BP and "
                "measure their amplitudes and local energies with torch.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "vmc-bp-importance-code",
            "metadata": {},
            "outputs": [],
            "source": [
                "from pepsy.sampling import PepsBpSampler\n",
                "\n",
                "use_bp_importance = False\n",
                "if use_bp_importance:\n",
                "    bp_importance = vmc_driver.importance_energy_estimate(\n",
                "        PepsBpSampler(native_peps),\n",
                "        n_samples=100,\n",
                "        sample_kwargs={\n",
                "            \"method\": \"mps\",\n",
                "            \"chi\": 32,\n",
                "            \"cutoff\": 1.0e-10,\n",
                "        },\n",
                "        progress=True,\n",
                "    )\n",
                "    print(\n",
                "        f\"BP importance: E/N={float(torch.real(bp_importance.energy_mean).cpu()) / num_sites:+.8f} \"\n",
                "        f\"ESS={float(bp_importance.effective_sample_size.cpu()):.1f}/\"\n",
                "        f\"{bp_importance.n_valid}\"\n",
                "    )\n",
            ],
        },
    ]

shutil.copy2(path, backup)
mode = stat.S_IMODE(path.stat().st_mode)
with tmp_path.open("w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
    f.write("\n")
os.chmod(tmp_path, mode)
os.replace(tmp_path, path)
print(f"updated {path}")
print(f"backup  {backup}")
