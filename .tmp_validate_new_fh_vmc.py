import nbformat
from nbclient import NotebookClient


path = "../pepsy_examples/fermi_hubbard/fh_peps.ipynb"
nb = nbformat.read(path, as_version=4)
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    cell.source = cell.source.replace(
        "tau = [0.2, 0.1, 0.03, 0.01]",
        "tau = [0.01]",
    )
    cell.source = cell.source.replace(
        "steps = [50, 50, 50, 50]",
        "steps = [1]",
    )
    if cell.get("id") == "vmc-energy-estimate-code":
        cell.source = cell.source.replace("burn_in=10", "burn_in=1")
        cell.source = cell.source.replace("sweeps_between=3", "sweeps_between=1")

limit = next(
    i for i, cell in enumerate(nb.cells)
    if cell.get("id") == "vmc-bp-importance"
)
nb.cells = nb.cells[:limit]
NotebookClient(nb, timeout=600, kernel_name="python3").execute()
for cell in nb.cells:
    for output in cell.get("outputs", ()):
        if output.get("output_type") == "stream":
            text = "".join(output.get("text", []))
            if "VMC boundary-MPS" in text or "deterministic boundary-MPS" in text:
                print(text, end="")
