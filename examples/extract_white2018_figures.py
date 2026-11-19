"""Extract numerical line coordinates from White et al.'s arXiv figure PDFs.

Download https://arxiv.org/src/1707.01506 and supply it as --source. The
optional pypdf dependency is needed only for this extraction, not evolution.
These are rounded vector-plot coordinates, not the authors' raw simulations.
"""

import argparse
from io import BytesIO
import json
from pathlib import Path
import tarfile

import numpy as np


FIGURES = {
    "fig7": "figures/TEBD87_paper-plots_v042-MPS-energy-L64-nearY-E.pdf",
    "fig8": "figures/TEBD87_PSE_v042-absolute-eps-L24-nearY-epsfc.pdf",
}


def line_paths(pdf):
    from pypdf import PdfReader
    from pypdf.generic import ContentStream
    reader = PdfReader(BytesIO(pdf))
    state = {"color": None, "dash": [], "clip": None}
    stack, paths, points, result = [], [], [], []
    rectangle = None
    for values, op in ContentStream(reader.pages[0].get_contents(), reader).operations:
        if op == b"q":
            stack.append(dict(state))
        elif op == b"Q":
            state = stack.pop()
        elif op == b"RG":
            state["color"] = tuple(map(float, values))
        elif op == b"G":
            state["color"] = (float(values[0]),) * 3
        elif op == b"d":
            state["dash"] = list(map(float, values[0]))
        elif op == b"re":
            rectangle = list(map(float, values))
        elif op == b"W":
            state["clip"] = rectangle
        elif op == b"m":
            if points:
                paths.append(points)
            points = [list(map(float, values))]
        elif op == b"l":
            points.append(list(map(float, values)))
        elif op in {b"S", b"n"}:
            if points:
                paths.append(points)
            color = state["color"]
            if (op == b"S" and color is not None and len(set(color)) > 1
                    and state["clip"] is not None and sum(map(len, paths)) > 3):
                result.append(dict(state, points=[p for path in paths for p in path]))
            paths, points = [], []
    return result


def calibrated(points, x0, xscale, y0, yscale, offset=0.):
    points = np.asarray(points)
    times = (points[:, 0] - x0) / xscale
    valid = abs(times - np.rint(times)) < 2e-5
    samples = {int(round(t)): float(offset + (y - y0) / yscale)
               for t, y in zip(times[valid], points[valid, 1])}
    return {"times": sorted(samples), "values": [samples[t] for t in sorted(samples)]}


def extract(archive):
    with tarfile.open(archive) as source:
        paths = {name: line_paths(source.extractfile(path).read()) for name, path in FIGURES.items()}
    fig7 = {}
    for line in paths["fig7"]:
        if abs(line["clip"][2] - 446.4) > .01:
            continue
        name = "mps" if line["dash"] else "dmt"
        fig7[name] = calibrated(line["points"], 62.820906464, 11.16,
                                56.3402954254, 264., .2)
    clips = sorted({tuple(line["clip"]) for line in paths["fig8"]
                    if abs(line["clip"][2] - 664.6435373696) < .01}, key=lambda r: -r[1])
    fig8 = {}
    for length, clip in zip((16, 20, 24), clips):
        series, colors = {}, []
        for line in paths["fig8"]:
            if tuple(line["clip"]) != clip:
                continue
            color = line["color"]
            if color not in colors:
                colors.append(color)
            cap = 16 * 2 ** colors.index(color)
            name = f"{'mps' if line['dash'] else 'dmt'}-chi{cap}"
            series[name] = calibrated(line["points"], clip[0], clip[2] / 100,
                                       clip[1], clip[3] / .03)
        fig8[str(length)] = series
    if len(fig7.get("dmt", {}).get("times", [])) != 40:
        raise ValueError("unexpected Fig. 7 PDF layout")
    if len(fig8.get("16", {}).get("mps-chi256", {}).get("times", [])) != 100:
        raise ValueError("unexpected Fig. 8 reference layout")
    return {"source": "https://arxiv.org/src/1707.01506v2", "figure_files": FIGURES,
            "kind": "digitized PDF vector coordinates; not original raw simulation data",
            "coordinate_error_scale": 3e-9, "fig7": fig7, "fig8": fig8}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/tmp/white2018_figures.json"))
    args = parser.parse_args()
    result = extract(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
