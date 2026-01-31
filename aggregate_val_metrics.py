import json
from pathlib import Path

import pandas as pd

PARAM_KEYS = ["lr", "bs", "wd", "dp", "hd", "L"]
METRIC_KEYS = ["IC", "RankIC", "QuantileReturn", "R2"]


def cast(v: str):
    try:
        return int(v)
    except Exception:
        try:
            return float(v)
        except Exception:
            return v


def parse_filename(fname: str) -> dict:
    stem = Path(fname).stem
    if stem.endswith("_val_metrics"):
        stem = stem[: -len("_val_metrics")]

    tokens = stem.split("_")

    params = {}
    for tok in tokens:
        for k in PARAM_KEYS:
            if tok.startswith(k):
                params[k] = cast(tok[len(k) :])
                break

    return params


def collect(root: str) -> pd.DataFrame:
    rows = []

    for path in Path(root).rglob("*_val_metrics.json"):
        params = parse_filename(path.name)

        with open(path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        row = {**params}
        for m in METRIC_KEYS:
            row[m] = metrics.get(m)

        row["file"] = path.name
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = collect("./results")
    print(df.head())
    df.to_csv("val_metrics_summary.csv", index=False)
