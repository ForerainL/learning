"""Grid-search sweep runner for Train1Config.

Each run is independent and saves outputs under cfg.save_dir / cfg.run_name.
"""

from __future__ import annotations

import itertools
from datetime import datetime

try:
    from train import Train1Config, run_once  # type: ignore
except Exception:  # pragma: no cover - fallback when train.py lacks the entry point
    import train1
    from train1 import Train1Config

    def run_once(cfg: Train1Config):  # type: ignore
        original = train1.Train1Config
        train1.Train1Config = lambda: cfg  # type: ignore
        try:
            return train1.main()
        finally:
            train1.Train1Config = original


def main() -> None:
    model_names = ["gru", "lstm", "tcn", "transformer"]
    learning_rates = [1e-3, 5e-4]
    batch_sizes = [256, 512]
    weight_decays = [0.0, 1e-4]
    dropouts = [0.0, 0.1]
    hidden_dims = [16, 32]
    num_layers = [1, 2]

    grid = list(
        itertools.product(
            model_names,
            learning_rates,
            batch_sizes,
            weight_decays,
            dropouts,
            hidden_dims,
            num_layers,
        )
    )

    total = len(grid)
    for idx, (model_name, lr, batch_size, weight_decay, dropout, hidden_dim, layers) in enumerate(grid, start=1):
        cfg = Train1Config()
        cfg.model_name = model_name
        cfg.batch_size = batch_size
        cfg.dropout = dropout
        cfg.hidden_dim = hidden_dim
        cfg.num_layers = layers

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        cfg.run_name = (
            f"{model_name}_lr{lr:g}_bs{batch_size}_wd{weight_decay:g}_"
            f"do{dropout:g}_h{hidden_dim}_L{layers}_{timestamp}"
        )

        print(f"[{idx}/{total}] run={cfg.run_name}")
        run_once(cfg)


if __name__ == "__main__":
    main()
