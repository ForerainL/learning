"""Generate job.toml files from a template and submit them with kbatch."""

from __future__ import annotations

import itertools
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

TEMPLATE_PATH = "job.template.toml"
OUTPUT_DIR = "generated_jobs"
RECEIPT_PATH = "submit_receipt.txt"
MAX_PARALLEL_SUBMITS = 8


@dataclass(frozen=True)
class JobSpec:
    run_name: str
    lr: float
    hidden_dim: int
    batch_size: int
    seed: int


def read_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def format_run_name(lr: float, hidden_dim: int, batch_size: int, seed: int) -> str:
    lr_tag = str(lr).replace(".", "p")
    return f"lr{lr_tag}_hd{hidden_dim}_bs{batch_size}_seed{seed}"


def build_grid() -> List[JobSpec]:
    hidden_dims = [64, 128]
    lrs = [1e-3, 3e-4]
    batch_sizes = [4096]
    seeds = [1, 2]

    specs = []
    for hidden_dim, lr, batch_size, seed in itertools.product(hidden_dims, lrs, batch_sizes, seeds):
        run_name = format_run_name(lr, hidden_dim, batch_size, seed)
        specs.append(JobSpec(run_name=run_name, lr=lr, hidden_dim=hidden_dim, batch_size=batch_size, seed=seed))
    return specs


def render_template(template: str, values: Dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def write_job_file(template: str, spec: JobSpec) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    job_filename = f"job_{spec.run_name}.toml"
    job_path = os.path.join(OUTPUT_DIR, job_filename)
    rendered = render_template(
        template,
        {
            "RUN_NAME": spec.run_name,
            "LR": str(spec.lr),
            "HIDDEN_DIM": str(spec.hidden_dim),
            "BATCH_SIZE": str(spec.batch_size),
            "SEED": str(spec.seed),
        },
    )
    with open(job_path, "w", encoding="utf-8") as handle:
        handle.write(rendered)
    return job_path


def submit_job(job_path: str) -> Tuple[str, str, int]:
    result = subprocess.run(
        ["kbatch", "submit", job_path],
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    output = stdout if stdout else stderr
    return job_path, output, result.returncode


def submit_jobs(job_paths: Iterable[str]) -> List[Tuple[str, str]]:
    receipts: List[Tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_SUBMITS) as executor:
        future_map = {executor.submit(submit_job, path): path for path in job_paths}
        for future in as_completed(future_map):
            job_path = future_map[future]
            try:
                path, output, returncode = future.result()
            except Exception as exc:  # noqa: BLE001 - surface failures but keep processing
                output = f"Submission failed: {exc}"
                returncode = 1
                path = job_path

            if returncode != 0:
                output = f"ERROR({returncode}): {output}"

            print(f"Submitted {os.path.basename(path)}")
            print(output)
            receipts.append((os.path.basename(path), output))
    return receipts


def write_receipt(receipts: List[Tuple[str, str]]) -> None:
    with open(RECEIPT_PATH, "w", encoding="utf-8") as handle:
        for job_file, output in receipts:
            handle.write(f"{job_file}\t{output}\n")


def main() -> None:
    template = read_template(TEMPLATE_PATH)
    specs = build_grid()
    job_paths = [write_job_file(template, spec) for spec in specs]
    receipts = submit_jobs(job_paths)
    write_receipt(receipts)


if __name__ == "__main__":
    main()
