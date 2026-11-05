import subprocess
import sys

from np_match.workflow.config import load_workflow_config


def main(
    batches: list[list[int]],
    job_idx: int,
    *args,
    config_path: str | None = None,
    **kwargs,
) -> tuple[bool, dict[str, list[int] | int]]:
    config = load_workflow_config(config_path, __file__)
    num_processes = len(batches[0])
    subprocess.run(
        [
            "mpiexec",
            "-n",
            str(num_processes),
            "python",
            "-m",
            "mpi4py",
            "../scripts/run_relax.py",
            "--config",
            str(config.source),
            str(job_idx),
            str(batches),
        ],
        check=True,
        text=True,
        stderr=sys.stderr,
        stdout=sys.stdout,
    )
    batch_indices = [idx for batch in batches for idx in batch]
    return True, dict(batch_indices=batch_indices, job_idx=job_idx)
