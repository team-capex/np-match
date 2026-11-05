import subprocess
import sys

from np_match.workflow.config import load_workflow_config

ImageRef = tuple[str, int, int]


def main(
    batches: list[list[ImageRef]],
    job_idx: int,
    *args,
    config_path: str | None = None,
    **kwargs,
) -> tuple[bool, dict[str, list[int] | int]]:
    config = load_workflow_config(config_path, __file__)
    if not batches or not batches[0]:
        raise ValueError(f"No MD batches provided for job {job_idx}")
    num_processes = len(batches[0])
    if any(len(batch) != num_processes for batch in batches):
        raise ValueError("All MD batches in a job must have the same size")

    sim_indices = [image[2] for batch in batches for image in batch]
    subprocess.run(
        [
            "mpiexec",
            "-n",
            str(num_processes),
            "python",
            "-m",
            "mpi4py",
            "../scripts/run_md.py",
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
    return True, dict(job_idx=job_idx, sim_indices=sim_indices)
