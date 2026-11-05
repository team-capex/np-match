from bisect import bisect_left
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ase.io import iread
from np_match.workflow.config import load_workflow_config
from perqueue import DynamicWidthGroup, PersistentQueue, Task, Workflow


@dataclass
class Partition:
    min_atoms: int
    max_atoms: int
    batch_sizes: Sequence[int]
    resources: str
    sizes: Sequence[int]
    mace_speed: float  # unit: num_atoms x num_steps / time [s^-1]
    indices: list[int] = field(init=False)

    def __post_init__(self):
        included_indices = [
            i
            for i, size in enumerate(self.sizes)
            if self.min_atoms < size <= self.max_atoms
        ]
        self.indices = sorted(included_indices, key=lambda i: self.sizes[i])


@dataclass
class Pointer:
    value: int


def get_relax_task(
    batches: list[list[int]],
    pointer: Pointer,
    partition: Partition,
    config_path: str,
):
    assert all(len(batch) == len(batches[0]) for batch in batches)
    task = Task(
        "../scripts/launch_batch_relax.py",
        {
            "batches": batches,
            "job_idx": pointer.value,
            "config_path": config_path,
        },
        partition.resources,
        name="relax",
    )
    pointer.value += 1
    return task


def get_pdf_task(config_path: str, resources: str):
    return Task(
        "../scripts/calc_iq.py",
        {"config_path": config_path},
        resources,
        name="pdf",
    )


def get_fit_dwg(config_path: str, resources: str):
    return DynamicWidthGroup(
        Task(
            "../scripts/fit_datasets.py",
            {"config_path": config_path},
            resources,
            name="fit",
        )
    )


def batch_images(
    sizes: list[int],
    indices: list[int],
    allowed_batch_sizes: Iterable[int],
    max_atoms: int,
) -> list[list[int]]:
    # Unique and sorted allowed batch sizes
    batch_sizes_set = set(allowed_batch_sizes)
    batch_sizes = sorted(batch_sizes_set)

    assert all(sizes[i] <= max_atoms for i in indices)
    assert batch_sizes[0] == 1, "Smallest allowed batch size must be 1"

    batches: list[list[int]] = [[]]
    batch_size = None
    for i in indices:
        # Add index to newest batch
        batches[-1].append(i)

        # Split batch if it contains more than max_atoms
        if sum(sizes[j] for j in batches[-1]) > max_atoms:
            new_batch = batches[-1][batch_size:]
            batches[-1] = batches[-1][:batch_size]

            # Split the remainder if it contains too many atoms
            if sum(sizes[j] for j in new_batch) > max_atoms:
                batches.append(new_batch[:-1])
                batches.append(new_batch[-1:])
                batch_size = 1
            else:
                batches.append(new_batch)

                # Get the new batch size cut
                index = bisect_left(batch_sizes, len(batches[-1]))
                if batch_sizes[index] > len(batches[-1]):
                    index -= 1
                batch_size = batch_sizes[index]

        # Update batch size cut
        if len(batches[-1]) in batch_sizes_set:
            batch_size = len(batches[-1])

        # New batch if we hit the maximum allowed batch size
        if batch_size == batch_sizes[-1]:
            batches.append([])
            batch_size = None  # reset
    return batches


def estimate_batch_time(
    sizes: list[int], indices: list[int], partition: Partition, num_steps: int
):
    """Estimate time to do num_steps force evaluations on the
    partition for the given batch.
    """
    num_atoms = sum(sizes[i] for i in indices)
    return num_atoms * num_steps / partition.mace_speed


def divide_batches_in_jobs(
    sizes: list[int],
    batches: list[list[int]],
    partition: Partition,
    job_hours: float,
    num_steps: int,
):
    # Sort by batch size to group batches with same batch size into jobs
    batches.sort(key=len)
    job_batches: list[list[list[int]]] = [[]]
    same_batch_size = True
    job_seconds = job_hours * 3600
    for batch in batches:
        job_time = sum(
            estimate_batch_time(sizes, indices, partition, num_steps)
            for indices in job_batches[-1]
        )
        batch_time = estimate_batch_time(sizes, batch, partition, num_steps)
        if job_batches[-1]:
            same_batch_size = len(batch) == len(job_batches[-1][0])
        if job_time + batch_time > job_seconds or not same_batch_size:
            job_batches.append([])
            same_batch_size = True
        job_batches[-1].append(batch)
    return job_batches


def main():
    config = load_workflow_config(None, __file__)
    config_path = str(config.source)
    scheduler = config.section("scheduler")
    sizes = [len(atoms) for atoms in iread(config.path("structures"), ":")]
    job_hours = scheduler["relax_job_hours"]
    num_steps = scheduler["relax_steps"]

    partitions = [
        Partition(
            min_atoms=partition["min_atoms"],
            max_atoms=partition["max_atoms"],
            batch_sizes=partition["batch_sizes"],
            resources=partition["resources"],
            sizes=sizes,
            mace_speed=partition["mace_speed"],
        )
        for partition in scheduler["partitions"]
    ]
    # Batches of image indices for each partition
    partition_batches: list[list[list[int]]] = [
        batch_images(sizes, p.indices, p.batch_sizes, p.max_atoms)
        for p in partitions
    ]

    # Cut batches into x hour segments, fit for job submission
    partition_job_batches: list[list[list[list[int]]]] = [
        divide_batches_in_jobs(sizes, batches, partition, job_hours, num_steps)
        for batches, partition in zip(partition_batches, partitions)
    ]

    task_gather = Task(
        "../scripts/gather_fits.py",
        {"config_path": config_path},
        scheduler["gather_resources"],
        name="gather",
    )
    job_idx_pointer = Pointer(0)
    wf = Workflow(
        {
            Workflow(
                [
                    get_relax_task(
                        batches, job_idx_pointer, partition, config_path
                    ),
                    get_pdf_task(config_path, scheduler["iq_resources"]),
                    get_fit_dwg(config_path, scheduler["fit_resources"]),
                ]
            ): []
            for partition, job_batches in zip(
                partitions, partition_job_batches
            )
            for batches in job_batches
            if batches[0]
        }
    )

    with PersistentQueue() as pq:
        pq.submit(Workflow([wf, task_gather]))


if __name__ == "__main__":
    main()
