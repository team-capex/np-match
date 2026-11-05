from bisect import bisect_left
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ase.io import iread
from np_match.workflow.config import load_workflow_config
from perqueue import DynamicWidthGroup, PersistentQueue, Task, Workflow

_Path = str
_SimIdx = int
_TrajIdx = int
ImageRef = tuple[_Path, _TrajIdx, _SimIdx]


@dataclass
class Partition:
    min_atoms: int
    max_atoms: int
    batch_sizes: Sequence[int]
    resources: str
    sizes: Sequence[int]
    mace_speed: float
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
    value: int = 0


def read_relaxed_images(
    relax_dir: Path, fmax: float
) -> tuple[list[ImageRef], list[int]]:
    images: list[ImageRef] = []
    sizes: list[int] = []
    for trajectory in relax_dir.glob(f"*/*_{fmax}.traj"):
        for image_idx, atoms in enumerate(iread(trajectory, index=":")):
            sim_idx = int(atoms.info["sim_idx"])
            images.append((str(trajectory), image_idx, sim_idx))
            sizes.append(len(atoms))
    sorted_items = sorted(zip(images, sizes), key=lambda item: item[0][2])
    sorted_images = [image for image, _ in sorted_items]
    sorted_sizes = [size for _, size in sorted_items]
    return sorted_images, sorted_sizes


def batch_images(
    sizes: list[int],
    indices: list[int],
    allowed_batch_sizes: Iterable[int],
    max_atoms: int,
) -> list[list[int]]:
    batch_sizes_set = set(allowed_batch_sizes)
    batch_sizes = sorted(batch_sizes_set)

    if not batch_sizes or batch_sizes[0] != 1:
        raise ValueError("Smallest allowed batch size must be 1")
    if any(sizes[i] > max_atoms for i in indices):
        raise ValueError("An image exceeds the partition's max_atoms")

    batches: list[list[int]] = [[]]
    batch_size = None
    for i in indices:
        batches[-1].append(i)

        if sum(sizes[j] for j in batches[-1]) > max_atoms:
            new_batch = batches[-1][batch_size:]
            batches[-1] = batches[-1][:batch_size]

            if sum(sizes[j] for j in new_batch) > max_atoms:
                batches.append(new_batch[:-1])
                batches.append(new_batch[-1:])
                batch_size = 1
            else:
                batches.append(new_batch)
                index = bisect_left(batch_sizes, len(batches[-1]))
                if batch_sizes[index] > len(batches[-1]):
                    index -= 1
                batch_size = batch_sizes[index]

        if len(batches[-1]) in batch_sizes_set:
            batch_size = len(batches[-1])

        if batch_size == batch_sizes[-1]:
            batches.append([])
            batch_size = None
    return [batch for batch in batches if batch]


def estimate_batch_time(
    sizes: list[int], indices: list[int], partition: Partition, num_steps: int
) -> float:
    num_atoms = sum(sizes[i] for i in indices)
    return num_atoms * num_steps / partition.mace_speed


def divide_batches_in_jobs(
    sizes: list[int],
    batches: list[list[int]],
    partition: Partition,
    job_hours: float,
    num_steps: int,
) -> list[list[list[int]]]:
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


def get_md_task(
    batches: list[list[ImageRef]],
    pointer: Pointer,
    resources: str,
    config_path: str,
) -> Task:
    task = Task(
        "../scripts/launch_batch_md.py",
        {
            "batches": batches,
            "job_idx": pointer.value,
            "config_path": config_path,
        },
        resources,
        name="md",
    )
    pointer.value += 1
    return task


def get_pdf_task(config_path: str, resources: str) -> Task:
    return Task(
        "../scripts/calc_iq.py",
        {"config_path": config_path},
        resources,
        name="pdf",
    )


def get_fit_dwg(config_path: str, resources: str) -> DynamicWidthGroup:
    return DynamicWidthGroup(
        Task(
            "../scripts/fit_datasets.py",
            {"config_path": config_path},
            resources,
            name="fit",
        )
    )


def main():
    config = load_workflow_config(None, __file__)
    config_path = str(config.source)
    scheduler = config.section("scheduler")
    fmax = config.value("relax", "fmax")
    images, sizes = read_relaxed_images(config.path("relax_dir"), fmax)
    if not images:
        raise FileNotFoundError(
            f"No final relaxation trajectories found in {config.path('relax_dir')}"
        )

    expected_particle_count = sum(
        1 for _ in iread(config.path("structures"), index=":")
    )
    sim_indices = [image[2] for image in images]
    expected_indices = set(range(expected_particle_count))
    if set(sim_indices) != expected_indices:
        missing = sorted(expected_indices - set(sim_indices))
        unexpected = sorted(set(sim_indices) - expected_indices)
        raise ValueError(
            f"Relaxed trajectories do not match structures: "
            f"{missing=}, {unexpected=}"
        )

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
    assigned_indices = [
        index for partition in partitions for index in partition.indices
    ]
    if sorted(assigned_indices) != list(range(len(images))):
        raise ValueError(
            "Scheduler partitions must assign every relaxed structure exactly once"
        )
    partition_batches = [
        batch_images(
            sizes,
            partition.indices,
            partition.batch_sizes,
            partition.max_atoms,
        )
        for partition in partitions
    ]
    md_steps = config.value("molecular_dynamics", "nsteps")
    partition_job_batches = [
        divide_batches_in_jobs(
            sizes,
            batches,
            partition,
            scheduler["md_job_hours"],
            md_steps,
        )
        for batches, partition in zip(partition_batches, partitions)
    ]

    pointer = Pointer()
    workflows = []
    for partition, job_batches in zip(partitions, partition_job_batches):
        for job in job_batches:
            if not job:
                continue
            image_batches = [
                [images[index] for index in batch] for batch in job
            ]
            workflows.append(
                Workflow(
                    [
                        get_md_task(
                            image_batches,
                            pointer,
                            partition.resources,
                            config_path,
                        ),
                        get_pdf_task(config_path, scheduler["iq_resources"]),
                        get_fit_dwg(config_path, scheduler["fit_resources"]),
                    ]
                )
            )

    task_gather = Task(
        "../scripts/gather_fits.py",
        {"config_path": config_path},
        scheduler["gather_resources"],
        name="gather",
    )
    workflow = Workflow({job_workflow: [] for job_workflow in workflows})
    with PersistentQueue() as pq:
        pq.submit(Workflow([workflow, task_gather]))


if __name__ == "__main__":
    main()
