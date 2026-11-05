from collections.abc import Iterable
from typing import Protocol, Self, TypeVar

from ase import Atoms
from ase.io import read
from np_match.workflow.config import load_workflow_config
from perqueue import PersistentQueue, StaticWidthGroup, Task, Workflow


class SupportsAddCompare(Protocol):
    def __add__(self, value: Self, /) -> Self: ...

    def __ge__(self, value: Self, /) -> bool: ...


T = TypeVar("T", bound=SupportsAddCompare)


def batch_iterable(
    data: Iterable[T],
    max_sum: T,
    sum_start: T,
) -> tuple[list[list[T]], list[list[int]]]:
    """Group iterable items into batches according to a running sum."""
    indices: list[list[int]] = [[]]
    batches: list[list[T]] = [[]]
    for idx, x in enumerate(data):
        if sum(batches[-1], start=sum_start) >= max_sum:
            batches.append([])
            indices.append([])
        batches[-1].append(x)
        indices[-1].append(idx)
    return batches, indices


def compress_indices(indices: list[list[int]]) -> list[tuple[int, int]]:
    compressed: list[tuple[int, int]] = []
    for index_group in indices:
        if not index_group:
            continue
        start = index_group[0]
        end = index_group[-1] + 1
        assert list(range(start, end)) == index_group, (
            f"Batch {index_group!r} is not contiguous and cannot be represented as "
            f"range({start}, {end})."
        )
        compressed.append((start, end))
    return compressed


def estimate_debye_time(
    size: int, num_images: int = 1, prefactor: float = 2.26e-11
) -> float:
    return num_images * prefactor * size**2


def main():
    config = load_workflow_config(None, __file__)
    config_path = str(config.source)
    scheduler = config.section("scheduler")
    images: list[Atoms] = read(config.path("structures"), ":")  # type: ignore
    debye_time_prefactor = config.value(
        "hardware", "debye_time_prefactor", 2.26e-11
    )
    times = [
        estimate_debye_time(len(atoms), prefactor=debye_time_prefactor)
        for atoms in images
    ]
    _, batches = batch_iterable(times, scheduler["iq_job_time"], 0.0)
    batches_compressed: list[tuple[int, int]] = compress_indices(batches)

    fit_batch_size = scheduler["fit_batch_size"]

    wf_iq = Workflow(
        {
            Workflow(
                [
                    Task(
                        "../scripts/calc_iq.py",
                        {
                            "batch_range": batch_range,
                            "config_path": config_path,
                        },
                        scheduler["iq_resources"],
                        name="pdf",
                    ),
                    StaticWidthGroup(
                        Task(
                            "../scripts/fit_datasets.py",
                            {
                                "batch_range": batch_range,
                                "batch_size": fit_batch_size,
                                "config_path": config_path,
                            },
                            scheduler["fit_resources"],
                            name="fit",
                        ),
                        width=(
                            batch_range[1]
                            - batch_range[0]
                            + fit_batch_size
                            - 1
                        )
                        // fit_batch_size,
                    ),
                ]
            ): []
            for batch_range in batches_compressed
        }
    )
    task_gather = Task(
        "../scripts/gather_fits.py",
        {"config_path": config_path},
        scheduler["gather_resources"],
        name="gather",
    )

    with PersistentQueue() as pq:
        pq.submit(Workflow([wf_iq, task_gather]))


if __name__ == "__main__":
    main()
