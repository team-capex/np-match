import torch
from ase import Atoms
from ase.io import read
from np_match.debye import BinnedSingleDebyeCalculator, compute_iq
from np_match.workflow.config import load_workflow_config
from np_match.workflow.utils import monitor_gpu
from tqdm import tqdm


def main(
    batch_range: tuple[int, int],
    *args,
    config_path: str | None = None,
    **kwargs,
) -> tuple[bool, None]:
    config = load_workflow_config(config_path, __file__)
    images: list[Atoms] = read(
        config.path("structures"), index=slice(*batch_range)
    )  # type: ignore
    scattering = config.section("scattering")
    hardware = config.section("hardware")
    calc = BinnedSingleDebyeCalculator(
        **scattering,
        device=hardware["device"],
    )
    monitor_gpu(
        interval=hardware["gpu_monitor_interval"],
        device=hardware["device"],
    )
    iq_dir = config.path("iq_dir")
    iq_dir.mkdir(parents=True, exist_ok=True)
    filename = iq_dir / f"iq_{batch_range[0]}-{batch_range[1]}.pt"
    try:
        iq_list = torch.load(filename)
    except Exception:
        iq_list = []
    for counter, atoms in enumerate(
        tqdm(images[len(iq_list) :]), start=len(iq_list)
    ):
        iq_list.append(compute_iq(atoms, calc))
        if counter % 100 == 0:
            torch.save(iq_list, filename)
    assert len(iq_list) == len(images), (len(iq_list), len(images))
    torch.save(iq_list, filename)
    return True, None


if __name__ == "__main__":
    main(batch_range=(0, 3655))
