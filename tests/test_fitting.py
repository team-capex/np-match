from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from ase import Atoms
from np_match.workflow import fitting
from np_match.workflow.fitting import (
    ScatteringData,
    fit_dataset,
    get_fq_loss,
    get_particle_info,
    get_rwp,
    load_experimental_data,
)
from torch import Tensor


def test_load_experimental_data_normalizes_and_closes_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "patterns.h5"
    with h5py.File(path, "w") as data_file:
        sample = data_file.create_group("sample_3")
        sample.create_dataset(
            "Fq",
            data=np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 1.0]]),
        )
        data_file.create_group("metadata")

    [pattern] = load_experimental_data(path)

    assert pattern.sample == 3
    assert pattern.path == str(path)
    assert torch.equal(pattern.q, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(pattern.fq, torch.tensor([0.5, 1.0, 0.25]))
    assert pattern.q.is_contiguous()
    assert pattern.fq.is_contiguous()

    with h5py.File(path, "r+") as data_file:
        assert "sample_3" in data_file


def test_fit_dataset_uses_explicit_qdamp_and_preserves_result_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exp = ScatteringData(
        q=torch.tensor([1.0, 2.0]),
        fq=torch.tensor([2.0, 4.0]),
    )
    captured: dict[str, float] = {}

    def fake_fit_xscale(
        q: Tensor,
        iq: Tensor,
        form_avg_sq: Tensor,
        size: int,
        exp_scat: ScatteringData,
        biso: float,
        qdamp: float,
    ) -> float:
        return 1.0

    def fake_compute_fq(
        form_avg_sq: Tensor | np.ndarray,
        size: int,
        q: Tensor | np.ndarray,
        iq_int: Tensor | np.ndarray,
        biso: float | Tensor | np.ndarray,
        qdamp: float,
        xscale: float = 1.0,
        q_ref: Tensor | np.ndarray | None = None,
    ) -> tuple[Tensor, ...]:
        captured["qdamp"] = qdamp
        return (torch.tensor([1.0, 2.0]),)

    monkeypatch.setattr(fitting, "fit_xscale", fake_fit_xscale)
    monkeypatch.setattr(fitting, "compute_fq", fake_compute_fq)

    result = fit_dataset(
        q=torch.tensor([1.0, 2.0]),
        iq=torch.tensor([3.0, 4.0]),
        form_avg_sq=torch.tensor([1.0, 1.0]),
        size=2,
        exp_scat=exp,
        biso=0.0,
        qdamp=0.17,
        verbose=False,
    )

    assert captured["qdamp"] == 0.17
    assert set(result) == {
        "xscale",
        "yscale",
        "biso",
        "mse_fq",
        "rwp_fq",
        "fq",
    }
    assert result["xscale"] == 1.0
    assert result["yscale"] == 2.0
    assert result["biso"] == 0.0
    assert result["mse_fq"] == 0.0
    assert result["rwp_fq"] == 0.0
    np.testing.assert_array_equal(result["fq"], [2.0, 4.0])


def test_fit_metrics() -> None:
    fq = torch.tensor([1.0, 2.0])
    ref = torch.tensor([2.0, 4.0])

    assert get_fq_loss(fq, ref) == (0.0, 2.0)
    assert get_rwp(2 * fq, ref).item() == 0.0


def test_particle_metadata_from_atoms() -> None:
    atoms = Atoms("Au2", positions=[[0, 0, 0], [1, 0, 0]])
    atoms.info.update(
        family="dec",
        wulff=True,
        miller_indices=[[1, 1, 1]],
        surface_energies=[2.0],
        twin_energy=1.0,
    )

    result = get_particle_info(atoms)

    assert result["family"] == "dec"
    assert result["wulff"] is True
    assert result["size"] == 2
    assert result["max_miller"] == 1
    assert result["twin_111_ratio"] == 0.5


def test_load_experimental_data_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_experimental_data(tmp_path / "missing.h5")
