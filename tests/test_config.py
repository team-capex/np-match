from pathlib import Path

import pytest
from np_match.workflow.config import WorkflowConfig, load_workflow_config


def test_paths_resolve_relative_to_config_file(tmp_path: Path) -> None:
    config_dir = tmp_path / "workflow"
    config_dir.mkdir()
    settings_file = config_dir / "settings.toml"
    settings_file.write_text(
        '[paths]\ninput = "../structures.traj"\n', encoding="utf-8"
    )

    config = WorkflowConfig.load(settings_file)

    input_path = config.path("input")
    assert isinstance(input_path, Path)
    assert input_path.is_absolute()
    assert input_path == (tmp_path / "structures.traj").resolve()


def test_absolute_paths_are_preserved(tmp_path: Path) -> None:
    config_file = tmp_path / "settings.toml"
    target = tmp_path / "structures.traj"
    config_file.write_text(f'[paths]\ninput = "{target}"\n', encoding="utf-8")

    assert WorkflowConfig.load(config_file).path("input") == target.resolve()


def test_missing_required_path_names_config(tmp_path: Path) -> None:
    config_file = tmp_path / "settings.toml"
    config_file.write_text("[paths]\n", encoding="utf-8")

    with pytest.raises(KeyError, match="paths.input"):
        WorkflowConfig.load(config_file).path("input")


# Maybe this test should be outside of the np-match package, but it is convenient to have it here for now.
@pytest.mark.parametrize("workflow", ["ideal", "md", "relax"])
def test_script_driver_finds_workflow_config(workflow: str) -> None:
    repo = Path(__file__).resolve().parents[1]
    workflow_root = repo / "workflows" / workflow
    driver = workflow_root / "scripts" / "pq_driver.py"

    assert driver.is_file()
    assert (workflow_root / "perqueue").is_dir()
    config = load_workflow_config(None, driver)

    assert config.source == workflow_root / "config.toml"
    assert config.path("structures") == (
        repo / "workflows" / "generate_particles" / "out" / "particles.traj"
    )
    assert config.path("iq_dir") == workflow_root / "iq"
    assert config.path("fits_dir") == workflow_root / "fits"
    for output in ("iq", "fits"):
        assert (workflow_root / output / ".gitignore").is_file()
    if workflow in {"md", "relax"}:
        assert config.path(f"{workflow}_dir") == workflow_root / workflow
