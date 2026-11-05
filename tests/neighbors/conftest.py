import pytest
import torch

_DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    _DEVICES.append(torch.device("cuda"))


@pytest.fixture(params=_DEVICES, ids=lambda device: str(device))
def device(request: pytest.FixtureRequest) -> torch.device:
    return request.param
