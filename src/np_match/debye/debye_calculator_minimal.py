# This file contains a minimal version of the DebyeCalculator class, modified
# from the implementation of release v1.0.14:
# https://github.com/FrederikLizakJohansen/DebyeCalculator/releases/tag/v1.0.14

import math
import warnings
from importlib.resources import files
from typing import Any

import yaml

# Handle import of torch (prerequisite)
try:
    import torch
except ModuleNotFoundError:
    raise ImportError(
        "\n\nDebyeCalculator requires PyTorch, which is not installed. "
        "Please install PyTorch before using DebyeCalculator. "
        "Follow the instructions on the official PyTorch website: "
        "https://pytorch.org/get-started/locally/. "
        "For more information about DebyeCalculator, visit the GitHub repository: "
        "https://github.com/FrederikLizakJohansen/DebyeCalculator"
    )


import numpy as np


def Profiler(*args, **kwargs):
    raise NotImplementedError()


class DebyeCalculator:
    """
    Calculate the scattering intensity I(Q) through the Debye scattering equation, the Total Scattering Structure Function S(Q),
    the Reduced Total Scattering Function F(Q), and the Reduced Atomic Pair Distribution Function G(r) for a given atomic structure.
    """

    def __init__(
        self,
        qmin: float = 1.0,
        qmax: float = 30.0,
        qstep: float | None = None,
        qdamp: float = 0.04,
        rmin: float = 0.0,
        rmax: float = 20.0,
        rstep: float = 0.01,
        rthres: float = 0.0,
        biso: float = 0.3,
        device: str = "cuda",
        batch_size: int = 10_000,
        lorch_mod: bool = False,
        radiation_type: str | None = None,
        rad_type: str | None = None,
        profile: bool = False,
        _max_batch_size: int = 4000,
        _lightweight_mode: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Initialize a DebyeCalculator instance with specified parameters.

        Args:
            qmin (float): Minimum q-value for the scattering calculation. Default is 1.0.
            qmax (float): Maximum q-value for the scattering calculation. Default is 30.0.
            qstep (float): Step size for the q-values in the scattering calculation. Default is None, and is calculated as [pi / (rmax + rstep)] if not provided.
            qdamp (float): Damping parameter caused by the truncated Q-range of the Fourier transformation. Default is 0.04.
            rmin (float): Minimum r-value for the pair distribution function (PDF) calculation. Default is 0.0.
            rmax (float): Maximum r-value for the PDF calculation. Default is 20.0.
            rstep (float): Step size for the r-values in the PDF calculation. Default is 0.01.
            rthres (float): Threshold value for exclusion of distances below this value in the scattering calculation. Default is 0.0.
            biso (float): Debye-Waller isotropic atomic displacement parameter. Default is 0.3.
            device (str): Device to use for computations ('cuda' for CUDA-enabled GPU's or 'cpu' for CPU)
            batch_size (int or None): Batch size for computation. If None, the batch size will be automatically set. Default is None.
            lorch_mod (bool): Flag to enable Lorch modification. Default is False.
            radiation_type (str): Type of radiation for form factor calculations ('xray' or 'neutron'). Default is 'xray'
            rad_type (str): Alias for 'radiation_type'. Default is 'xray'.
            profile (bool): Activate profiler. Default is False.
            dtype (torch.dtype): Data type for tensors. 32-bit and 64-bit floats are currently supported. Default is torch.float32.
        """

        # Handling CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            warnings.warn(
                "Warning: Your system might have a CUDA-enabled GPU, but CUDA is not available. Computations will run on the CPU instead. "
                "For optimal performance, please install Pytorch with CUDA support. "
                "If you do not have a CUDA-enabled GPU, you can surpress this warning by specifying the 'device' argument as 'cpu'",
                UserWarning,
                stacklevel=2,
            )
            self.device = "cpu"
        elif device == "cpu" and torch.cuda.is_available():
            warnings.warn(
                "Warning: Your system has a CUDA-enabled GPU, but CPU was explicitly specified for computations. "
                "To utilise GPU acceleration, omit the 'device' argument or specify 'cuda'",
                UserWarning,
                stacklevel=2,
            )
            self.device = "cpu"
        else:
            self.device = device

        # Set parameters
        self.qmin = qmin
        self.qmax = qmax
        self.qstep = qstep
        self.qdamp = qdamp
        self.rmin = rmin
        self.rmax = rmax
        self.rstep = rstep
        self.rthres = rthres
        self.biso = biso
        self.batch_size = batch_size
        self.lorch_mod = lorch_mod
        self.dtype = dtype

        if radiation_type is not None:
            self.radiation_type = radiation_type
        elif rad_type is not None:
            self.radiation_type = rad_type
        else:
            self.radiation_type = "xray"

        # Set qstep seperately (Nyquist)
        if qstep is None:
            self.qstep = math.pi / (self.rmax + self.rstep)
        else:
            self.qstep = qstep

        # Parameter constraint assertion
        self.parameter_constraint_assertion()

        # Profiler
        self.profile = profile
        if self.profile:
            self.profiler = Profiler()

        # Initialise ranges
        self.q = torch.arange(
            self.qmin,
            self.qmax,
            self.qstep,
            device=self.device,
            dtype=self.dtype,
        ).unsqueeze(-1)
        self.r = torch.arange(
            self.rmin,
            self.rmax,
            self.rstep,
            device=self.device,
            dtype=self.dtype,
        ).unsqueeze(-1)

        resource = files("np_match").joinpath("debye/elements_info.yaml")
        with resource.open("r", encoding="utf-8") as yaml_file:
            self.FORM_FACTOR_COEF = yaml.safe_load(yaml_file)

        # Form factor coefficients
        self.atomic_numbers_to_elements = {}
        for i, (key, value) in enumerate(self.FORM_FACTOR_COEF.items()):
            if i > 97:
                break
            self.atomic_numbers_to_elements[value[12]] = key

        # Formfactor retrieval lambda
        for k, v in self.FORM_FACTOR_COEF.items():
            if None in v:
                v = [value if value is not None else np.nan for value in v]
            self.FORM_FACTOR_COEF[k] = torch.tensor(
                v, device=self.device, dtype=self.dtype
            )
        if self.radiation_type.lower() in ["xray", "x"]:
            self.form_factor_func = lambda p: (
                torch.sum(
                    p[:5]
                    * torch.exp(
                        -1 * p[6:11] * (self.q / (4 * torch.pi)).pow(2)
                    ),
                    dim=1,
                )
                + p[5]
            )
        elif self.radiation_type.lower() in ["neutron", "n"]:
            self.form_factor_func = lambda p: p[11].unsqueeze(-1)
        else:
            # Should not reach this point, here for safety
            raise ValueError("Invalid radiation type")

        # Max batch size
        self._max_batch_size = _max_batch_size

        # Lightweight mode
        self._lightweight_mode = _lightweight_mode

    def __repr__(self):
        return (
            f"{self.__class__.__name__} instance:\n"
            f"{'Q-min:':<12} {self.qmin:5.2f}\n"
            f"{'Q-max:':<12} {self.qmax:5.2f}\n"
            f"{'Q-step:':<12} {self.qstep:5.2f}\n"
            f"{'Q-damp:':<12} {self.qdamp:5.2f}\n"
            f"\n"
            f"{'r-min:':<12} {self.rmin:5.2f}\n"
            f"{'r-max:':<12} {self.rmax:5.2f}\n"
            f"{'r-step:':<12} {self.rstep:5.2f}\n"
            f"{'r-thres:':<12} {self.rthres:5.2f}\n"
            f"\n"
            f"{'B-iso:':<12} {self.biso:5.2f}\n"
            f"\n"
            f"{'rad_type:':<12} {self.radiation_type}\n"
            f"{'lorch_mod:':<12} {self.lorch_mod}\n"
            f"\n"
            f"{'batch_size:':<12} {self.batch_size}\n"
            f"{'device:':<12} {self.device}\n"
            f"{'profile:':<12} {self.profile}\n"
        )

    def parameter_constraint_assertion(
        self,
    ) -> None:
        """
        Assert that all parameters meet valid constraints.
        Provides warning if certain parameters are not within certain constraints

        Raises:
            ValueError: If any of the parameters violate the specified constraints.
            UserWarning: If any of the specified parameters are not within the specified constraints.
        """

        # Positivity constraints
        if self.qmin < 0:
            raise ValueError("qmin must be non-negative.")
        if self.qmax < 0:
            raise ValueError("qmax must be non-negative.")
        if self.qstep < 0:
            raise ValueError("qstep must be non-negative.")
        if self.qdamp < 0:
            raise ValueError("qdamp must be non-negative.")
        if self.rmin < 0:
            raise ValueError("rmin must be non-negative.")
        if self.rmax < 0:
            raise ValueError("rmax must be non-negative.")
        if self.rstep < 0:
            raise ValueError("rstep must be non-negative.")
        if self.rthres < 0:
            raise ValueError("rthres must be non-negative.")
        if self.biso < 0:
            raise ValueError("biso must be non-negative.")

        # Batch-size constraints
        if self.batch_size is not None and self.batch_size < 0:
            raise ValueError("batch_size must be non-negative.")

        # Device constraints
        if self.device not in ["cpu", "cuda"]:
            raise ValueError("Invalid device")

        # Radiation type constraints
        if self.radiation_type not in ["xray", "x", "neutron", "n"]:
            raise ValueError("Invalid radiation type")

        # Optimal qstep UserWarning
        optimal_qstep = math.pi / (self.rmax + self.rstep)
        if self.qstep > optimal_qstep:
            warnings.warn(
                f"The qstep that was chosen is too large and might result in unwanted signal artifacts. With rmax={self.rmax} and rstep={self.rstep}, consider using qstep<={optimal_qstep:2.3f}.",
                UserWarning,
            )

        # Dtype
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError(
                f"Invalid dtype {self.dtype}. Only torch.float32 and torch.float64 are supported."
            )

    def update_parameters(
        self,
        **kwargs: Any,
    ) -> None:
        """
        Set or update the parameters of the DebyeCalculator.

        Parameters:
            **kwargs: Arbitrary keyword arguments to update the parameters.

        Raises:
            ValueError: If any of the updated parameters violate the specified constraints.
        """

        for k, v in kwargs.items():
            try:
                setattr(self, k, v)
                if k == "rad_type":
                    self.radiation_type = v
            except:
                print(
                    "Failed to update parameters because of unexpected parameter names"
                )
                return

        # Run constrain assertion
        self.parameter_constraint_assertion()

        # Re-initialise ranges
        if np.any(
            [
                k
                in [
                    "qmin",
                    "qmax",
                    "qstep",
                    "rmin",
                    "rmax",
                    "rstep",
                    "device",
                    "dtype",
                ]
                for k in kwargs
            ]
        ):
            self.q = (
                torch.arange(self.qmin, self.qmax, self.qstep)
                .unsqueeze(-1)
                .to(device=self.device, dtype=self.dtype)
            )
            self.r = (
                torch.arange(self.rmin, self.rmax, self.rstep)
                .unsqueeze(-1)
                .to(device=self.device, dtype=self.dtype)
            )
            for key, val in self.FORM_FACTOR_COEF.items():
                self.FORM_FACTOR_COEF[key] = val.to(device=self.device)

        if np.any([k in ["rad_type", "radiation_type"] for k in kwargs]):
            if self.radiation_type.lower() in ["xray", "x"]:
                self.form_factor_func = lambda p: (
                    torch.sum(
                        p[:5]
                        * torch.exp(
                            -1 * p[6:11] * (self.q / (4 * torch.pi)).pow(2)
                        ),
                        dim=1,
                    )
                    + p[5]
                )
            elif self.radiation_type.lower() in ["neutron", "n"]:
                self.form_factor_func = lambda p: p[11].unsqueeze(-1)
            else:
                # Should not reach this point, here for safety
                raise ValueError("Invalid radiation type")
