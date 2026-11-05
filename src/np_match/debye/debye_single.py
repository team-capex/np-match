from abc import ABC, abstractmethod
from collections import namedtuple
from itertools import combinations_with_replacement
from math import ceil, isqrt
from typing import TypedDict

import numpy as np
import torch
from ase import Atoms
from ase.data import chemical_symbols
from torch import Tensor

from .debye_calculator_minimal import DebyeCalculator

MinStructureTuple = namedtuple("MinStructureTuple", "size xyz form_avg_sq")

torch.set_float32_matmul_precision("high")


class SingleDebyeCalculator(DebyeCalculator, ABC):
    def _initialise_structure(self, atoms: Atoms) -> MinStructureTuple:
        assert isinstance(atoms, Atoms)
        if not np.all(atoms.numbers == atoms.numbers[0]):
            raise ValueError(
                "Structure must only contain atoms of one element, got "
                f"{np.unique(atoms.numbers, return_counts=True)=}"
            )
        if len(atoms) < 2:
            raise ValueError(
                f"Structure must contain at least 2 atoms, got {len(atoms)=}"
            )
        xyz = torch.from_numpy(atoms.positions).to(
            device=self.device, dtype=self.dtype, copy=True
        )
        form_avg_sq = (
            self.form_factor_func(
                self.FORM_FACTOR_COEF[chemical_symbols[atoms.numbers[0]]]
            )
            ** 2
        )
        return MinStructureTuple(
            size=len(atoms), xyz=xyz, form_avg_sq=form_avg_sq
        )

    @abstractmethod
    def compute_iq_intermediate(
        self, structure: MinStructureTuple, compile: bool = True
    ) -> Tensor: ...


class CdistSingleDebyeCalculator(SingleDebyeCalculator):
    def compute_iq_intermediate(
        self, structure: MinStructureTuple, compile: bool = True
    ) -> Tensor:
        iq = batched_iq(
            self.q.squeeze(), structure.xyz, isqrt(self.batch_size)
        )
        iq *= structure.form_avg_sq
        return iq


class BinnedSingleDebyeCalculator(SingleDebyeCalculator):
    def get_distance_boundaries(
        self, xyz: Tensor, bin_width: float = 1e-3
    ) -> Tensor:
        """Calculate bin edges for binning distances with a fixed bin width."""
        assert xyz.ndim == 2, xyz.shape
        # Upper bound on maximum interatomic distance
        low = xyz.amin(dim=0)
        high = xyz.amax(dim=0)
        max_dist = torch.linalg.vector_norm(high - low).item()
        return torch.arange(
            0,
            max_dist + bin_width,
            step=bin_width,
            device=self.device,
            dtype=self.dtype,
        )

    def compute_iq_intermediate(
        self,
        structure: MinStructureTuple,
        compile: bool = True,
        boundaries: Tensor | None = None,
        bin_width: float = 1e-3,
        batch_size: int = 65_536,
    ) -> Tensor:
        """
        Compute the scattering intensity I(Q) for a given structure.

        Parameters:
            structure (StructureTuple): The atomic structure.
            batch_size (int): Number of distance bins to process at once.

        Returns:
            torch.Tensor: The computed I(Q) values.
        """
        # Use the bin edges provided by the user or calculate them.
        # For user-provided bins, a slightly slower binary search is used for
        # non-uniform bins. For equally sized bins, a faster integer division
        # is used.
        if boundaries is None:
            boundaries = self.get_distance_boundaries(structure.xyz, bin_width)
            weight_func = binned_pdist_constant_bin_width
        else:
            weight_func = binned_pdist
        bin_centers = 0.5 / torch.pi * (boundaries[1:] + boundaries[:-1])
        weights = weight_func(
            structure.xyz, isqrt(self.batch_size), boundaries, compile=compile
        )
        iq = torch.zeros_like(self.q.squeeze(-1))
        q = self.q.view(-1, 1)

        # Batch the computation of I(Q) to avoid memory issues for large structures.
        # # Pad to ensure uniform batch sizes for same shape compilation.
        # padding = (-weights.numel()) % batch_size
        # if padding:
        #     zeros = weights.new_zeros(padding)
        #     weights = torch.cat((weights, zeros))
        #     bin_centers = torch.cat((bin_centers, zeros))

        for start in range(0, weights.numel(), batch_size):
            end = start + batch_size
            iq += torch.einsum(
                "r, qr -> q",
                weights[start:end],
                torch.sinc(q * bin_centers[start:end].view(1, -1)),
            )
        return structure.form_avg_sq * iq


# @torch.compile(fullgraph=True)
# def batched_iq_bin(
#     q: Tensor, bin_centers: Tensor, weights: Tensor, out: Tensor
# ) -> None:
#     out += torch.einsum("r, qr -> q", weights, torch.sinc(q * bin_centers))


def _cdist(x: Tensor, y: Tensor):
    return torch.cdist(x, y, p=2.0, compute_mode="use_mm_for_euclid_dist")


_cdist_compiled = torch.compile(_cdist, fullgraph=True, dynamic=True)


@torch.compile(fullgraph=True)
def _batched_iq_diag(qpi: Tensor, xbatch: Tensor, triu: Tensor, out: Tensor):
    distances = _cdist_compiled(xbatch, xbatch)[*triu]
    out += torch.sinc(qpi * distances).sum(dim=-1)


@torch.compile(fullgraph=True)
def _batched_iq_diag_masked(
    qpi: Tensor,
    xbatch: Tensor,
    triu: Tensor,
    invalid_edges: Tensor,
    out: Tensor,
) -> None:
    distances = _cdist_compiled(xbatch, xbatch)[*triu]
    values = torch.sinc(qpi * distances)
    values[:, invalid_edges] = 0.0
    out += values.sum(dim=-1)


@torch.compile(fullgraph=True)
def _batched_iq_offdiag(
    qpi: Tensor, xbatch: Tensor, ybatch: Tensor, out: Tensor
):
    distances = _cdist_compiled(xbatch, ybatch).view(-1)
    out += torch.sinc(qpi * distances).sum(dim=-1)


@torch.compile(fullgraph=True)
def _batched_iq_offdiag_masked(
    qpi: Tensor,
    xbatch: Tensor,
    ybatch: Tensor,
    invalid_y: Tensor,
    out: Tensor,
) -> None:
    distances = _cdist_compiled(xbatch, ybatch)
    values = torch.sinc(qpi.view(-1, 1, 1) * distances.unsqueeze(0))
    values[:, :, invalid_y] = 0.0
    out += values.sum(dim=(1, 2))


def batched_iq(q: Tensor, x: Tensor, batch_size: int):
    assert x.ndim == 2
    n, dim = x.shape
    nbatches = (n + batch_size - 1) // batch_size

    out = torch.zeros_like(q)

    # Make sure all batches have same size
    pad = nbatches * batch_size - n
    triu = torch.triu_indices(
        batch_size, batch_size, offset=1, device=x.device
    )
    if pad:
        x = torch.cat((x, x.new_zeros((pad, dim))), dim=0)
        invalid_cols = (
            torch.arange(batch_size, device=x.device) >= batch_size - pad
        )
        invalid_edges = triu[1] >= batch_size - pad
    assert x.size(0) % batch_size == 0, (x.shape, batch_size)

    qpi = q.view(-1, 1) / torch.pi

    for i, j in combinations_with_replacement(range(nbatches), 2):
        xbatch = x[i * batch_size : (i + 1) * batch_size]
        if i == j:
            if pad and i == nbatches - 1:
                _batched_iq_diag_masked(qpi, xbatch, triu, invalid_edges, out)
            else:
                _batched_iq_diag(qpi, xbatch, triu, out)
        else:
            ybatch = x[j * batch_size : (j + 1) * batch_size]
            if pad and j == nbatches - 1:
                _batched_iq_offdiag_masked(
                    qpi, xbatch, ybatch, invalid_cols, out
                )
            else:
                _batched_iq_offdiag(qpi, xbatch, ybatch, out)
    return out


def binned_pdist(
    x: Tensor,
    batch_size: int,
    boundaries: Tensor,
    compile: bool = True,
):
    assert x.ndim == 2
    assert boundaries.ndim == 1
    n, dim = x.shape
    nbatches = (n + batch_size - 1) // batch_size

    out = x.new_zeros(boundaries.size(0) - 1)

    # Make sure all batches have same size
    pad = nbatches * batch_size - n
    triu = triu_final = torch.triu_indices(
        batch_size, batch_size, offset=1, device=x.device, dtype=torch.int32
    )
    if pad > 0:
        x = torch.cat((x, x.new_zeros((pad, dim))), dim=0)
        triu_final = triu_final[:, triu_final[1] < batch_size - pad]
    assert x.size(0) % batch_size == 0, (x.shape, batch_size)

    tmp_indices = triu.new_zeros(batch_size * batch_size)

    cdist = _cdist_compiled if compile else _cdist

    for i, j in combinations_with_replacement(range(nbatches), 2):
        final = j == nbatches - 1
        xbatch = x[i * batch_size : (i + 1) * batch_size]
        if i == j:
            distances = cdist(xbatch, xbatch)[*(triu_final if final else triu)]
        else:
            ybatch = x[j * batch_size : (j + 1) * batch_size]
            distances = cdist(xbatch, ybatch)
            if final and pad > 0:
                distances = distances[:, :-pad]
            distances = distances.ravel()
        bin_out = tmp_indices[: distances.numel()]
        torch.searchsorted(
            boundaries[1:-1],
            distances,
            side="right",
            out_int32=True,
            out=bin_out,
        )
        out += torch.bincount(bin_out, minlength=out.size(0))
    return out


def binned_pdist_constant_bin_width(
    x: Tensor,
    batch_size: int,
    boundaries: Tensor,
    compile: bool = True,
):
    assert x.ndim == 2
    assert boundaries.ndim == 1
    n, dim = x.shape
    nbatches = (n + batch_size - 1) // batch_size

    out = x.new_zeros(boundaries.size(0) - 1)
    bin_width = boundaries[1] - boundaries[0]

    # Make sure all batches have same size
    pad = nbatches * batch_size - n
    triu = triu_final = torch.triu_indices(
        batch_size, batch_size, offset=1, device=x.device, dtype=torch.int32
    )
    if pad > 0:
        x = torch.cat((x, x.new_zeros((pad, dim))), dim=0)
        triu_final = triu_final[:, triu_final[1] < batch_size - pad]
    assert x.size(0) % batch_size == 0, (x.shape, batch_size)

    cdist = _cdist_compiled if compile else _cdist

    for i, j in combinations_with_replacement(range(nbatches), 2):
        final = j == nbatches - 1
        xbatch = x[i * batch_size : (i + 1) * batch_size]
        if i == j:
            distances = cdist(xbatch, xbatch)[*(triu_final if final else triu)]
        else:
            ybatch = x[j * batch_size : (j + 1) * batch_size]
            distances = cdist(xbatch, ybatch)
            if final and pad > 0:
                distances = distances[:, :-pad]
            distances = distances.ravel()
        bin_out = torch.floor_divide(distances, bin_width).int()
        out += torch.bincount(bin_out, minlength=out.size(0))
    return out


def _apply_qdamp_convolution(
    fq: Tensor, qdamp: Tensor | float, qstep: Tensor | float
) -> Tensor:
    if qdamp == 0.0:
        return fq
    assert fq.ndim == 1, fq.size()
    sigma = qdamp
    dq = qstep

    # Compute kernel half-width in bins and cap to avoid over-padding
    half_width = ceil(5.0 * sigma / dq)
    fq_len = fq.numel()
    if half_width >= fq_len:
        half_width = max(fq_len - 1, 0)
    x = dq * torch.arange(
        -half_width,
        half_width + 1,
        device=fq.device,
        dtype=fq.dtype,
    )
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()

    # Ensure contiguous layout so reshaping is always safe (even for sliced tensors)
    fq_ = fq.contiguous().view(1, 1, -1)
    k_ = kernel.view(1, 1, -1)
    fq_pad = torch.nn.functional.pad(
        fq_, (half_width, half_width), mode="reflect"
    )
    fq_conv = torch.nn.functional.conv1d(fq_pad, k_)
    return fq_conv.view(-1)


def _apply_biso(q: Tensor, iq_int: Tensor, biso: float | Tensor) -> Tensor:
    # Apply Debye-Waller isotropic atomic displacement
    return iq_int * torch.exp(-q.squeeze(-1).pow(2) * biso / (8 * torch.pi**2))


def get_all_from_iq_int(
    q: Tensor,
    iq_int: Tensor,
    biso: float | Tensor,
    size: int,
    form_avg_sq: Tensor,
    r: Tensor,
    qdamp: float,
    xscale: float = 1.0,
    q_ref: Tensor | None = None,
):
    assert q.ndim == 1, q.size()
    assert r.ndim == 1, r.size()

    q_ref = q if q_ref is None else q_ref
    q = q / xscale

    qstep = q[1] - q[0]
    if (isinstance(biso, (int, float)) or biso.numel() == 1) and biso == 0.0:
        iq = iq_int
    else:
        iq = _apply_biso(q.unsqueeze(-1), iq_int, biso)
    sq = iq / form_avg_sq / size
    fq = q * sq
    damp = torch.exp(-(r * qdamp).pow(2) / 2)
    gr = (
        (2 / torch.pi)
        * torch.sum(
            fq.view(1, -1) * torch.sin(q.view(1, -1) * r.view(-1, 1)) * qstep,
            dim=-1,
        )
        * damp
    )
    fq = _apply_qdamp_convolution(fq, qdamp, qstep)
    fq = interpolate(x_old=q, y_old=fq, x_new=q_ref)
    return iq, sq, fq, gr


def compute_fq(
    form_avg_sq: Tensor | np.ndarray,
    size: int,
    q: Tensor | np.ndarray,
    iq_int: Tensor | np.ndarray,
    biso: float | Tensor | np.ndarray,
    qdamp: float,
    xscale: float = 1.0,
    q_ref: Tensor | np.ndarray | None = None,
):
    # xscale transforms interatomic distances r -> r*xscale. If we transform
    # q as q -> q/xscale, we leave the intermediate I(q) (before biso) unchanged.
    # This means, we're evaluating I(q) at the points q'=q/xscale and
    # linearly interpolating F(q) afterwards at the reference points.
    form_avg_sq = (
        torch.from_numpy(form_avg_sq)
        if isinstance(form_avg_sq, np.ndarray)
        else form_avg_sq
    )
    q = torch.from_numpy(q) if isinstance(q, np.ndarray) else q
    iq_int = (
        torch.from_numpy(iq_int) if isinstance(iq_int, np.ndarray) else iq_int
    )
    q_ref = torch.from_numpy(q_ref) if isinstance(q_ref, np.ndarray) else q_ref
    biso = torch.from_numpy(biso) if isinstance(biso, np.ndarray) else biso
    q_ref = q if q_ref is None else q_ref
    q = q / xscale
    qstep = q[1] - q[0]

    if (isinstance(biso, (int, float)) or biso.numel() == 1) and biso == 0.0:
        sq = iq_int / form_avg_sq / size
    else:
        sq = _apply_biso(q.unsqueeze(-1), iq_int / form_avg_sq / size, biso)
    fq = q * sq
    fq = _apply_qdamp_convolution(fq, qdamp, qstep)

    # Interpolate F(q) at the original q-values.
    fq = interpolate(x_old=q, y_old=fq, x_new=q_ref)
    qmin = q[0]
    qmax = q[-1]
    return fq, qmin, qmax


def interpolate(x_old: Tensor, y_old: Tensor, x_new: Tensor):
    """Interpolate linearly between points (x_old, y_old) at new positions x_new.
    It's assumed that x_old is monotonically increasing. If x_new is smaller or
    greater than all points in x_old, the slope between the two outermost points
    is used.
    """
    right_indices = torch.clamp_(
        torch.searchsorted(x_old, x_new), min=1, max=x_old.size(0) - 1
    )
    left_indices = right_indices - 1
    yl = y_old[left_indices]
    yr = y_old[right_indices]
    xl = x_old[left_indices]
    xr = x_old[right_indices]
    slopes = (yr - yl) / (xr - xl)
    y_new = slopes * (x_new - xl) + yl
    return y_new


class SingleIQResult(TypedDict):
    q: Tensor
    iq_int: Tensor
    size: int
    form_avg_sq: Tensor


class BatchIQResult(TypedDict):
    q: Tensor
    iq_int: Tensor
    size: list[int]
    form_avg_sq: Tensor


def compute_iq(
    atoms: Atoms, calculator: SingleDebyeCalculator, **kwargs
) -> SingleIQResult:
    structure = calculator._initialise_structure(atoms)
    return {
        "q": calculator.q.squeeze(),
        "iq_int": calculator.compute_iq_intermediate(structure, **kwargs),
        "size": structure.size,
        "form_avg_sq": structure.form_avg_sq,
    }


def compute_batch_iq(
    images: list[Atoms], calculator: SingleDebyeCalculator, **kwargs
) -> BatchIQResult:
    if not images:
        raise ValueError("At least one structure is required")
    structures = [calculator._initialise_structure(atoms) for atoms in images]
    iq_ints = [
        calculator.compute_iq_intermediate(structure, **kwargs)
        for structure in structures
    ]
    return {
        "q": calculator.q.squeeze(),
        "iq_int": torch.stack(iq_ints),
        "size": [len(atoms) for atoms in images],
        "form_avg_sq": torch.stack(
            [structure.form_avg_sq for structure in structures]
        ),
    }
