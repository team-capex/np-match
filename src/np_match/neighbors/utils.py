from typing import Literal

import torch
from torch import Tensor


def wrap_positions(
    positions: Tensor,
    cell: Tensor,
    pbc: Tensor,
    cell_inv: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Wrap positions back to the unit cell."""
    # solve for cell_coordinates: cell_coordinates @ cell = pos_to_wrap
    # to find the linear combinations of unit cell vectors that matches the
    # positions.
    # Use the inverse of the cell matrix if provided, otherwise solve the
    # linear system directly.
    if cell_inv is None:
        # This method is more numerically stable than using the inverse
        # of the cell matrix but also slower.
        unit_shifts = torch.linalg.solve(cell, positions, left=False)  # (N, 3)
    else:
        unit_shifts = positions @ cell_inv  # (N, 3)

    # Floor the result to get the unit shifts.
    unit_shifts.floor_()

    # Only wrap periodic dimensions.
    unit_shifts *= pbc.bool()

    # Move atoms a whole number of unit cell vectors to map them back to the
    # unit cell.
    cell_shifts = unit_shifts @ cell
    pos_wrapped = positions - cell_shifts
    return pos_wrapped, unit_shifts, cell_shifts


def check_cell(cell: Tensor) -> None:
    """Check if the cell is a valid 3x3 matrix."""
    msg = "Cell must be a tensor of shape (3, 3) with float dtype."
    if not isinstance(cell, Tensor):
        raise TypeError(msg)
    if cell.shape != (3, 3):
        raise ValueError(msg)
    if cell.dtype not in (torch.float32, torch.float64):
        raise ValueError(msg)


def check_pbc(pbc: Tensor) -> None:
    """Check if the periodic boundary conditions are a boolean tensor."""
    msg = "PBC must be a boolean tensor of shape (3,)."
    if not isinstance(pbc, Tensor):
        raise TypeError(msg)
    if pbc.shape != (3,):
        raise ValueError(msg)
    if pbc.dtype != torch.bool:
        raise ValueError(msg)


def check_positions(positions: Tensor) -> None:
    """Check if the positions are a valid tensor."""
    msg = "Positions must be a tensor of shape (N, 3) with float dtype."
    if not isinstance(positions, Tensor):
        raise TypeError(msg)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(msg)
    if positions.dtype not in (torch.float32, torch.float64):
        raise ValueError(msg)


# Use torch.unique(dim=1) instead...
def _sort_helper(tensor: Tensor) -> Tensor:
    """Recursively sort the columns of a 2D tensor by multiple keys/rows."""
    # Stop recursion if tensor has only one row.
    if tensor.size(0) == 1:
        return tensor
    # Sort the first row and get the sorted indices.
    sorted_row, sort_indices = torch.sort(tensor[0])
    print(tensor, sort_indices)
    # Order the columns in the next rows in accordance with the sorted first
    # row.
    next_rows = tensor[1:, sort_indices]
    # Find the sizes of each block of identical elements in the sorted row.
    block_sizes: list[int] = (
        sorted_row.diff().nonzero().squeeze() + 1
    ).tolist()
    # torch.Tensor.tolist() returns a number if there is only one element
    if not isinstance(block_sizes, list):
        block_sizes = [block_sizes]
    # Last block size is the remaining elements
    block_sizes.append(sorted_row.size(0) - sum(block_sizes))
    # Split the next rows into blocks of identical first row elements.
    next_rows_blocks = list(torch.split(next_rows, block_sizes, dim=1))
    # Recursively sort each block.
    for i, block in enumerate(next_rows_blocks):
        next_rows_blocks[i] = _sort_helper(block)
    sorted_next_rows = torch.cat(next_rows_blocks, dim=1)
    return torch.cat((sorted_row.unsqueeze(0), sorted_next_rows), dim=0)


def sort_edges(
    sender: Tensor, receiver: Tensor, unit_shifts: Tensor, *tensors: Tensor
) -> tuple[Tensor, Tensor, Tensor, list[Tensor], Tensor]:
    """
    Sort edges by sender index, receiver index, and unit shifts. This should
    uniquely identify each edge in a neighbour list.
    """
    sorted_tensor, inverse_indices, edge_counts = torch.unique(
        torch.cat(
            (sender.view(1, -1), receiver.view(1, -1), unit_shifts.T), dim=0
        ),
        dim=1,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    sorted_sender, sorted_receiver = sorted_tensor[:2, :].to(
        torch.int64
    )  # shape (2, num_edges)
    sorted_unit_shifts = sorted_tensor[2:, :].T  # shape (num_edges, 3)
    sort_indices = torch.argsort(inverse_indices)  # shape (num_edges,)
    sorted_tensors = [t[sort_indices] for t in tensors]
    return (
        sorted_sender,
        sorted_receiver,
        sorted_unit_shifts,
        sorted_tensors,
        edge_counts,
    )


def ase_atoms_to_torch(
    atoms: "ase.Atoms",
    device: str = "cpu",
    dtype: Literal["float32", "float64"] = "float32",
    copy: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Extract positions, cell, and periodic boundary conditions (PBC) from an
    Atomic Simulation Environment (ASE) Atoms object and convert them to
    PyTorch tensors.
    """
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be 'float32' or 'float64'")
    t_dtype = torch.float32 if dtype == "float32" else torch.float64
    positions = torch.from_numpy(atoms.positions).to(
        device=device, dtype=t_dtype, copy=copy
    )
    cell = torch.from_numpy(atoms.cell.array).to(
        device=device, dtype=t_dtype, copy=copy
    )
    pbc = torch.from_numpy(atoms.pbc).to(
        device=device, dtype=torch.bool, copy=copy
    )
    return positions, cell, pbc
