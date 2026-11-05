from abc import ABC, abstractmethod

import torch
from torch import Tensor

from .supercell import Supercell
from .utils import check_positions, wrap_positions


class NeighbourListBuilder(ABC):
    def __init__(
        self,
        positions: Tensor,
        supercell: Supercell,
        indices: Tensor | None = None,
        filter_by_cell_distance: bool = False,
    ):
        if not positions.device == supercell.device:
            raise ValueError(
                "Positions, cell, and pbc tensors must be on the same device."
            )
        if not positions.dtype == supercell.dtype:
            raise ValueError(
                "Positions and cell tensors must have the same dtype."
            )
        self.supercell = supercell

        # Check and wrap positions in the unit cell
        check_positions(positions)
        self._positions, self._wrap_unit_shifts, _ = wrap_positions(
            positions, supercell.cell, supercell.pbc, supercell.cell_inverse
        )
        self._positions_unwrapped = positions.clone()

        if indices is None:
            self._indices = torch.arange(
                self.nbasis, dtype=torch.int64, device=self.device
            )
        else:
            if not indices.device == self.device:
                raise ValueError(
                    "Indices tensor must be on the same device as positions."
                )
            if not indices.dtype == torch.int64:
                raise ValueError("Indices tensor must have dtype int64.")
            if indices.shape[0] != self.nbasis:
                raise ValueError(
                    "Indices tensor must have the same number of elements as "
                    "the number of atoms in the unit cell."
                )
            self._indices = indices

        sender, receiver, unit_shifts, distances = self.build(
            filter_by_cell_distance=filter_by_cell_distance
        )
        # Convert unit shifts to match the original positions before wrapping
        unit_shifts += self._wrap_unit_shifts[sender]
        unit_shifts -= self._wrap_unit_shifts[receiver]

        # self._edge_indices = edge_indices
        self._sender = sender
        self._receiver = receiver
        self._unit_shifts = unit_shifts
        self._distances = distances

    @property
    def device(self) -> torch.device:
        """The device on which the tensors are stored."""
        return self.positions.device

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the tensors."""
        return self.positions.dtype

    @property
    def positions(self) -> Tensor:
        """The atomic positions of the unit cell. The positions are wrapped
        within the unit cell.
        """
        return self._positions

    def get_original_positions(self) -> Tensor:
        """
        Get the original atomic positions before wrapping.

        Returns
        -------
        torch.Tensor
            The original atomic positions with shape (N, 3).
        """
        return self._positions_unwrapped

    @property
    def nbasis(self) -> int:
        """The number of atoms in the unit cell."""
        return self.positions.shape[0]

    @property
    def indices(self) -> Tensor:
        """The indices of the atoms in the unit cell."""
        return self._indices

    @property
    def cell(self) -> Tensor:
        """The unit cell tensor with shape (3, 3)."""
        return self.supercell.cell

    @property
    def pbc(self) -> Tensor:
        """The periodic boundary conditions tensor with shape (3,)."""
        return self.supercell.pbc

    @property
    def cutoff(self) -> float:
        """The cutoff distance for the neighbour list."""
        return self.supercell.cutoff

    @property
    def sender(self) -> Tensor:
        """The sender indices of the neighbour list. This is a tensor of shape
        (num_edges,), where each element contains the index of the atom in the
        unit cell that is sending the interaction.

        Returns
        -------
        torch.Tensor
            The sender indices of the neighbour list with shape (num_edges,).
        """
        return self._sender

    @property
    def receiver(self) -> Tensor:
        """The receiver indices of the neighbour list. This is a tensor of shape
        (num_edges,), where each element contains the index of the atom in the
        supercell that is receiving the interaction.

        Returns
        -------
        torch.Tensor
            The receiver indices of the neighbour list with shape (num_edges,).
        """
        return self._receiver

    @property
    def unit_shifts(self) -> Tensor:
        """The unit shifts of the neighbour list. This is a tensor of shape
        (num_edges, 3), where each row contains the unit shift vector for the
        corresponding edge in the neighbour list.

        Returns
        -------
        torch.Tensor
            The unit shifts of the neighbour list with shape (num_edges, 3).
        """
        return self._unit_shifts

    @property
    def distances(self) -> Tensor:
        """The distances between the atoms in the unit cell and their
        corresponding neighbours in the supercell. This is a tensor of shape
        (num_edges,), where each element contains the distance for the
        corresponding edge in the neighbour list.

        Returns
        -------
        torch.Tensor
            The neighbour distances with shape (num_edges,).
        """
        return self._distances

    @abstractmethod
    def build(
        self, filter_by_cell_distance: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Abstract method to build the neighbor list. This method must be
        implemented by subclasses.

        Returns
        -------
        edge_indices : torch.Tensor
            The edge indices of the neighbour list with shape (2, num_edges).
        unit_shifts : torch.Tensor
            The unit shifts for each edge with shape (num_edges, 3).
        distances : torch.Tensor
            The distances for each edge with shape (num_edges,).

        Notes
        -----
        The unit shifts should be calculated based on the wrapped positions
        stored in the `positions` property. After building the neighbour list,
        the unit shifts are adjusted to match the original, unwrapped positions
        in the __init__ method.
        """
        ...
