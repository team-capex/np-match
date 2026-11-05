import torch
from torch import Tensor

from .builder_base import NeighbourListBuilder
from .supercell import Supercell


@torch.compiler.disable
class NeighbourList:
    def __init__(
        self,
        builder_type: type[NeighbourListBuilder],
        cutoff: float,
        skin: float = 0.0,
        minimal_tiling: bool = True,
        filter_by_cell_distance: bool = False,
        filter_edges: bool = True,
        **builder_kwargs,
    ):
        if cutoff <= 0:
            raise ValueError("Cutoff distance must be positive.")
        if skin < 0:
            raise ValueError("Skin depth must be non-negative.")
        self.builder_type = builder_type
        self.skin = float(skin)
        self.cutoff = float(cutoff)
        self.minimal_tiling = bool(minimal_tiling)
        self.filter_by_cell_distance = bool(filter_by_cell_distance)
        self.filter_edges = bool(filter_edges)
        self.builder_kwargs = builder_kwargs
        self._nupdates = 0

    @property
    def nupdates(self) -> int:
        """The number of times the neighbour list has been updated."""
        return self._nupdates

    @property
    def extended_cutoff(self) -> float:
        """The extended cutoff distance, which is the cutoff plus the skin
        depth."""
        return self.cutoff + self.skin

    @property
    def builder(self) -> NeighbourListBuilder:
        """The builder used to create the neighbour list."""
        if not hasattr(self, "_builder"):
            raise AttributeError(
                "Neighbour list has not been built yet. Call update() first."
            )
        return self._builder

    @property
    def positions(self) -> Tensor:
        return self._positions

    @property
    def cell(self) -> Tensor:
        return self._cell

    @property
    def pbc(self) -> Tensor:
        return self._pbc

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
        return self.builder.sender

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
        return self.builder.receiver

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
        return self.builder.unit_shifts

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

    def build(
        self,
        positions: Tensor,
        cell: Tensor | None = None,
        pbc: Tensor | None = None,
        supercell: Supercell | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Build the neighbour list with the given positions, cell, and PBCs.
        If a supercell is provided, it is used to build the neighbour list.
        Otherwise, a new supercell is created from the cell and PBCs.
        """
        if not (
            (cell is not None and pbc is not None and supercell is None)
            or (cell is None and pbc is None and supercell is not None)
        ):
            raise ValueError(
                "Either cell and pbc must be provided or a supercell must be "
                "provided."
            )
        if supercell is None:
            supercell = Supercell(
                cell=cell,  # type: ignore[arg-type]
                pbc=pbc,  # type: ignore[arg-type]
                cutoff=self.extended_cutoff,
                minimal_tiling=self.minimal_tiling,
            )
        self._builder = self.builder_type(
            positions,
            supercell,
            filter_by_cell_distance=self.filter_by_cell_distance,
            **self.builder_kwargs,
        )
        self._positions = positions
        self._cell = supercell.cell
        self._pbc = supercell.pbc
        self._distances = self.builder.distances
        self._nupdates += 1

        # For updating the distances
        self._tmp1 = positions.new_empty(self.unit_shifts.shape)
        self._tmp2 = positions.new_empty(self.unit_shifts.shape)
        return self.sender, self.receiver, self.unit_shifts, self.distances

    def update(
        self,
        positions: Tensor | None = None,
        cell: Tensor | None = None,
        pbc: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Update the neighbour list with new positions, cell, and PBCs.
        If no arguments are provided, the neighbour list is not updated.
        If this is the first update, the positions, cell, and PBCs must be
        provided.
        """
        sender, receiver, unit_shifts, distances = self._update_helper(
            positions=positions, cell=cell, pbc=pbc
        )
        if self.skin == 0.0 or not self.filter_edges:
            return sender, receiver, unit_shifts, distances

        indices = (distances <= self.cutoff).nonzero(as_tuple=True)[0]
        return (
            sender[indices],
            receiver[indices],
            unit_shifts[indices],
            distances[indices],
        )

    def _update_helper(
        self,
        positions: Tensor | None = None,
        cell: Tensor | None = None,
        pbc: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Update the neighbour list with new positions, cell, and PBCs.
        If no arguments are provided, the neighbour list is not updated.
        If this is the first update, the positions, cell, and PBCs must be
        provided.
        """
        if self.nupdates == 0:
            if positions is None or cell is None or pbc is None:
                raise ValueError(
                    "Initial positions, cell, and pbc must be provided for "
                    "the first update call."
                )
            return self.build(positions=positions, cell=cell, pbc=pbc)

        if positions is None and cell is None and pbc is None:
            # If no inputs are provided, we do not update the neighbour list.
            return self.sender, self.receiver, self.unit_shifts, self.distances

        max_dist_change = 0.0
        new_supercell = None
        needs_supercell_update = False
        if pbc is None or torch.equal(self.pbc, pbc):
            pbc = self.pbc
        else:
            max_dist_change = self.skin  # Force rebuild
            needs_supercell_update = True

        # Only check displacements if PBCs are unchanged, otherwise we always
        # rebuild.
        if positions is None:
            positions = self.positions
        else:
            if max_dist_change < self.skin:
                # Force rebuild if atoms are added or removed.
                if positions.shape != self.builder.positions.shape:
                    max_dist_change = self.skin
                else:
                    max_dist_change += (
                        2
                        * torch.linalg.vector_norm(
                            positions - self.builder.get_original_positions(),
                            dim=1,
                        ).max()
                    )

        if cell is None or torch.equal(self.cell, cell):
            cell = self.cell
        else:
            needs_supercell_update = True

            # Add cell change contribution to the maximum distance change.
            if max_dist_change < self.skin:
                new_supercell = Supercell(
                    cell=cell,
                    pbc=pbc,
                    cutoff=self.extended_cutoff,
                    minimal_tiling=self.minimal_tiling,
                )
                cell_diff = cell - self.builder.cell
                new_max_shift = torch.linalg.vector_norm(
                    new_supercell.unit_shifts @ cell_diff,
                    dim=1,
                ).max()
                old_max_shift = torch.linalg.vector_norm(
                    self.builder.supercell.unit_shifts @ cell_diff,
                    dim=1,
                ).max()
                max_dist_change += max(new_max_shift, old_max_shift)

        # Update neighbour list if the maximum distance change is larger than
        # the skin depth.
        if max_dist_change >= self.skin:
            if needs_supercell_update:
                # If the PBC or cell have changed compared to the current
                # supercell, we update the supercell for the new neighbour list
                supercell = (
                    Supercell(
                        cell=cell,
                        pbc=pbc,
                        cutoff=self.extended_cutoff,
                        minimal_tiling=self.minimal_tiling,
                    )
                    if new_supercell is None
                    else new_supercell
                )
            else:
                supercell = self.builder.supercell
            return self.build(positions=positions, supercell=supercell)

        self._positions = positions
        self._cell = cell
        self._pbc = pbc

        self._update_distances()
        return self.sender, self.receiver, self.unit_shifts, self.distances

    def _update_distances(self) -> None:
        """Update the distances based on the current positions and cell."""
        # Below is an in-place version with reuseable tensors of the following
        # code:
        # self._distances = (
        #     positions[self.edge_indices[1]]
        #     - positions[self.edge_indices[0]]
        #     + self.unit_shifts @ cell
        # ).norm(dim=1)

        # Save positions of sender and receiver atoms in temporary tensors
        torch.index_select(self.positions, 0, self.receiver, out=self._tmp1)
        torch.index_select(self.positions, 0, self.sender, out=self._tmp2)

        # Compute the displacements between the sender and receiver atoms
        displacements = self._tmp1
        displacements -= self._tmp2
        displacements.addmm_(self.unit_shifts, self.cell)

        # Compute the distances and store them in the distances tensor
        torch.linalg.vector_norm(displacements, dim=1, out=self.distances)
