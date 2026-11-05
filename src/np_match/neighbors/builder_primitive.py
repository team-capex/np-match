import torch
from torch import Tensor

from .builder_base import NeighbourListBuilder


def calc_squared_dists(
    positions: Tensor,
    super_positions: Tensor,
) -> Tensor:
    # (a-b)^2 = a^2 + b^2 - 2*a.dot(b)
    out = positions.square().sum(dim=-1).view(
        -1, 1
    ) + super_positions.square().sum(dim=-1).view(1, -1)
    out.addmm_(positions, super_positions.T, alpha=-2.0)
    out.clamp_min_(0.0)  # Avoid negative distances due to numerical errors
    return out


# TODO: Move filter_by_cell_distance to the Supercell __init__ method
class PrimitiveBuilder(NeighbourListBuilder):
    def build(
        self,
        filter_by_cell_distance: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Build the neighbour list. This method finds all neighbouring atoms
        within the cutoff distance by tiling unit cells and checking distances
        between all atoms in the unit cell and all atoms in the tiled
        supercell. It scales quadratically with the number of atoms.
        """
        # Tile the space around the unit cell to create a supercell with all
        # atoms within a bounding box. We can set wrap=False to avoid wrapping
        # the positions again, as this has already been done in the
        # positions setter.
        (
            super_positions,
            super_indices,
            super_unit_shifts,
        ) = self.supercell.build_supercell(
            self.positions,
            wrap=False,
            filter_by_cell_distance=filter_by_cell_distance,
        )

        # Find all neighbouring atoms within the cutoff distance.
        # distances = torch.cdist(self.positions, super_positions)
        distances = calc_squared_dists(self.positions, super_positions)
        distances.fill_diagonal_(float("inf"))  # Ignore self-interactions
        sender_indices, receiver_super_indices = torch.nonzero(
            distances <= self.cutoff**2, as_tuple=True
        )
        neighbour_distances = distances[sender_indices, receiver_super_indices]
        self.num_calculated_distances = distances.numel()
        self.acceptance_ratio = sender_indices.size(0) / distances.numel()

        receiver_indices = super_indices[receiver_super_indices]

        # Get the corresponding cell shifts and distances for the edges
        unit_shifts = super_unit_shifts[receiver_super_indices]
        return (
            sender_indices,
            receiver_indices,
            unit_shifts,
            neighbour_distances.sqrt_(),
        )
