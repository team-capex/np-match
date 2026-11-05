# TODO: Use squared distances instead of distances to avoid sqrt?

from math import ceil

import torch
from torch import Tensor

from .utils import check_cell, check_pbc, wrap_positions


def get_cell_corners(cell: Tensor) -> Tensor:
    """Get the Cartesian coordinates of the 8 corners of the unit cell."""
    unit_corners = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=cell.dtype,
        device=cell.device,
    )
    cell_corners = unit_corners @ cell  # shape: (8, 3)
    return cell_corners


def get_cell_normal_vectors(cell: Tensor) -> Tensor:
    """Get the normal vectors of the faces of the unit cell."""
    # The cell vectors are given as rows of the cell tensor: a, b, c.
    # The normal vector rows are ordered as follows: b x c, c x a, a x b
    normal_vectors = torch.linalg.cross(
        cell.roll(-1, dims=0), cell.roll(-2, dims=0), dim=1
    )  # shape: (3, 3)
    normal_vectors /= torch.linalg.vector_norm(
        normal_vectors, dim=1, keepdim=True
    )
    return normal_vectors


class Supercell:
    """A class to do the supercell tiling for neighbour lists."""

    def __init__(
        self,
        cell: Tensor,
        pbc: Tensor,
        cutoff: float,
        minimal_tiling: bool = True,
    ):
        # TODO: Have supercell construction on CPU then move to GPU at the end of __init__
        self._cutoff = float(cutoff)
        if self._cutoff <= 0:
            raise ValueError("Cutoff distance must be positive.")
        check_cell(cell)
        check_pbc(pbc)
        self._cell = cell.clone()
        self._cell_normals = get_cell_normal_vectors(cell)
        self._cell_corners = get_cell_corners(cell)
        self._cell_inverse = torch.linalg.inv(cell)
        self._pbc = pbc.to(dtype=torch.bool, device=cell.device, copy=True)
        self._pbc_list = self._pbc.tolist()
        self._minimal_tiling = bool(minimal_tiling)
        self.build_cell_tiling()

    @property
    def cell(self) -> Tensor:
        """The unit cell tensor with shape (3, 3)."""
        return self._cell

    @property
    def pbc(self) -> Tensor:
        """The periodic boundary conditions tensor with shape (3,)."""
        return self._pbc

    @property
    def cutoff(self) -> float:
        """The cutoff distance for the neighbour list."""
        return self._cutoff

    @property
    def dtype(self) -> torch.dtype:
        """The floating point data type of the cell tensor."""
        return self.cell.dtype

    @property
    def device(self) -> torch.device:
        """The device of the cell tensor."""
        return self.cell.device

    @property
    def cell_normals(self) -> Tensor:
        """The normal vectors of the unit cell faces - shape (3, 3). The rows
        are ordered as the following cross products: b x c, c x a, a x b, where
        a, b, c are the lattice vectors of the unit cell.
        """
        return self._cell_normals

    @property
    def cell_corners(self) -> Tensor:
        """The Cartesian coordinates of the corners of the unit cell - shape
        (8, 3).
        """
        return self._cell_corners

    @property
    def cell_inverse(self) -> Tensor:
        """The inverse of the unit cell tensor - shape (3, 3). This is used to
        convert Cartesian coordinates to unit cell coordinates.
        """
        return self._cell_inverse

    @property
    def cell_shifts(self) -> Tensor:
        """The Cartesian shifts of cells in the supercell - shape
        (num_cells, 3). Note that the central cell [0, 0, 0] is also included
        in this tensor.
        """
        return self._cell_shifts

    @property
    def unit_shifts(self) -> Tensor:
        """The integer shifts of cells in the supercell - shape (num_cells, 3).
        Note that the central cell [0, 0, 0] is also included in this tensor.
        """
        return self._unit_shifts

    @property
    def bbox_min(self) -> Tensor:
        """Minimum Cartesian coordinates of the bounding box - shape (3,)."""
        return self._bbox_min

    @property
    def bbox_max(self) -> Tensor:
        """Maximum Cartesian coordinates of the bounding box - shape (3,)."""
        return self._bbox_max

    @property
    def num_cells(self) -> int:
        """Number of cells within the cutoff."""
        return self.cell_shifts.shape[0]

    @property
    def minimal_tiling(self) -> bool:
        """Whether to use the minimal tiling, i.e., filter out cells that are
        outside the cutoff distance. The alternative is a full tiling of
        all cells within a bounding box defined by the cutoff distance. Minimal
        tiling is mostly useful for large cutoffs compared to cell lengths or
        very skewed cells. It can reduce the number of cells by up to
        (1-pi/6)=48%.
        """
        return self._minimal_tiling

    def build_cell_tiling(self) -> None:
        """Build the cell tiling based on the current cell, pbc, and cutoff."""
        self._bbox_min, self._bbox_max = self.get_bounding_box()
        cell_repetitions = self.get_cell_repetitions()
        unit_shifts = self.get_unit_shifts_from_cell_repetitions(
            cell_repetitions, device=self.device, dtype=self.dtype
        )
        if self.minimal_tiling and unit_shifts.shape[0] > 3 ** sum(
            self._pbc_list
        ):
            # Filter out the cells that are outside the cutoff distance. They
            # may be present in the unit_shifts if the cell is not
            # orthorhombic.
            min_cell_distances = self.get_minimum_distances_to_cells(
                unit_shifts
            )
            unit_shifts = unit_shifts[min_cell_distances <= self.cutoff]
        self._unit_shifts = unit_shifts
        self._cell_shifts = unit_shifts @ self.cell

    def get_cell_repetitions(self) -> list[int]:
        """Get the maximum number of needed cell repetitions along each lattice
        vector to cover the cutoff distance.
        """
        # Project the cell vectors onto the normal vectors to find the distance
        # between the planes of neighbouring cells.
        interplane_distances = torch.linalg.vecdot(
            self.cell, self.cell_normals, dim=1
        )  # shape: (3,)
        return [
            ceil(self.cutoff / interplane_distance) if pbc else 0
            for interplane_distance, pbc in zip(
                interplane_distances.tolist(), self._pbc_list
            )
        ]

    @staticmethod
    def get_unit_shifts_from_cell_repetitions(
        cell_repetitions: list[int],
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Get the integer shifts of all cells from the number of repetitions
        along each lattice vector. The first row is always [0, 0, 0] for the
        shift of the central cell.
        """
        # Fast return for no PBC and the nearest neighbour case
        if cell_repetitions == [0, 0, 0]:
            return torch.zeros(1, 3, dtype=dtype, device=device)
        if cell_repetitions == [1, 1, 1]:
            return torch.tensor(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 0, -1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [0, 1, -1],
                    [0, -1, 0],
                    [0, -1, 1],
                    [0, -1, -1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 0, -1],
                    [1, 1, 0],
                    [1, 1, 1],
                    [1, 1, -1],
                    [1, -1, 0],
                    [1, -1, 1],
                    [1, -1, -1],
                    [-1, 0, 0],
                    [-1, 0, 1],
                    [-1, 0, -1],
                    [-1, 1, 0],
                    [-1, 1, 1],
                    [-1, 1, -1],
                    [-1, -1, 0],
                    [-1, -1, 1],
                    [-1, -1, -1],
                ],
                dtype=dtype,
                device=device,
            )
        shifts_list = [
            list(range(n + 1)) + list(range(-n, 0)) for n in cell_repetitions
        ]
        unit_shifts = torch.cartesian_prod(
            *[torch.tensor(shifts, dtype=dtype) for shifts in shifts_list]
        )  # shape: (num_shifts, 3)
        return unit_shifts.to(device=device)

    def get_bounding_box(self) -> tuple[Tensor, Tensor]:
        """Get the bounding box for the supercell. The bounding box is a
        rectangular cuboid that contains all points within the cutoff distance
        from the unit cell, taking into account the periodic boundary
        conditions.
        """
        cutoff_bounds = torch.tensor(
            [
                [self.cutoff, 0, 0],
                [-self.cutoff, 0, 0],
                [0, self.cutoff, 0],
                [0, -self.cutoff, 0],
                [0, 0, self.cutoff],
                [0, 0, -self.cutoff],
            ],
            dtype=self.dtype,
            device=self.device,
        )
        # Apply periodic boundary conditions
        cutoff_bounds *= self.pbc

        # Bounding box edges
        bbox_min, bbox_max = torch.aminmax(
            (
                self.cell_corners.view(1, 8, 3) + cutoff_bounds.view(6, 1, 3)
            ).view(-1, 3),
            dim=0,
        )  # shape: (3,)
        return bbox_min, bbox_max

    def get_minimum_distances_to_cells(self, unit_shifts: Tensor) -> Tensor:
        """Calculate the minimum distances between the central and translated
        cells. Cells that share at least one vertex have zero distance between
        them.
        """
        # Since the cell tiling is with identical parallelepipeds, the minimum
        # distance between the central cell and a translated cell is the
        # minimum distance between the vertices of the central cell and the
        # faces of the translated cell. This is because the faces of the
        # translated cell are parallel to the faces of the central cell.

        # We want to: For each vertex of the central cell, find the point on
        # each translated cell that is closest to the vertex and calculate the
        # distance to it.

        # We do this by:
        # 1) Find the closest point on the infinite plane that contains the
        # face of the translated cell to the vertex of the central cell.
        # 2) Clamp the infinite plane point to the finite plane (parallelogram)
        # that is the face of the translated cell. This has to be done in unit
        # cell coordinates for the point to remain in the plane of the face.
        # 3) Calculate the distance from the vertex to the clamped point.

        # Cells that are touching the central cell are at a distance of 0.
        not_nn_mask = (unit_shifts.abs() > 1).any(dim=1)
        min_cell_distances = torch.zeros(
            unit_shifts.size(0), dtype=self.dtype, device=self.device
        )
        if not_nn_mask.sum() == 0:
            return min_cell_distances
        unit_shifts_next = unit_shifts[not_nn_mask].int()

        max_shift = unit_shifts_next.abs().max().item()
        # Create a tensor of shifts for the infinite planes of the translated
        # cells. The order is 0, 1, ..., max_shift, -max_shift, ..., -1. This
        # is done to be able to easily index the planes later when clamping.
        arange = torch.cat(
            (
                torch.arange(
                    0, max_shift + 1, dtype=self.dtype, device=self.device
                ),
                torch.arange(
                    -max_shift, 0, dtype=self.dtype, device=self.device
                ),
            )
        )
        plane_shifts = torch.stack(3 * [arange], dim=1)  # (num_planes, abc)
        inplane_points = plane_shifts.unsqueeze(-1) * self.cell.unsqueeze(
            0
        )  # (num_planes, abc, xyz)

        # Length of line segments connecting corners of the central cell to the
        # infinite planes of the boundaries of the tiling cells.
        # It's the projection of the line segment onto the normal vector of the
        # plane.
        length_inf = torch.linalg.vecdot(
            (
                inplane_points.view(-1, 1, 3, 3)
                - self.cell_corners.view(1, 8, 1, 3)
            ),
            self.cell_normals.view(1, 1, 3, 3),
            dim=-1,
        )  # (num_planes, corners, n)

        # Cartesian coordianates of the intersection points of the line segments
        # with the infinite planes.
        intersection_cartesian = self.cell_corners.view(
            1, 8, 1, 3
        ) + length_inf.unsqueeze(-1) * self.cell_normals.view(
            1, 1, 3, 3
        )  # (num_planes, corners, n, xyz)

        # The points in cell coordinates
        # (num_planes, corners, n, abc)
        intersection_cell = intersection_cartesian @ self.cell_inverse
        # intersection_cell = torch.linalg.solve(
        #     self.cell, intersection_cartesian, left=False
        # )  # (num_planes, corners, n, abc)

        # Minimum and maximum cell coordinates of each cell in the tiling.
        bounds = torch.stack(
            (unit_shifts_next, unit_shifts_next + 1), dim=1
        )  # (num_cells, 2, n)

        # Clamp the intersection points to the bounds of the neighbouring cells.
        # For all corners, neighbouring cells, and faces, clamp the points to the
        # bounds of the neighbouring cells.
        clamped_points = torch.clamp(
            intersection_cell[
                bounds, :, torch.arange(3, device=self.device), :
            ],
            min=bounds[:, 0, :].view(-1, 1, 1, 1, 3),
            max=bounds[:, 1, :].view(-1, 1, 1, 1, 3),
        )  # (num_cells, 2, n, corners, abc)

        # Calculate the distance from the corners of the central cell to the
        # clamped points on the neighbouring cells.
        distances = torch.linalg.vector_norm(
            clamped_points @ self.cell - self.cell_corners.view(1, 1, 1, 8, 3),
            dim=-1,
        )  # (num_cells, 2, n, corners)
        min_cell_distances[not_nn_mask] = (
            distances.view(-1, 2 * 3 * 8).min(dim=1).values
        )  # (num_cells,)
        return min_cell_distances

    def get_minimum_distances_to_positions(self, positions: Tensor) -> Tensor:
        """Calculate the minimum distances between the cell boundary and the
        positions.
        """
        # cell: (3, 3) or (abc, xyz)
        # positions: (N, 3) or (N, xyz)

        # Project the positions onto the normal vectors to find the distance
        # from the positions to the infinite planes of the extended cell faces.
        # We do this for the 3 faces that contain the origin of the cell and the
        # 3 faces that contain the opposite corner of the cell.
        length_inf0 = (
            positions @ self.cell_normals.T
        )  # (N, xyz) @ (xyz, n) -> (N, n)
        length_inf1 = (positions - self.cell.sum(dim=0)) @ self.cell_normals.T

        # TODO: Do more inplace operations. Maybe some .addcmul()
        # Calculate intersection points with infinite planes
        intersection_inf0 = (
            positions.unsqueeze(1)
            - length_inf0.unsqueeze(-1) * self.cell_normals
        )  # (N, n, xyz)
        intersection_inf1 = (
            positions.unsqueeze(1)
            - length_inf1.unsqueeze(-1) * self.cell_normals
        )  # (N, n, xyz)

        # TODO: Don't cat but create an empty tensor and fill it in place with
        # torch.matmul(out=...)
        intersection_inf = torch.cat(
            (intersection_inf0, intersection_inf1), dim=1
        )  # (N, 2*n, xyz)

        # Convert to cell coordinates
        intersection_inf_cell = (
            intersection_inf @ self.cell_inverse
        )  # (N, 2*n, abc)
        # intersection_inf_cell = torch.linalg.solve(
        #     self.cell, intersection_inf, left=False
        # )  # (N, 2*n, abc)

        # TODO: Clamp inplace
        # Clamp the intersection points to the finite planes of the cell faces.
        # intersection_inf_cell.clamp_
        clamped_points_cell = torch.clamp(
            intersection_inf_cell,
            min=0.0,
            max=1.0,
        )  # (N, 2*n, abc)

        # Convert back to Cartesian coordinates and calculate the distance
        # from the original positions to the clamped points which are the closest
        # points on the cell faces.
        # TODO: inplace sub_
        clamped_points_cartesian = (
            clamped_points_cell @ self.cell
        )  # (N, 2*n, xyz)
        min_distances = torch.linalg.vector_norm(
            clamped_points_cartesian - positions.unsqueeze(1), dim=-1
        )  # (N, 2*n)
        return min_distances.min(dim=1).values  # (N,)

    # TODO: should filter_by_cell_distance default to True? Performance impact?
    def build_supercell(
        self,
        positions: Tensor,
        wrap: bool = True,
        filter_by_cell_distance: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Repeat the unit cell positions in all neighbouring cells to construct a
        supercell. The atoms are filtered to be within the bounding box defined
        by the cutoff distance.

        Parameters
        ----------
        positions : torch.Tensor
            Tensor of shape (n_atoms, 3) containing the atomic positions of the
            unit cell.
        wrap : bool, optional
            Whether to wrap the positions back to the unit cell before
            constructing the supercell. This should only be set to False if the
            positions are already wrapped. Default is True.

        Returns
        -------
        super_positions : torch.Tensor
            Tensor of shape (n_super_atoms, 3) containing the positions of
            atoms in the supercell, filtered to be within the bounding box.
        super_indices : torch.Tensor
            Tensor of shape (n_super_atoms,) containing the indices of the
            original basis atoms for each atom in the supercell.
        super_unit_shifts : torch.Tensor
            Tensor of shape (n_super_atoms, 3) containing the integer cell
            shift vectors for each atom in the supercell.
        super_cell_shifts : torch.Tensor
            Tensor of shape (n_super_atoms, 3) containing the Cartesian cell
            shift vectors for each atom in the supercell.

        Notes
        -----
        The first n_atoms in the supercell correspond to the positions in the
        central unit cell in the input order, while the rest are the repeated
        positions in the neighbouring cells.
        """
        if positions.device != self.device:
            raise ValueError(
                "Positions must be on the same device as the cell tensor."
            )
        if wrap:
            positions, _, _ = wrap_positions(positions, self.cell, self.pbc)
        nbasis = positions.shape[0]
        if not any(self._pbc_list):
            # If there are no periodic boundary conditions, the supercell is
            # just the unit cell.
            return (
                positions.clone(),
                torch.arange(nbasis, dtype=torch.int64, device=self.device),
                torch.zeros(nbasis, 3, dtype=self.dtype, device=self.device),
            )

        # First, repeat unit cell positions in all neighbouring cells.
        super_positions = positions.view(1, -1, 3) + self.cell_shifts.view(
            -1, 1, 3
        )  # shape: (num_cells, nbasis, 3)

        # Second, apply filters to remove atoms outside the cutoff.
        # These should scale at most linearly with the number of atoms, so they
        # are cheaper than the neighbour list construction.
        # Filter 1: Remove atoms that are not inside the bounding box.
        cell_indices, super_indices = (
            (
                (self.bbox_min <= super_positions)
                & (super_positions <= self.bbox_max)
            )
            .all(dim=-1)
            .nonzero(as_tuple=True)
        )
        super_positions = super_positions[cell_indices, super_indices]

        # Filter 2: Remove atoms that are further than the cutoff distance
        # from the central cell.
        if filter_by_cell_distance:
            min_distances = torch.zeros(
                super_positions.shape[0],
                dtype=self.dtype,
                device=self.device,
            )
            min_distances[nbasis:] = self.get_minimum_distances_to_positions(
                super_positions[nbasis:]
            )
            indices_inside_cutoff = (min_distances <= self.cutoff).nonzero(
                as_tuple=True
            )
            super_positions = super_positions[indices_inside_cutoff]
            super_indices = super_indices[indices_inside_cutoff]
            cell_indices = cell_indices[indices_inside_cutoff]
        super_unit_shifts = self.unit_shifts[cell_indices]

        return (
            super_positions,
            super_indices,
            super_unit_shifts,
        )
