import pytest
import torch
from ase import Atoms
from ase.cluster import Icosahedron
from debyecalculator import DebyeCalculator
from np_match.debye.debye_single import (
    BinnedSingleDebyeCalculator,
    CdistSingleDebyeCalculator,
    SingleDebyeCalculator,
    batched_iq,
    binned_pdist_constant_bin_width,
    compute_iq,
    get_all_from_iq_int,
)

_DTYPES = (torch.float32, torch.float64)
_DEVICES = ("cpu",) + (("cuda",) if torch.cuda.is_available() else ())
_OUTPUT_TOLERANCES = {
    (CdistSingleDebyeCalculator, torch.float32): (
        (2e-4, 0.3),  # I(Q)
        (2e-4, 2e-4),  # S(Q)
        (2e-4, 2e-3),  # F(Q)
        (2e-4, 2e-3),  # G(r)
        1e-5,  # Rwp for F(Q)
    ),
    (CdistSingleDebyeCalculator, torch.float64): (
        (2e-3, 5e-2),
        (1e-5, 1e-6),
        (1e-5, 1e-5),
        (5e-5, 1e-5),
        2e-6,
    ),
    (BinnedSingleDebyeCalculator, torch.float32): (
        (2e-3, 160.0),
        (2e-2, 7e-4),
        (2e-2, 0.012),
        (2e-2, 0.017),
        1e-3,
    ),
    (BinnedSingleDebyeCalculator, torch.float64): (
        (2e-3, 160.0),
        (2e-2, 7e-4),
        (2e-2, 0.012),
        (2e-2, 0.017),
        1e-3,
    ),
}


@pytest.mark.parametrize("dtype", _DTYPES, ids=["float32", "float64"])
@pytest.mark.parametrize("size", [3, 5, 8, 9, 13])
@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICES)
def test_batched_iq_matches_direct_pair_sum(
    dtype: torch.dtype, size: int, device: str
) -> None:
    batch_size = 4
    q = torch.tensor(
        [1e-5, 0.2, 0.7, 1.3, 25.7, 100.3], dtype=dtype, device=device
    )
    positions = torch.randn(size, 3, dtype=dtype, device=device)

    actual = batched_iq(q, positions, batch_size)
    distances = torch.pdist(positions, p=2.0)
    expected = torch.sinc(q[:, None] * distances[None, :] / torch.pi).sum(
        dim=-1
    )

    tolerance = 1e-4 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=tolerance)


@pytest.mark.parametrize("bin_batch_size", [1, 3, 32])
@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICES)
@pytest.mark.parametrize("dtype", _DTYPES, ids=["float32", "float64"])
@pytest.mark.parametrize("compile", [False, True])
def test_binned_iq_matches_unbatched_calculation(
    bin_batch_size: int, device: str, dtype: torch.dtype, compile: bool
) -> None:
    calculator = BinnedSingleDebyeCalculator(
        qmin=0.2,
        qmax=3.6,
        qstep=0.01,
        device=device,
        batch_size=4,
        dtype=dtype,
    )
    # A non-zero bin width approximates the direct Debye sum, but only the
    # zero-width limit is exact.
    structure = calculator._initialise_structure(
        Atoms(
            "Au3",
            positions=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        )
    )
    boundaries = calculator.get_distance_boundaries(structure.xyz, 1e-4)

    actual = calculator.compute_iq_intermediate(
        structure,
        compile=compile,
        batch_size=bin_batch_size,
        bin_width=1e-4,
    )
    weights = binned_pdist_constant_bin_width(
        structure.xyz, 2, boundaries, compile=False
    )
    bin_centers = 0.5 / torch.pi * (boundaries[1:] + boundaries[:-1])
    expected = structure.form_avg_sq * torch.einsum(
        "r, qr -> q",
        weights,
        torch.sinc(calculator.q * bin_centers.view(1, -1)),
    )

    # Finite binning is an approximation; exact equality is only recovered in
    # the zero-width limit.
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=0.2)


@pytest.mark.parametrize(
    "calculator_type",
    [CdistSingleDebyeCalculator, BinnedSingleDebyeCalculator],
    ids=["cdist", "binned"],
)
@pytest.mark.parametrize(
    "dtype",
    _DTYPES,
    ids=lambda dtype: str(dtype).removeprefix("torch."),
)
@pytest.mark.parametrize("device", _DEVICES, ids=_DEVICES)
def test_single_debye_calculators_match_original(
    calculator_type: type[SingleDebyeCalculator],
    dtype: torch.dtype,
    device: str,
) -> None:
    atoms = Icosahedron("Au", 3)
    options = {
        "qmin": 1.0,
        "qmax": 20.0,
        "qstep": 0.01,
        "qdamp": 0.0,
        "rmin": 0.0,
        "rmax": 10.0,
        "rstep": 0.05,
        "biso": 0.5,
        "device": device,
        "dtype": dtype,
    }
    output_tolerances = _OUTPUT_TOLERANCES[(calculator_type, dtype)]
    calculator = calculator_type(**options)
    kwargs = (
        {"bin_width": 1e-4}
        if calculator_type is BinnedSingleDebyeCalculator
        else {}
    )
    iq_data = compute_iq(atoms, calculator, **kwargs)
    iq, sq, fq, gr = get_all_from_iq_int(
        q=iq_data["q"],
        iq_int=iq_data["iq_int"],
        biso=0.5,
        size=iq_data["size"],
        form_avg_sq=iq_data["form_avg_sq"],
        r=calculator.r.squeeze(),
        qdamp=0.0,
    )

    original = DebyeCalculator(**options)
    # The new calculators return pair scattering without self-scattering.
    original_iq = original.iq(
        atoms, keep_on_device=True, include_self_scattering=False
    )
    original_sq = original.sq(atoms, keep_on_device=True)
    original_fq = original.fq(atoms, keep_on_device=True)
    original_gr = original.gr(atoms, keep_on_device=True)

    torch.testing.assert_close(iq_data["q"], original_iq.q, rtol=0, atol=2e-6)
    torch.testing.assert_close(
        calculator.r.squeeze(), original_gr.r, rtol=0, atol=2e-6
    )
    for name, actual, expected, (rtol, atol) in (
        ("I(Q)", iq, original_iq.i, output_tolerances[0]),
        ("S(Q)", sq, original_sq.s, output_tolerances[1]),
        ("F(Q)", fq, original_fq.f, output_tolerances[2]),
        ("G(r)", gr, original_gr.g, output_tolerances[3]),
    ):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            msg=lambda msg, name=name: f"{name} mismatch: {msg}",
        )

    rwp = torch.sqrt(
        torch.mean(torch.square(fq - original_fq.f))
        / torch.mean(torch.square(original_fq.f))
    )
    assert rwp < output_tolerances[4], (
        f"Rwp for F(Q) is too high: {rwp.item()}"
    )
