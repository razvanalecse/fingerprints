"""Model-based completion of a ridge orientation field, without learning.

Section 25 of the ablation table showed that harmonic (membrane) completion of
the doubled-angle field is far weaker than a trained model. Harmonic
interpolation is the weakest reasonable prior: it minimises the *first*
derivative, so it produces a flat solution inside a hole and forgets the slope
it had at the boundary. This module implements the stronger geometric priors
that the classical fingerprint literature actually uses.

Everything operates on the doubled-angle representation

    v = (cos 2t, sin 2t)

because ridge orientation is axial: t and t + pi are the same direction.

Methods, in increasing order of fingerprint-specific structure:

  harmonic      minimise |grad v|^2                     (membrane; the floor)
  biharmonic    minimise |laplacian v|^2                (thin plate: continues
                                                         the boundary *slope*)
  polynomial    least-squares global low-order surface  (captures global flow)
  zero_pole     theta0 + 1/2 [sum arg(z - core) - sum arg(z - delta)],
                singularity positions fitted to the observed field
  combination   zero-pole global model + harmonic completion of its residual

The zero-pole form is the classical singularity model: a core behaves like a
zero and a delta like a pole of a complex function, which is what makes ridge
flow curve coherently around them rather than merely smoothly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import least_squares


def _clamped_neighbour_index(height: int, width: int, dy: int, dx: int) -> np.ndarray:
    ys = np.clip(np.arange(height) + dy, 0, height - 1)
    xs = np.clip(np.arange(width) + dx, 0, width - 1)
    return (ys[:, None] * width + xs[None, :]).ravel()


def laplacian_matrix(height: int, width: int) -> sp.csr_matrix:
    """5-point Laplacian with zero-flux (Neumann) borders."""
    size = height * width
    centre = np.arange(size)
    rows = [centre]
    cols = [centre]
    data = [-4.0 * np.ones(size)]
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rows.append(centre)
        cols.append(_clamped_neighbour_index(height, width, dy, dx))
        data.append(np.ones(size))
    return sp.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(size, size),
    )


def _solve_with_dirichlet(operator: sp.csr_matrix, values: np.ndarray,
                          known: np.ndarray) -> np.ndarray:
    """Minimise ||operator u||^2 subject to u = values on `known`."""
    shape = values.shape
    flat_known = known.ravel()
    unknown = ~flat_known
    count = int(unknown.sum())
    if count == 0:
        return values.copy()
    normal = (operator.T @ operator).tocsr()
    lhs = normal[unknown][:, unknown]
    rhs = -normal[unknown][:, flat_known] @ values.ravel()[flat_known]
    solution = spla.spsolve(lhs.tocsc(), rhs)
    out = values.ravel().astype(np.float64).copy()
    out[unknown] = solution
    return out.reshape(shape)


def complete_harmonic(component: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Membrane prior: minimise |grad u|^2."""
    height, width = component.shape
    gradient = sp.vstack([
        _difference_matrix(height, width, axis=0),
        _difference_matrix(height, width, axis=1),
    ]).tocsr()
    return _solve_with_dirichlet(gradient, component, known)


def complete_biharmonic(component: np.ndarray, known: np.ndarray,
                        damping: float = 0.0) -> np.ndarray:
    """Thin-plate prior: minimise |laplacian u|^2 + damping * |grad u|^2.

    Pure thin-plate extrapolation continues the boundary slope, which is what a
    membrane prior cannot do, but it overshoots badly when the boundary data is
    noisy and the hole is large -- on real orientation fields it can be worse
    than doing nothing. The damping term blends back towards the membrane
    solution and buys stability; damping=0 is the pure thin plate.
    """
    height, width = component.shape
    operator = laplacian_matrix(height, width)
    if damping > 0:
        gradient = sp.vstack([
            _difference_matrix(height, width, axis=0),
            _difference_matrix(height, width, axis=1),
        ])
        operator = sp.vstack([operator, np.sqrt(damping) * gradient]).tocsr()
    return _solve_with_dirichlet(operator, component, known)


def fit_zero_pole_best(theta: np.ndarray, known: np.ndarray, weights: np.ndarray,
                       *, configurations=((1, 1), (1, 0), (2, 1), (0, 0)),
                       seed: int = 0):
    """Try several singularity counts and keep the one that fits the observed field best.

    A real print may be an arch (no singularity), a loop (one core, one delta)
    or a whorl (two cores, one delta); forcing a single configuration on all of
    them is a guaranteed source of error.
    """
    best, best_fit = None, None
    for num_cores, num_deltas in configurations:
        try:
            fit = fit_zero_pole(theta, known, weights, num_cores=num_cores,
                                num_deltas=num_deltas, seed=seed)
        except (ValueError, TypeError):
            continue
        if best is None or fit.cost < best:
            best, best_fit = fit.cost, fit
    if best_fit is None:
        raise ValueError("no zero-pole configuration converged")
    return best_fit


def _difference_matrix(height: int, width: int, axis: int) -> sp.csr_matrix:
    size = height * width
    centre = np.arange(size)
    forward = _clamped_neighbour_index(height, width, *( (1, 0) if axis == 0 else (0, 1) ))
    rows = np.concatenate([centre, centre])
    cols = np.concatenate([forward, centre])
    data = np.concatenate([np.ones(size), -np.ones(size)])
    return sp.csr_matrix((data, (rows, cols)), shape=(size, size))


def _polynomial_basis(height: int, width: int, degree: int) -> np.ndarray:
    ys, xs = np.mgrid[:height, :width]
    y = (ys / max(height - 1, 1)) * 2.0 - 1.0
    x = (xs / max(width - 1, 1)) * 2.0 - 1.0
    terms = [ (x ** i) * (y ** j) for i in range(degree + 1)
              for j in range(degree + 1 - i) ]
    return np.stack([t.ravel() for t in terms], axis=1)


def complete_polynomial(cos_component: np.ndarray, sin_component: np.ndarray,
                        known: np.ndarray, weights: np.ndarray,
                        degree: int = 5) -> np.ndarray:
    """Fit a global low-order surface to each component and evaluate everywhere."""
    height, width = cos_component.shape
    basis = _polynomial_basis(height, width, degree)
    rows = known.ravel()
    weight = np.sqrt(np.clip(weights.ravel()[rows], 0.0, None))[:, None]
    design = basis[rows] * weight
    fitted = []
    for component in (cos_component, sin_component):
        target = component.ravel()[rows] * weight[:, 0]
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        fitted.append((basis @ coefficients).reshape(height, width))
    return 0.5 * np.arctan2(fitted[1], fitted[0])


@dataclass(frozen=True)
class ZeroPoleFit:
    theta0: float
    cores: np.ndarray
    deltas: np.ndarray
    cost: float


def zero_pole_theta(height: int, width: int, theta0: float,
                    cores: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    ys, xs = np.mgrid[:height, :width]
    z = xs + 1j * ys
    angle = np.full((height, width), float(theta0))
    for cy, cx in np.atleast_2d(cores):
        angle = angle + 0.5 * np.angle(z - (cx + 1j * cy))
    for dy, dx in np.atleast_2d(deltas):
        angle = angle - 0.5 * np.angle(z - (dx + 1j * dy))
    return np.mod(angle, np.pi)


def fit_zero_pole(theta: np.ndarray, known: np.ndarray, weights: np.ndarray,
                  *, num_cores: int = 1, num_deltas: int = 1,
                  seed: int = 0) -> ZeroPoleFit:
    """Fit singularity positions so the model matches the observed orientations.

    Positions are free parameters rather than detected, because the interesting
    case is exactly the one where the singularity falls inside the hole and
    therefore cannot be detected from the observed region at all.
    """
    height, width = theta.shape
    rows = known & (weights > 0)
    if rows.sum() < 50:
        raise ValueError("not enough observed orientation for a zero-pole fit")
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(rows)
    if ys.size > 2000:                      # the fit has <=5 parameters; 2000
        pick = rng.choice(ys.size, 2000, replace=False)   # points is plenty
        ys, xs = ys[pick], xs[pick]
    rows = (ys, xs)
    observed = theta[rows]
    weight = np.sqrt(weights[rows])

    def unpack(params):
        theta0 = params[0]
        cores = params[1:1 + 2 * num_cores].reshape(-1, 2) if num_cores else np.zeros((0, 2))
        deltas = params[1 + 2 * num_cores:].reshape(-1, 2) if num_deltas else np.zeros((0, 2))
        return theta0, cores, deltas

    observed_vector = np.stack([np.cos(2.0 * observed), np.sin(2.0 * observed)])

    def residual(params):
        theta0, cores, deltas = unpack(params)
        model = zero_pole_theta(height, width, theta0, cores, deltas)[rows]
        # Difference of the doubled-angle unit vectors. Using sin(2*delta)
        # alone is degenerate: it also vanishes at a 90-degree offset, and the
        # fit then happily converges to the perpendicular field.
        model_vector = np.stack([np.cos(2.0 * model), np.sin(2.0 * model)])
        return (weight * (observed_vector - model_vector)).ravel()

    best = None
    for _ in range(4):
        start = np.concatenate([
            [rng.uniform(0, np.pi)],
            rng.uniform([0.25 * height, 0.25 * width], [0.75 * height, 0.75 * width],
                        size=(num_cores, 2)).ravel(),
            rng.uniform([0.25 * height, 0.25 * width], [0.75 * height, 0.75 * width],
                        size=(num_deltas, 2)).ravel(),
        ])
        try:
            result = least_squares(residual, start, method="lm", max_nfev=2000)
        except Exception:
            continue
        if best is None or result.cost < best.cost:
            best = result
    if best is None:
        raise ValueError("zero-pole fit did not converge")
    theta0, cores, deltas = unpack(best.x)
    return ZeroPoleFit(theta0=float(theta0), cores=cores, deltas=deltas,
                       cost=float(best.cost))


def complete_combination(theta: np.ndarray, known: np.ndarray, weights: np.ndarray,
                         fit: ZeroPoleFit) -> np.ndarray:
    """Global zero-pole model plus harmonic completion of its residual."""
    height, width = theta.shape
    model = zero_pole_theta(height, width, fit.theta0, fit.cores, fit.deltas)
    residual = theta - model
    cos_filled = complete_harmonic(np.cos(2.0 * residual), known)
    sin_filled = complete_harmonic(np.sin(2.0 * residual), known)
    correction = 0.5 * np.arctan2(sin_filled, cos_filled)
    return np.mod(model + correction, np.pi)


def mirror_orientation(theta: np.ndarray, axis_x: float) -> np.ndarray:
    """Reflect an axial orientation field about a vertical line at ``axis_x``.

    Reflection maps a direction to its mirror, so the angle becomes pi - theta,
    and the field is resampled from the mirrored column. Ridge orientation is
    axial, so the result stays in [0, pi).
    """
    height, width = theta.shape
    columns = np.clip(np.round(2.0 * axis_x - np.arange(width)).astype(int), 0, width - 1)
    return np.mod(np.pi - theta[:, columns], np.pi), columns


def fit_symmetry_axis(theta: np.ndarray, known: np.ndarray, weights: np.ndarray,
                      *, margin: float = 0.25):
    """Find the vertical axis whose reflection best explains the observed field.

    The fit uses only observed pixels, so the residual doubles as a
    ground-truth-free measure of how mirror-symmetric this particular print is.
    Returns (axis_x, mean_disagreement) with disagreement in [0, 2].
    """
    height, width = theta.shape
    best = (float(width) / 2.0, np.inf)
    lo, hi = int(margin * width), int((1.0 - margin) * width)
    for axis_x in range(lo, hi + 1):
        mirrored, columns = mirror_orientation(theta, float(axis_x))
        both = known & known[:, columns]
        if both.sum() < 100:
            continue
        w = weights[both]
        if w.sum() <= 0:
            continue
        disagreement = float(
            np.average(1.0 - np.cos(2.0 * (theta[both] - mirrored[both])), weights=w))
        if disagreement < best[1]:
            best = (float(axis_x), disagreement)
    return best


def complete_mirror_cascade(theta: np.ndarray, known: np.ndarray, weights: np.ndarray):
    """Fill from the mirrored side first, then close the leftovers harmonically.

    Where the hole's mirror image is itself unobserved -- which is most of a
    central hole, the case that matters -- reflection has nothing to copy. Those
    leftovers are completed with the smoothness prior, so the cascade degrades
    to plain harmonic completion rather than to nonsense.
    """
    axis_x, disagreement = fit_symmetry_axis(theta, known, weights)
    mirrored, columns = mirror_orientation(theta, axis_x)
    source_known = known[:, columns]
    filled = np.where(known, theta, np.where(source_known, mirrored, 0.0))
    supported = known | source_known
    cos_filled = complete_harmonic(np.cos(2.0 * filled), supported)
    sin_filled = complete_harmonic(np.sin(2.0 * filled), supported)
    return (np.mod(0.5 * np.arctan2(sin_filled, cos_filled), np.pi),
            axis_x, disagreement, float(supported.mean()))
