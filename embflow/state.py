"""Linear state dynamics and normalized readout for embedding sequences.

The core recurrence of the calculus is the leaky integrator

    s_k = alpha * s_{k-1} + m_k * e_k,        s_{-1} = 0

where ``e_k`` are the input vectors and ``m_k`` an optional mass (e.g.
token count). This is the explicit-Euler step of the overdamped ODE

    dx/dt = -lambda * x + f(t),    with    alpha = exp(-lambda * dt)

so the exponential lens is derived, not chosen: it is what you get when
semantic state decays at a constant rate while inputs arrive as impulses.

Lens convention (package-wide): the state above weights input j at step
k by ``w(j, k) = alpha**(k - j)``. HIGHER alpha = LONGER memory;
``alpha -> 1`` approaches the running mean. The half-life
``h = log(0.5) / log(alpha)`` measures memory length in steps
(``alpha_to_half_life`` / ``half_life_to_alpha`` convert).

The dynamics are LINEAR in R^d. Unit normalization

    x_k = s_k / ||s_k||

is a READOUT for cosine comparison, not part of the dynamics. The API
exposes both: ``leaky_state`` returns raw states, ``trajectory`` the
normalized readout. ``trajectory(E, alpha)`` agrees with
``smooth_exponential(E, alpha)`` (the prefix-weighted-mean form divides
by a positive scalar, which cannot change direction); use ``trajectory``
when you need masses or the unnormalized state alongside.
"""
import numpy as np


def _unit_rows(matrix):
    """Unit-normalize each row; zero-norm rows pass through unchanged."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=matrix.copy(), where=norms > 0)


def alpha_to_half_life(alpha):
    """Steps until a message's weight halves: h = log(0.5) / log(alpha).

    Lens convention w(j,k) = alpha^(k-j): higher alpha = longer memory.
    ``alpha = 0.5`` halves every step (h = 1); ``alpha = 1`` is the
    running mean (h = inf). Requires 0 < alpha <= 1.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if alpha == 1.0:
        return np.inf
    return float(np.log(0.5) / np.log(alpha))


def half_life_to_alpha(half_life):
    """Inverse of ``alpha_to_half_life``: alpha = 0.5 ** (1 / h).

    Requires h > 0; ``h = inf`` gives alpha = 1 (running mean).
    """
    if not half_life > 0.0:
        raise ValueError(f"half_life must be positive, got {half_life}")
    if np.isinf(half_life):
        return 1.0
    return float(0.5 ** (1.0 / half_life))


def leaky_state(vectors, alpha=0.85, masses=None):
    """Unnormalized leaky-integrator states: s_k = alpha*s_{k-1} + m_k*e_k.

    Lens convention w(j,k) = alpha^(k-j): HIGHER alpha = LONGER memory;
    alpha -> 1 approaches the (mass-weighted) running sum. Half-life
    h = log(0.5)/log(alpha) steps.

    Parameters
    ----------
    vectors : ndarray of shape (n, d)
    alpha : float
        Decay per step, in (0, 1].
    masses : array-like of shape (n,), optional
        Per-item mass (e.g. token count). None = all ones. The state is
        linear in the masses: scaling all masses scales all states.

    Returns
    -------
    ndarray of shape (n, d)
        Raw states. NOT normalized; see ``trajectory`` for the readout.
    """
    vectors = np.asarray(vectors, dtype=float)
    if len(vectors) == 0:
        raise ValueError("leaky_state requires at least one vector")
    if masses is None:
        masses = np.ones(len(vectors))
    masses = np.asarray(masses, dtype=float)
    if masses.shape != (len(vectors),):
        raise ValueError(
            f"masses must have shape ({len(vectors)},), got {masses.shape}"
        )
    states = np.empty_like(vectors)
    s = np.zeros(vectors.shape[1])
    for k in range(len(vectors)):
        s = alpha * s + masses[k] * vectors[k]
        states[k] = s
    return states


def trajectory(vectors, alpha=0.85, masses=None):
    """Normalized readout of the leaky state: x_k = s_k / ||s_k||.

    Row k is the direction of the exponentially-accumulated state after
    seeing ``vectors[0..k]``. Equivalent to ``smooth_exponential`` when
    ``masses`` is None (the weighted-mean denominator is a positive
    scalar and cannot change direction). The readout is invariant to
    scaling all masses by a constant.

    Lens convention w(j,k) = alpha^(k-j): HIGHER alpha = LONGER memory;
    alpha -> 1 approaches the running mean. Half-life
    h = log(0.5)/log(alpha) steps.

    Parameters
    ----------
    vectors : ndarray of shape (n, d)
    alpha : float
    masses : array-like of shape (n,), optional

    Returns
    -------
    ndarray of shape (n, d) with unit rows (zero states pass through).
    """
    return _unit_rows(leaky_state(vectors, alpha, masses))
