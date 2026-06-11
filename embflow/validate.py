"""Prefix-validation protocol: does lens accumulation approximate the
true prefix-embedding path?

Port of the embedding-dynamics paper's gating experiment (exp04): for
each conversation, embed every message AND every prefix concatenation
``embed(x_1..x_k)``, then test how closely weighted accumulations of the
per-message embeddings track the prefix path. Null calibration uses
mismatched accumulation/prefix pairs and cross-prefix baselines; a
pooled top-1 retrieval metric checks that accumulations identify their
own prefix among all others. Gate criterion (the paper's defaults):
mean cosine(accumulation, prefix) for the best lens >= 0.8 AND
>= mismatched-pair null + 0.25.

Having this protocol in the package is what makes "embflow implements
the calculus" a strong claim: any embedding model can be validated
against it with one function call and an ``embed_fn``
(see ``embflow.backends``).
"""
import numpy as np

from embflow.ops import adaptive_alpha

EPS = 1e-8

# Paper defaults (exp04): recency alphas, primacy betas.
EXP_ALPHAS = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99)
PRIMACY_BETAS = (0.80, 0.90, 0.95)


def lens_weights(kind, param, k):
    """Lens weight vector over positions j = 1..k.

    - "uniform": all ones.
    - "exp": recency, w(j,k) = alpha^(k-j). HIGHER alpha = LONGER
      memory; alpha -> 1 approaches uniform. Half-life
      log(0.5)/log(alpha) steps.
    - "primacy": w(j) = beta^(j-1); the FIRST item has weight 1.

    Parameters
    ----------
    kind : str
    param : float
        alpha or beta (ignored for "uniform").
    k : int
        Prefix length.

    Returns
    -------
    ndarray of shape (k,)
    """
    j = np.arange(1, k + 1)
    if kind == "uniform":
        return np.ones(k)
    if kind == "exp":
        return param ** (k - j)
    if kind == "primacy":
        return param ** (j - 1)
    raise ValueError(f"unknown lens kind: {kind!r}")


def default_lenses(exp_alphas=EXP_ALPHAS, primacy_betas=PRIMACY_BETAS):
    """The exp04 lens family: (label, kind, param, mass_weighted) tuples.

    Uniform as the explicit baseline, exponential (recency) and primacy
    lenses, each with and without token-mass weighting (the prefix
    concatenation is intrinsically length-weighted: a 500-token message
    contributes more to the prefix embedding than a 5-token one).
    """
    lenses = []
    for mass in (False, True):
        suffix = "+mass" if mass else ""
        lenses.append((f"uniform{suffix}", "uniform", 0.0, mass))
        for a in exp_alphas:
            lenses.append((f"exp{a}{suffix}", "exp", a, mass))
        for b in primacy_betas:
            lenses.append((f"primacy{b}{suffix}", "primacy", b, mass))
    return lenses


def accumulate(vectors, masses, kind, param, mass, k):
    """Unit-normalized lens accumulation of the first k vectors."""
    w = lens_weights(kind, param, k)
    if mass:
        w = w * masses[:k]
    v = (w[:, None] * vectors[:k]).sum(axis=0)
    return v / (np.linalg.norm(v) + EPS)


def _approx_token_len(text):
    """Heuristic token count: ceil(len/3.5) (~3.5 chars/token English).

    Pass an exact ``token_len_fn`` (e.g. tiktoken) to
    ``prefix_experiment`` when the budget guard must be precise.
    """
    return int(np.ceil(len(text) / 3.5))


def _default_prefix(messages, k):
    """The exp04 prefix format: "[role]: content" blocks joined by blank
    lines; messages without a "role" key contribute bare content."""
    parts = []
    for m in messages[:k]:
        if "role" in m:
            parts.append(f"[{m['role']}]: {m['content']}")
        else:
            parts.append(m["content"])
    return "\n\n".join(parts)


def prefix_experiment(conversations, embed_fn, lenses=None,
                      max_prefix_tokens=None, token_len_fn=None,
                      prefix_fn=None, ids=None, n_nulls=400, seed=42,
                      gate_mean=0.8, gate_margin=0.25):
    """Run the prefix-validation protocol over a set of conversations.

    For every conversation, embeds all messages and all prefix
    concatenations, scores every lens's accumulation against the prefix
    path, calibrates against mismatched-pair and cross-prefix nulls,
    and computes pooled top-1 retrieval and per-conversation alpha
    coherence (best-fit lens alpha vs ``adaptive_alpha``).

    Embedding cost: n messages + valid_k prefixes per conversation; wrap
    ``embed_fn`` with ``embflow.backends.cached_embed_fn`` so re-runs
    are free. Embeddings are unit-normalized internally.

    Parameters
    ----------
    conversations : list of list of dict
        Each message dict needs "content"; "role" is included in the
        prefix text when present. At least 2 conversations with >= 2
        valid prefixes each (nulls and retrieval are cross-conversation).
    embed_fn : callable (list[str]) -> ndarray (n, d)
        See ``embflow.backends``.
    lenses : list of (label, kind, param, mass) tuples, optional
        Default ``default_lenses()`` (the exp04 family).
    max_prefix_tokens : int, optional
        Positions are evaluated only while the prefix fits this budget
        (exp04 used 7500 against an 8191-token window). None = no guard.
    token_len_fn : callable (str) -> int, optional
        Default: ceil(len/3.5) heuristic; pass tiktoken for exact.
    prefix_fn : callable (messages, k) -> str, optional
        Prefix text builder; default exp04 "[role]: content" format.
    ids : list of str, optional
        Conversation identifiers (default "conv0", "conv1", ...).
    n_nulls : int
        Mismatched/cross-prefix null draws (exp04: 400).
    seed : int
    gate_mean, gate_margin : float
        Gate: best-lens mean >= gate_mean AND >= null + gate_margin.

    Returns
    -------
    dict with lens_means, best_lens, null_mismatched_mean,
    null_crossprefix_mean, retrieval_top1_exact, retrieval_top1_conv,
    alpha_coherence, gate, per_conversation (agreement curves per lens).
    """
    if lenses is None:
        lenses = default_lenses()
    token_len = token_len_fn or _approx_token_len
    prefix = prefix_fn or _default_prefix
    if ids is None:
        ids = [f"conv{i}" for i in range(len(conversations))]
    if len(ids) != len(conversations):
        raise ValueError("ids must align with conversations")

    rng = np.random.default_rng(seed)
    lens_by_label = {label: (kind, param, mass) for label, kind, param, mass in lenses}

    msg_embs, pre_embs, valid_k, masses = {}, {}, {}, {}
    for cid, msgs in zip(ids, conversations):
        if len(msgs) == 0:
            raise ValueError(f"conversation {cid!r} is empty")
        masses[cid] = np.array([token_len(m["content"]) for m in msgs], dtype=float)
        K = 0
        for k in range(1, len(msgs) + 1):
            if max_prefix_tokens is not None and token_len(prefix(msgs, k)) > max_prefix_tokens:
                break
            K = k
        valid_k[cid] = K
        E = np.asarray(embed_fn([m["content"] for m in msgs]), dtype=float)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + EPS)
        P = np.asarray(embed_fn([prefix(msgs, k) for k in range(1, K + 1)]), dtype=float)
        if K > 0:
            P = P / (np.linalg.norm(P, axis=1, keepdims=True) + EPS)
        msg_embs[cid], pre_embs[cid] = E, P

    eligible = [cid for cid in ids if valid_k[cid] >= 2]
    if len(eligible) < 2:
        raise ValueError(
            "prefix_experiment requires >= 2 conversations with >= 2 valid "
            f"prefixes (got {len(eligible)}); nulls and retrieval are "
            "cross-conversation"
        )

    # Agreement curves: cos(accumulation_k, prefix_k) per lens per k.
    per_conversation = {}
    for cid in ids:
        E, P, K = msg_embs[cid], pre_embs[cid], valid_k[cid]
        sims = {}
        for label, kind, param, mass in lenses:
            if K > 0:
                acc = np.stack(
                    [accumulate(E, masses[cid], kind, param, mass, k)
                     for k in range(1, K + 1)]
                )
                sims[label] = (acc * P[:K]).sum(axis=1).tolist()
            else:
                sims[label] = []
        per_conversation[cid] = {"K": K, "sims": sims}

    # Per-lens summary over eligible conversations, positions k >= 2.
    lens_means = {}
    for label, *_ in lenses:
        vals = [np.mean(per_conversation[cid]["sims"][label][1:]) for cid in eligible]
        lens_means[label] = float(np.mean(vals))
    best_lens = max(lens_means, key=lens_means.get)
    bkind, bparam, bmass = lens_by_label[best_lens]

    # Nulls: mismatched accumulation/prefix pairs + cross-prefix baseline.
    null_mismatch, null_crossprefix = [], []
    for _ in range(n_nulls):
        a, b = rng.choice(eligible, size=2, replace=False)
        k = int(rng.integers(2, min(valid_k[a], valid_k[b]) + 1))
        acc_b = accumulate(msg_embs[b], masses[b], bkind, bparam, bmass, k)
        null_mismatch.append(float(pre_embs[a][k - 1] @ acc_b))
        null_crossprefix.append(float(pre_embs[a][k - 1] @ pre_embs[b][k - 1]))

    # Pooled top-1 retrieval with the best lens.
    pool_pre, pool_acc, owners = [], [], []
    for cid in eligible:
        for k in range(2, valid_k[cid] + 1):
            pool_pre.append(pre_embs[cid][k - 1])
            pool_acc.append(
                accumulate(msg_embs[cid], masses[cid], bkind, bparam, bmass, k)
            )
            owners.append((cid, k))
    PRE, ACC = np.stack(pool_pre), np.stack(pool_acc)
    nn = (PRE @ ACC.T).argmax(axis=1)
    top1_exact = float(np.mean([owners[i] == owners[j] for i, j in enumerate(nn)]))
    top1_conv = float(np.mean([owners[i][0] == owners[j][0] for i, j in enumerate(nn)]))

    # Alpha coherence: best-fit exp alpha (mass-weighted variants when
    # available, matching exp04) vs the adaptive alpha of the messages.
    exp_lenses = [(label, param) for label, kind, param, mass in lenses
                  if kind == "exp" and mass]
    if not exp_lenses:
        exp_lenses = [(label, param) for label, kind, param, mass in lenses
                      if kind == "exp"]
    alpha_coherence = []
    for cid in eligible:
        best_a, best_v = None, -np.inf
        for label, a in exp_lenses:
            v = float(np.mean(per_conversation[cid]["sims"][label][1:]))
            if v > best_v:
                best_a, best_v = a, v
        alpha_coherence.append({
            "id": cid,
            "best_fit_alpha": best_a,
            "adaptive_alpha": adaptive_alpha(msg_embs[cid]),
        })

    gate_value = lens_means[best_lens]
    null_mean = float(np.mean(null_mismatch))
    return {
        "lens_means": lens_means,
        "best_lens": best_lens,
        "null_mismatched_mean": null_mean,
        "null_crossprefix_mean": float(np.mean(null_crossprefix)),
        "retrieval_top1_exact": top1_exact,
        "retrieval_top1_conv": top1_conv,
        "alpha_coherence": alpha_coherence,
        "gate": {
            "best_lens_mean": gate_value,
            "criterion": f"mean >= {gate_mean} and mean >= mismatched_null + {gate_margin}",
            "passed": bool(gate_value >= gate_mean and gate_value >= null_mean + gate_margin),
        },
        "per_conversation": per_conversation,
    }
