"""Prefix-vs-smoothed-per-item experiment.

Tests whether the prefix-embedding path

    t_k = embed(x_1 .. x_k)

can be approximated by a weighted combination of per-message embeddings

    t_k^(w) = normalize( sum_{j<=k} w_kj * embed(x_j) )

for various weighting schemes. See README.md for context, design, and
how to interpret the output.
"""
