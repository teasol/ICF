"""CT branch: deterministic farthest-point tokens over context cells."""

from __future__ import annotations

from src.models.ct_readout import CTReadoutConfig, ct_margins


def ct_features(config, context_bags, labels, query_bags, basis=None):
    """Two-token abundance readout, delegated to `ct_readout` (docs SS148).

    Steps 1-5 (sample, standardise, farthest-point tokens, soft assign, per-bag
    average) live in `ct_readout.ct_abundance` so that the readout experiments
    cannot accidentally differ from this path in the REPRESENTATION -- only in
    step 6-7. `mode="extreme"` is today's behaviour and stays the default, so
    v107's output is unchanged; `tests/test_training_free.py` pins that against
    the lineage and `tests/test_ct_readout.py` pins the refactor itself.
    """
    margins, _ = ct_margins(
        context_bags, labels, query_bags,
        CTReadoutConfig(
            num_tokens=config.ct_num_tokens,
            cells_per_bag=config.ct_cells_per_bag,
            abundance_cells_per_bag=config.ct_abundance_cells_per_bag,
            cells_fraction=config.ct_cells_fraction,
            cells_min=config.ct_cells_min,
            cells_scale=config.ct_cells_scale,
            sampling=config.ct_sampling,
            sampling_seed=config.ct_sampling_seed,
            distance_kernel=config.ct_distance_kernel,
            tokenizer=config.ct_tokenizer,
            bisect_iterations=config.ct_bisect_iterations,
            bisect_power_iterations=config.ct_bisect_power_iterations,
            tree_reduction=config.ct_tree_reduction,
            hdbscan_min_cluster_size=config.ct_hdbscan_min_cluster_size,
            hdbscan_min_cluster_fraction=config.ct_hdbscan_min_cluster_fraction,
            hdbscan_min_samples=config.ct_hdbscan_min_samples,
            hdbscan_cluster_selection_method=(
                config.ct_hdbscan_cluster_selection_method
            ),
            hdbscan_build_algo=config.ct_hdbscan_build_algo,
            hdbscan_allow_single_cluster=config.ct_hdbscan_allow_single_cluster,
            dbscan_eps=config.ct_dbscan_eps,
            dbscan_min_samples=config.ct_dbscan_min_samples,
            temperature=config.ct_temperature,
            eps=config.ct_eps,
            pca_dim=config.ct_pca_dim,
            kmeans_iterations=config.ct_kmeans_iterations,
            kmeans_max_iterations=config.ct_kmeans_max_iterations,
            kmeans_tolerance=config.ct_kmeans_tolerance,
            kmeans_seed=config.ct_kmeans_seed,
        ),
        mode=config.ct_readout,
        # The SAME within-slide basis the CV branch uses, sliced to
        # `ct_pca_dim`. Reusing it costs no extra eigh -- and is also why the
        # gain is capped, since CT then lives inside a subspace CV already
        # covers (SS149-4).
        pca_basis=basis,
    )
    # The head consumes (q0, q1) and weighs q1 - q0, so hand back a pair whose
    # difference IS the margin. For "extreme" this returns exactly the two
    # standardised token abundances it always did.
    return -0.5 * margins.query, 0.5 * margins.query
