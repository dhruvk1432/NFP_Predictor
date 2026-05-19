import pandas as pd


def test_training_dataset_cache_reads_from_prior_output_root(tmp_path, monkeypatch):
    import Train.training_dataset_cache as cache

    monkeypatch.setattr(
        cache,
        "CACHE_DIR",
        tmp_path / "current_output" / "cache" / "training_dataset",
    )
    monkeypatch.setattr(cache, "compute_cache_key", lambda *args, **kwargs: "abc123")

    prior_root = tmp_path / "prior_output"
    prior_cache_dir = prior_root / "cache" / "training_dataset"
    prior_cache_dir.mkdir(parents=True)
    x_path, y_path = cache._paths_for(
        "nsa",
        "first",
        "revised",
        "abc123",
        prior_cache_dir,
    )
    pd.DataFrame({
        "ds": [pd.Timestamp("2024-01-01")],
        "feature": [1.5],
    }).to_parquet(x_path)
    pd.DataFrame({"y_mom": [123.0]}).to_parquet(y_path)

    monkeypatch.setenv("NFP_TRAIN_DATASET_CACHE_READ_ROOTS", str(prior_root))

    loaded = cache.load_cached_dataset(
        pd.DataFrame({"ds": [pd.Timestamp("2024-01-01")]}),
        "nsa",
        "first",
        "revised",
        start_date=None,
        end_date=None,
    )

    assert loaded is not None
    X, y = loaded
    assert X["feature"].tolist() == [1.5]
    assert y.tolist() == [123.0]
