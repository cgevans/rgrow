import numpy as np
import pytest
from rgrow import Bond, RBFFSRunConfig, RBFFSResult, Tile, TileSet


def make_test_tileset(**overrides):
    defaults = dict(
        tiles=[Tile(["a", "a", "b", "b"]), Tile(["b", "b", "a", "a"])],
        bonds=[Bond("a", 1), Bond("b", 1)],
        gse=5.7,
        gmc=9.7,
        alpha=-7.1,
        canvas_type="periodic",
        size=32,
        fission="keep-largest",
    )
    defaults.update(overrides)
    return TileSet(**defaults)


def test_rbffs_basic_run():
    ts = make_test_tileset()
    result = ts.run_rbffs(
        n_trials=50, n_trajectories=5, target_size=10, size_step=2
    )
    assert isinstance(result, RBFFSResult)
    assert result.nucleation_rate > 0

    fps = result.forward_probabilities
    assert isinstance(fps, np.ndarray)
    assert fps.ndim == 1
    assert len(fps) > 0
    assert np.all(fps >= 0)
    assert np.all(fps <= 1)

    weights = result.trajectory_weights
    assert isinstance(weights, np.ndarray)
    assert len(weights) >= 5


def test_rbffs_config_construction():
    cfg = RBFFSRunConfig(
        n_trials=200,
        n_trajectories=50,
        target_size=20,
        canvas_size=(64, 64),
        size_step=3,
        parallel=True,
        num_workers=4,
    )
    assert cfg.n_trials == 200
    assert cfg.n_trajectories == 50
    assert cfg.target_size == 20
    assert cfg.canvas_size == (64, 64)
    assert cfg.size_step == 3
    assert cfg.parallel is True
    assert cfg.num_workers == 4


def test_rbffs_config_defaults():
    cfg = RBFFSRunConfig()
    assert cfg.n_trials == 1000
    assert cfg.n_trajectories == 1000
    assert cfg.target_size == 100
    assert cfg.canvas_size == (32, 32)
    assert cfg.size_step == 1
    assert cfg.parallel is False
    assert cfg.num_workers is None
    assert cfg.keep_full_trajectories is True
    assert cfg.store_system is False


def test_rbffs_bootstrap_ci():
    ts = make_test_tileset()
    result = ts.run_rbffs(
        n_trials=50, n_trajectories=10, target_size=8, size_step=2
    )
    bs = result.bootstrap_ci(200, 0.95)

    lo, hi = bs.nucleation_rate_ci
    assert lo <= hi
    assert lo >= 0

    median = bs.nucleation_rate_median
    assert lo <= median <= hi

    samples = bs.nucleation_rate_samples
    assert isinstance(samples, np.ndarray)
    assert len(samples) == 200

    fp_samples = bs.forward_probability_samples
    assert isinstance(fp_samples, np.ndarray)
    assert fp_samples.shape[0] == 200

    fp_cis = bs.forward_probability_cis
    for flo, fhi in fp_cis:
        assert flo <= fhi
        assert flo >= 0
        assert fhi <= 1


def test_rbffs_trajectories():
    ts = make_test_tileset()
    result = ts.run_rbffs(
        n_trials=50, n_trajectories=5, target_size=8, size_step=2
    )
    trajs = result.trajectories
    assert len(trajs) >= 5
    for traj in trajs:
        assert len(traj) > 0
        # Each state should have a canvas_view with proper shape
        state = traj[-1]
        cv = state.canvas_view
        assert cv.shape == (32, 32)


def test_rbffs_resample():
    ts = make_test_tileset()
    result = ts.run_rbffs(
        n_trials=50, n_trajectories=5, target_size=8, size_step=2
    )
    resampled = result.resample_trajectories(3)
    assert len(resampled) == 3

    unique = result.select_unique_trajectories(3)
    assert len(unique) <= 3


def test_rbffs_extend():
    ts = make_test_tileset()
    result = ts.run_rbffs(
        n_trials=50, n_trajectories=3, target_size=8, size_step=2,
        store_system=True
    )
    initial = result.n_trajectories
    result.extend(3)
    assert result.n_trajectories >= initial + 3


def test_rbffs_extend_raises():
    ts = make_test_tileset()
    result = ts.run_rbffs(
        n_trials=50, n_trajectories=3, target_size=8, size_step=2,
        store_system=False
    )
    with pytest.raises(TypeError):
        result.extend(3)


def test_rbffs_failed_trajectories():
    ts = make_test_tileset()
    result = ts.run_rbffs(
        n_trials=50, n_trajectories=5, target_size=10, size_step=2
    )
    # n_failed_trajectories should be a non-negative int
    assert result.n_failed_trajectories >= 0
    # failed_at_size is an array with length == n_failed_trajectories
    fas = result.failed_at_size
    assert isinstance(fas, np.ndarray)
    assert len(fas) == result.n_failed_trajectories


def test_rbffs_repr():
    ts = make_test_tileset()
    result = ts.run_rbffs(
        n_trials=50, n_trajectories=3, target_size=8, size_step=2
    )
    r = repr(result)
    assert "RBFFSResult" in r
