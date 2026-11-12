import numpy as np

from engine_health_flow.data import read_csv
from engine_health_flow.scaling import Prepared, Scaler, informative_columns


def test_scaler_gives_zero_mean_unit_std_and_inverts():
    rng = np.random.default_rng(0)
    x = rng.normal([5.0, -3.0], [2.0, 0.1], size=(1000, 2)).astype(np.float32)
    s = Scaler.fit(x)
    z = s.transform(x)
    assert np.allclose(z.mean(0), 0, atol=1e-4)
    assert np.allclose(z.std(0), 1, atol=1e-4)
    assert np.allclose(s.inverse(z), x, atol=1e-4)


def test_constant_columns_are_not_informative():
    rng = np.random.default_rng(0)
    x = np.c_[rng.normal(size=100), np.full(100, 4031.0), rng.normal(10, 1, size=100)]
    assert informative_columns(x.astype(np.float32)).tolist() == [0, 2]


def test_a_column_with_float32_rounding_noise_counts_as_constant():
    x = np.full((100, 1), 4031.0, dtype=np.float32)
    x[::2] += np.float32(5e-4)  # one ulp at this magnitude
    assert informative_columns(x).size == 0


def test_prepared_uses_training_statistics_only(toy_csv):
    ds = read_csv(toy_csv)
    train = ds.take([0, 1])
    prep = Prepared.fit(train)
    assert prep.x(train).shape == (len(train.sensors), len(prep.cols))
    assert np.allclose(prep.x(train).mean(0), 0, atol=1e-4)
    # another sequence is standardised with the same numbers, so it is not centred
    other = ds.take([2])
    assert not np.allclose(prep.x(other).mean(0), 0, atol=1e-2)
    assert prep.y(other).shape == other.health.shape
