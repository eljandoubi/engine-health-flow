import numpy as np

from engine_health_flow.data import HEALTH, SENSOR_COLUMNS, read_csv


def test_read_csv_groups_and_sorts_rows(toy_csv):
    ds = read_csv(toy_csv)
    assert ds.n_seq == 3
    assert ds.health.shape == (16, len(HEALTH))
    assert ds.sensors.shape == (16, len(SENSOR_COLUMNS))
    assert ds.health.dtype == np.float32

    # ids come out in order of first appearance, so look sequences up by name
    for i, n in zip(range(3), (5, 7, 4), strict=True):
        s = int(np.flatnonzero(ds.ids == f"seq-{i}")[0])
        assert ds.lengths[s] == n
        steps = ds.health[ds.rows(s)][:, 0]
        assert np.allclose(steps, i * 100 + np.arange(n))  # sorted by timestep


def test_maintenance_flag_is_kept(toy_csv):
    ds = read_csv(toy_csv)
    assert ds.maintenance.dtype == bool
    assert ds.maintenance.sum() == 3  # one per sequence


def test_take_keeps_sequence_boundaries(toy_csv):
    ds = read_csv(toy_csv)
    sub = ds.take([2, 0])
    assert sub.ids.tolist() == [ds.ids[2], ds.ids[0]]
    assert sub.starts[0] == 0
    assert sub.lengths.tolist() == [ds.lengths[2], ds.lengths[0]]
    assert np.array_equal(sub.health[sub.rows(0)], ds.health[ds.rows(2)])
    assert np.array_equal(sub.sensors[sub.rows(1)], ds.sensors[ds.rows(0)])
