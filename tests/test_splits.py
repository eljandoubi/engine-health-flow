import numpy as np

from engine_health_flow.splits import make_folds


def test_every_sequence_is_tested_once():
    folds = make_folds(559)
    tested = np.concatenate([f.test for f in folds])
    assert sorted(tested.tolist()) == list(range(559))


def test_sets_do_not_overlap_and_cover_everything():
    for f in make_folds(103, seed=3):
        parts = [set(f.train.tolist()), set(f.val.tolist()), set(f.test.tolist())]
        assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
        assert len(parts[0] | parts[1] | parts[2]) == 103


def test_shares_are_close_to_70_10_20():
    f = make_folds(559)[0]
    n = 559
    assert abs(len(f.test) / n - 0.2) < 0.01
    assert abs(len(f.val) / n - 0.1) < 0.01
    assert abs(len(f.train) / n - 0.7) < 0.01


def test_same_seed_same_folds_other_seed_other_folds():
    a, b, c = make_folds(80, seed=1), make_folds(80, seed=1), make_folds(80, seed=2)
    assert all(np.array_equal(x.test, y.test) for x, y in zip(a, b, strict=True))
    assert not all(np.array_equal(x.test, y.test) for x, y in zip(a, c, strict=True))


def test_indices_come_out_sorted():
    f = make_folds(50)[2]
    for part in (f.train, f.val, f.test):
        assert np.all(np.diff(part) > 0)
