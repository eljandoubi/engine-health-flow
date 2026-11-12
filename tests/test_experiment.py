import pytest

from engine_health_flow.experiment import fit_model, run_baselines
from engine_health_flow.scaling import Prepared
from engine_health_flow.splits import make_folds
from engine_health_flow.windows import EVAL_STRIDE, WINDOW

FAST = {
    "mlp": {"steps": 30, "batch": 64, "hidden": 16, "depth": 1},
    "gru": {"steps": 3, "batch": 16, "hidden": 8},
    "ukf": {
        "steps": 30,
        "batch": 64,
        "hidden": 16,
        "depth": 1,
        "q_scales": (1.0,),
        "r_scales": (1.0,),
    },
}


def test_every_model_gets_val_and_test_metrics(synthetic):
    ds = synthetic(n_seq=10, length=80, jump_every=30)
    out = run_baselines(ds, names=("ridge", "mlp", "gru", "ukf"), options=FAST)
    assert list(out["models"]) == ["ridge", "mlp", "gru", "ukf"]
    for res in out["models"].values():
        for part in ("val", "test"):
            assert len(res[part]["rmse"]) == 10
            assert res[part]["pooled"]["rmse"] > 0


def test_only_rows_with_a_full_window_are_scored(synthetic):
    ds = synthetic(n_seq=10, length=80)
    out = run_baselines(ds, fold=2, names=("ridge",))
    per_sequence = len(range(WINDOW - 1, 80, EVAL_STRIDE))
    fold = make_folds(10)[2]
    assert out["scored_rows"] == {
        "val": len(fold.val) * per_sequence,
        "test": len(fold.test) * per_sequence,
    }
    assert out["sequences"] == {
        "train": len(fold.train),
        "val": len(fold.val),
        "test": len(fold.test),
    }


def test_ridge_follows_linear_synthetic_data(synthetic):
    out = run_baselines(synthetic(n_seq=10, length=80), names=("ridge",))
    assert out["models"]["ridge"]["test"]["pooled"]["pearson"] > 0.9


def test_a_model_name_must_be_known(synthetic):
    ds = synthetic(n_seq=3)
    with pytest.raises(ValueError, match="svm"):
        fit_model("svm", Prepared.fit(ds), ds, ds, seed=0)
