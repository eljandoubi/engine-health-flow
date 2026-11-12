import pytest

from engine_health_flow.experiment import fit_model, run_fold
from engine_health_flow.scaling import Prepared
from engine_health_flow.splits import make_folds
from engine_health_flow.windows import EVAL_STRIDE, WINDOW

FAST_UKF = {
    "steps": 30,
    "batch": 64,
    "hidden": 16,
    "depth": 1,
    "q_scales": (1.0,),
    "r_scales": (1.0,),
}
FAST_FLOW = {"steps": 5, "batch": 16, "repeats": 2, "hidden": 8, "field_hidden": 16}
FAST = {
    "mlp": {"steps": 30, "batch": 64, "hidden": 16, "depth": 1},
    "gru": {"steps": 3, "batch": 16, "hidden": 8},
    "ukf": FAST_UKF,
    "flow": FAST_FLOW,
    "hybrid": {"ukf_options": FAST_UKF, **FAST_FLOW},
}


def test_every_model_gets_val_and_test_metrics(synthetic):
    ds = synthetic(n_seq=10, length=80, jump_every=30)
    names = ("ridge", "mlp", "gru", "ukf", "flow", "hybrid")
    out = run_fold(ds, names=names, options=FAST)
    assert list(out["models"]) == list(names)
    for res in out["models"].values():
        for part in ("val", "test"):
            assert len(res[part]["rmse"]) == 10
            assert res[part]["pooled"]["rmse"] > 0


def test_only_models_that_sample_get_posterior_scores(synthetic):
    ds = synthetic(n_seq=10, length=80, jump_every=30)
    out = run_fold(ds, names=("ridge", "ukf", "flow"), options=FAST)
    has_posterior = {name: "posterior" in res["test"] for name, res in out["models"].items()}
    assert has_posterior == {"ridge": False, "ukf": True, "flow": True}
    assert "spread_scales" not in out["models"]["ridge"]
    posterior = out["models"]["ukf"]["test"]["posterior"]
    assert len(posterior["crps"]) == 10
    assert set(posterior["intervals"]) == {"0.5", "0.9"}


def test_rescaled_posterior_is_fitted_on_validation(synthetic):
    ds = synthetic(n_seq=10, length=80, jump_every=30)
    res = run_fold(ds, names=("ukf",), options=FAST)["models"]["ukf"]
    assert len(res["spread_scales"]) == 10
    # the factor 1 is one of the candidates, so on validation the rescaling cannot make it worse
    raw, rescaled = res["val"]["posterior"], res["val"]["posterior_rescaled"]
    assert rescaled["crps_mean"] <= raw["crps_mean"] + 1e-12
    assert "posterior_rescaled" in res["test"]


def test_only_rows_with_a_full_window_are_scored(synthetic):
    ds = synthetic(n_seq=10, length=80)
    out = run_fold(ds, fold=2, names=("ridge",))
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
    out = run_fold(synthetic(n_seq=10, length=80), names=("ridge",))
    assert out["models"]["ridge"]["test"]["pooled"]["pearson"] > 0.9


def test_a_model_name_must_be_known(synthetic):
    ds = synthetic(n_seq=3)
    with pytest.raises(ValueError, match="svm"):
        fit_model("svm", Prepared.fit(ds), ds, ds, seed=0)
