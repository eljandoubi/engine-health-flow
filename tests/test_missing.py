import numpy as np
import pytest
import torch

from engine_health_flow.data import PHASES, SENSORS
from engine_health_flow.missing import PhaseFlags, drop_phases, phase_of_channel, run_missing
from engine_health_flow.report import missing_table
from engine_health_flow.scaling import Prepared

SMALL_UKF = {
    "q_scales": (1.0,),
    "r_scales": (1.0,),
    "steps": 30,
    "batch": 64,
    "hidden": 16,
    "depth": 1,
}
SMALL_FLOW = {"steps": 5, "batch": 16, "repeats": 2, "hidden": 8, "field_hidden": 16}


def test_channels_are_grouped_by_flight_phase(synthetic):
    prep = Prepared.fit(synthetic())
    phase = phase_of_channel(prep)
    assert len(phase) == len(prep.cols)
    assert phase.min() == 0 and phase.max() == len(PHASES) - 1
    assert np.array_equal(phase, np.sort(phase))  # the csv columns are ordered by phase
    assert (phase == 0).sum() == len(SENSORS)


def test_phase_flags_zero_the_lost_phase_and_say_so(synthetic):
    ds = synthetic(n_seq=3)
    prep = Prepared.fit(ds)
    phase = phase_of_channel(prep)
    full = PhaseFlags(prep).x(ds)
    n = len(prep.cols)
    assert full.shape == (len(ds.health), n + len(PHASES))
    np.testing.assert_array_equal(full[:, :n], prep.x(ds))
    assert (full[:, n:] == 1).all()

    lost = PhaseFlags(prep, missing=2).x(ds)
    assert (lost[:, :n][:, phase == 2] == 0).all()
    np.testing.assert_array_equal(lost[:, :n][:, phase != 2], full[:, :n][:, phase != 2])
    assert lost[:, n:].mean(0).tolist() == [1, 1, 0, 1]
    np.testing.assert_array_equal(PhaseFlags(prep, missing=2).x(ds, np.array([4, 9])), lost[[4, 9]])


def test_dropping_phases_hits_whole_windows_and_updates_the_flags(synthetic):
    prep = Prepared.fit(synthetic())
    n = len(prep.cols)
    phase = torch.from_numpy(phase_of_channel(prep))
    window = torch.ones(2000, 7, n + len(PHASES))

    torch.manual_seed(0)
    out = drop_phases(prep, p=0.3)(window)
    flags = out[:, 0, n:]
    assert torch.equal(out[:, :, n:], flags[:, None, :].expand(-1, 7, -1))  # same at every step
    assert flags.mean().item() == pytest.approx(0.7, abs=0.03)
    # the readings of a phase are zero exactly when its flag is
    assert torch.equal(out[:, 0, :n], flags[:, phase])
    assert (window == 1).all()  # the input batch was not changed in place


def test_dropping_nothing_leaves_the_windows_alone(synthetic):
    prep = Prepared.fit(synthetic())
    window = torch.randn(10, 5, len(prep.cols) + len(PHASES))
    window[:, :, len(prep.cols) :] = 1.0
    assert torch.equal(drop_phases(prep, p=0.0)(window), window)


def test_ukf_knows_less_without_a_phase(fitted_ukf):
    model, _, test = fitted_ukf
    phase = phase_of_channel(model.prep)
    mean, std = model.filtered(test)
    all_channels = np.arange(len(phase))
    same_mean, same_std = model.filtered(test, keep=all_channels)
    np.testing.assert_allclose(same_mean, mean)
    np.testing.assert_allclose(same_std, std)

    _, wider = model.filtered(test, keep=np.flatnonzero(phase != 0))
    assert wider.mean() > std.mean()


def test_run_missing_scores_every_lost_phase(synthetic):
    ds = synthetic(n_seq=10, length=80, jump_every=30)
    out = run_missing(ds, ukf_options=SMALL_UKF, flow_options=SMALL_FLOW)
    assert list(out["conditions"]) == ["none", *PHASES]
    for scores in out["conditions"].values():
        for model in ("ukf", "flow"):
            assert scores[model]["rmse"] > 0
            assert 0 <= scores[model]["coverage_90"] <= 1
            rescaled = scores[model]["rescaled"]
            assert rescaled["crps_mean"] > 0 and 0 <= rescaled["coverage_90"] <= 1
    assert set(out["spread_scales"]) == {"ukf", "flow"}
    assert all(len(s) == 10 for s in out["spread_scales"].values())


def test_missing_table_has_one_row_per_condition():
    scores = {
        "rmse": 0.001,
        "coverage_90": 0.9,
        "rescaled": {"crps_mean": 0.0003, "coverage_90": 0.85},
    }
    result = {
        "conditions": {
            "none": {"ukf": scores, "flow": scores},
            "Cruise": {"ukf": scores, "flow": scores},
        }
    }
    lines = missing_table([result]).splitlines()
    assert lines[0].startswith("| lost phase | RMSE x1e3 ukf | RMSE x1e3 flow |")
    assert [line.split("|")[1].strip() for line in lines[2:]] == ["none", "Cruise"]
    assert "1.00" in lines[2] and "0.90" in lines[2]
    assert "0.300" in lines[2] and "0.85" in lines[2]  # rescaled CRPS and coverage
