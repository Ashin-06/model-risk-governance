import numpy as np
import pandas as pd

from mrmgov.realdata import FEATURES, experiments


def test_uci_checks_run_on_feature_split_windows(tmp_path) -> None:
    rng = np.random.default_rng(0)
    n = 1200
    df = pd.DataFrame({c: rng.integers(-2, 9, n).astype(float) for c in FEATURES})
    df["LIMIT_BAL"] = rng.integers(10_000, 500_000, n)
    df["AGE"] = rng.integers(21, 70, n)
    df["SEX"] = rng.integers(1, 3, n)
    df["default payment next month"] = (df["PAY_0"] + rng.normal(0, 2, n) > 3).astype(int)
    f = tmp_path / "uci.csv"
    df.to_csv(f, index=False)

    res = experiments(f, log=lambda *_: None)
    assert set(res["scenarios"]) == {"random", "age", "limit"}
    age = res["scenarios"]["age"]
    assert "AGE" in age["drift"]["covariate_drifted_features"]          # the split feature itself
    assert age["rows"]["production"] == int((df["AGE"] >= 35).sum())
    for sc in res["scenarios"].values():
        assert set(sc["conformal"]["production"]["by_sex"]) == {"female", "male"}
        assert sum(sc["rows"].values()) == n
    assert set(res["fairness"]["group_thresholds_from_calibration"]) == {"female", "male"}
    assert set(res["abstention_by_alpha"]) == {"0.05", "0.1", "0.2"}
