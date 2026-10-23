import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import churn_pipeline as pipeline
from churn_pipeline import analyze_segments, evaluate_policy, prepare_data, run_pipeline


def sample_data(n=120):
    rng = np.random.default_rng(7)
    tenure = rng.integers(1, 72, n)
    charges = rng.uniform(25, 110, n)
    churn = (rng.random(n) < (0.15 + charges / 500 - tenure / 600)).astype(int)
    return pd.DataFrame({
        "customerID": [f"C{i:04d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(["Yes", "No"], n), "Dependents": rng.choice(["Yes", "No"], n),
        "tenure": tenure, "PhoneService": rng.choice(["Yes", "No"], n),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check"], n),
        "MonthlyCharges": charges, "TotalCharges": charges * tenure,
        "Churn": np.where(churn, "Yes", "No"),
    })


def test_prepare_data_converts_blank_total_charges():
    frame = sample_data()
    frame["TotalCharges"] = frame["TotalCharges"].astype(object)
    frame.loc[0, "TotalCharges"] = " "
    X, y, ids = prepare_data(frame)
    assert X.TotalCharges.isna().sum() == 1
    assert y.nunique() == 2
    assert ids.is_unique


@pytest.mark.parametrize("change", ["missing", "duplicate", "single_label"])
def test_bad_input(change):
    frame = sample_data()
    if change == "missing":
        frame = frame.drop(columns="Contract")
    elif change == "duplicate":
        frame["customerID"] = frame["customerID"].iloc[0]
    else:
        frame["Churn"] = "No"
    with pytest.raises(ValueError):
        prepare_data(frame)


def test_run_pipeline_writes_results(monkeypatch):
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        output = Path(directory) / "outputs"
        captured = {}
        original_analyze = pipeline.analyze_segments

        def capture_training_segments(training_frame):
            captured["ids"] = set(training_frame.customerID)
            return original_analyze(training_frame)

        monkeypatch.setattr(pipeline, "analyze_segments", capture_training_segments)
        metrics = run_pipeline(sample_data(), output)
        assert set(metrics["模型"]) == {"逻辑回归", "随机森林"}
        assert metrics["交叉验证"]["折数"] == 5
        assert metrics["交叉验证"]["选择模型"] in metrics["模型"]
        assert (output / "metrics.json").is_file()
        predictions = pd.read_csv(output / "predictions.csv", dtype={"customerID": "string"})
        assert len(predictions) == 24
        assert predictions.customerID.isin(sample_data().customerID).all()
        assert len(captured["ids"]) == metrics["训练集样本数"]
        assert captured["ids"].isdisjoint(set(predictions.customerID))
        assignments = pd.read_csv(output / "policy_assignments.csv", dtype={"customerID": "string"})
        assert assignments.groupby("策略").size().le(int(len(predictions) * 0.1)).all()
        assert len(pd.read_csv(output / "policy_metrics.csv")) == 4
        segments = pd.read_csv(output / "segment_summary.csv")
        assert segments["客户数"].sum() >= metrics["训练集样本数"]
        assert (output / "overall_summary.csv").is_file()
        assert (output / "confusion_matrices.png").is_file()
        assert (output / "roc_curve.png").is_file()
        assert (output / "feature_importance.csv").is_file()
        saved = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        assert saved["测试集样本数"] == 24


def test_sql_segments_match_pandas():
    frame = sample_data(150)
    frame["Churn"] = frame["Churn"].map({"No": 0, "Yes": 1})
    _, segments = analyze_segments(frame)
    contract = segments.loc[segments["分群维度"].eq("合同类型")]
    expected = frame.groupby("Contract").Churn.sum()
    actual = contract.set_index("分群")["流失人数"].sort_index()
    pd.testing.assert_series_equal(actual, expected.astype(int).sort_index(), check_names=False)


def test_policy_ties_are_stable_and_zero_budget_is_defined():
    result, order = evaluate_policy(
        "模型", labels=[1, 0, 1], ids=["C3", "C1", "C2"], scores=[0.8, 0.8, 0.2], budget_count=1)
    assert order.tolist() == [1, 0, 2]
    assert result["名额"] == 1
    assert result["命中流失"] == 0
    empty, order = evaluate_policy(
        "模型", labels=[1, 0], ids=["C1", "C2"], scores=[0.8, 0.2], budget_count=0)
    assert len(order) == 2
    assert empty["Precision@K"] == empty["Recall@K"] == empty["Lift@K"] == 0


def test_dashboard_renders_chinese_headers(monkeypatch):
    from streamlit.testing.v1 import AppTest

    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        output = Path(directory) / "outputs"
        run_pipeline(sample_data(), output)
        monkeypatch.setattr(sys, "argv", ["dashboard.py", "--output", str(output)])
        app = AppTest.from_file(str(ROOT / "dashboard.py")).run(timeout=30)
        assert not app.exception and not app.error
        assert app.title[0].value == "电信客户流失分析与用户排序"
        assert {tab.label for tab in app.tabs} == {"客户差异", "模型验证", "触达排序"}
        assert app.dataframe[0].value.columns[0] == "客户数"


@pytest.fixture(scope="module")
def report_project():
    """独立的小型项目副本，覆盖旧报告与当前报告共存的启动情形。"""
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        root = Path(directory)
        shutil.copyfile(ROOT / "dashboard.py", root / "dashboard.py")
        current = root / "outputs" / "current"
        run_pipeline(sample_data(), current)
        legacy = root / "outputs" / "run_01"
        legacy.mkdir()
        shutil.copyfile(current / "metrics.json", legacy / "metrics.json")
        yield root


@pytest.mark.parametrize("arguments", [[], ["--output", "outputs/run_01"], ["--output=outputs/current"]])
def test_dashboard_selects_complete_report(report_project, monkeypatch, arguments):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(sys, "argv", ["dashboard.py", *arguments])
    monkeypatch.chdir(ROOT.parent)
    app = AppTest.from_file(str(report_project / "dashboard.py")).run(timeout=30)
    assert not app.exception and not app.error
    assert Path(app.selectbox[0].value) == report_project / "outputs" / "current"
    assert len(app.dataframe) == 6
    if "outputs/run_01" in arguments:
        assert any("不包含完整" in item.value for item in app.info)
    assert "实习" not in app.caption[0].value
    assert "学习版" not in " ".join(item.value for item in app.caption)


def test_dashboard_without_reports_gives_generation_command(monkeypatch):
    from streamlit.testing.v1 import AppTest

    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        script = Path(directory) / "dashboard.py"
        shutil.copyfile(ROOT / "dashboard.py", script)
        monkeypatch.setattr(sys, "argv", ["dashboard.py", "--output", "outputs/run_01"])
        app = AppTest.from_file(str(script)).run(timeout=30)
        assert not app.exception
        assert "未找到完整的项目报告" in app.error[0].value
        assert "python run_project.py" in app.code[0].value


def test_dashboard_rejects_missing_output_argument(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(sys, "argv", ["dashboard.py", "--output"])
    app = AppTest.from_file(str(ROOT / "dashboard.py")).run(timeout=30)
    assert not app.exception
    assert "结果目录参数有误" in app.error[0].value
