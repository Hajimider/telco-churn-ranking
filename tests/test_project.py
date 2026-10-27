import json
import os
import shutil
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import churn_pipeline as pipeline
import ai_assistant
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
        ai_context = pd.read_csv(output / "ai_customer_context.csv")
        assert len(ai_context) == int(len(predictions) * 0.1)
        assert "actual_churn" not in ai_context.columns
        assert ai_context["model_name"].iloc[0] == metrics["交叉验证"]["选择模型"]
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
        assert not app.main.radio
        assert app.sidebar.radio[0].options == ["客户差异", "模型验证", "触达排序", "AI风险解读"]
        labels = [metric.label for metric in app.metric]
        assert "优先名单命中率" in labels and "相对随机提升倍数" in labels
        assert any("客户数" in item.value and "流失率" in item.value for item in app.markdown)


def test_dashboard_ai_button_renders_offline_explanation(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("CHURN_AI_BASE_URL", "")
    monkeypatch.setenv("CHURN_AI_MODEL", "")
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        output = Path(directory) / "outputs"
        run_pipeline(sample_data(), output)
        monkeypatch.setattr(sys, "argv", ["dashboard.py", "--output", str(output)])
        app = AppTest.from_file(str(ROOT / "dashboard.py")).run(timeout=30)
        app.sidebar.radio[0].set_value("AI风险解读").run(timeout=30)
        assert not app.exception and app.button[0].label == "生成 AI 风险解读"
        app.button[0].click().run(timeout=30)
        assert not app.exception
        rendered = "\n".join(item.value for item in app.markdown)
        assert "风险摘要" in rendered and "风险证据" in rendered and "建议动作" in rendered
        assert '<ul class="ai-analysis-list"><li>' in rendered
        assert 'class="ai-analysis-text"' in rendered
        assert any("离线模板模式" in item.value for item in app.caption)


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
    report_selector = next(item for item in app.selectbox if item.label == "分析报告")
    assert Path(report_selector.value) == report_project / "outputs" / "current"
    assert app.radio[0].options == ["客户差异", "模型验证", "触达排序", "AI风险解读"]
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


def test_ai_prompt_and_template_exclude_evaluation_label(monkeypatch):
    context = {
        "customerID": "C001",
        "model_name": "逻辑回归",
        "rank": 1,
        "risk_score": 0.82,
        "Contract": "Month-to-month",
        "TechSupport": "No",
        "actual_churn": "已流失标签不应出现",
    }
    prompt = ai_assistant.build_prompt(context)
    assert "actual_churn" not in prompt
    assert "已流失标签不应出现" not in prompt
    monkeypatch.setenv("CHURN_AI_BASE_URL", "")
    monkeypatch.setenv("CHURN_AI_MODEL", "")
    result = ai_assistant.generate_explanation(context)
    assert result["mode"] == "离线模板模式"
    assert result["evidence"]
    assert result["actions"]


def test_ai_project_env_loader_preserves_existing_environment(monkeypatch):
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        env_file = Path(directory) / ".env"
        env_file.write_text(
            'CHURN_AI_BASE_URL="https://example.test/v1"\n'
            "CHURN_AI_MODEL=demo-model\n"
            "CHURN_AI_API_KEY=from-file\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("CHURN_AI_BASE_URL", raising=False)
        monkeypatch.delenv("CHURN_AI_MODEL", raising=False)
        monkeypatch.setenv("CHURN_AI_API_KEY", "from-environment")
        ai_assistant._load_project_env(env_file)
        assert os.environ["CHURN_AI_BASE_URL"] == "https://example.test/v1"
        assert os.environ["CHURN_AI_MODEL"] == "demo-model"
        assert os.environ["CHURN_AI_API_KEY"] == "from-environment"


def test_ai_remote_failure_falls_back_to_template(monkeypatch):
    monkeypatch.setenv("CHURN_AI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("CHURN_AI_MODEL", "test-model")
    monkeypatch.setenv("CHURN_AI_TIMEOUT_SECONDS", "1")

    def fail(*args, **kwargs):
        raise ai_assistant.urllib_error.URLError("offline")

    monkeypatch.setattr(ai_assistant.urllib_request, "urlopen", fail)
    result = ai_assistant.generate_explanation({"risk_score": 0.8, "rank": 1})
    assert result["mode"] == "远程接口不可用，已回退模板"


@pytest.mark.parametrize("content", ['{"summary":"summary only"}', "这是一段普通文本"])
def test_ai_incomplete_remote_response_falls_back_to_template(monkeypatch, content):
    monkeypatch.setenv("CHURN_AI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("CHURN_AI_MODEL", "test-model")

    payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")
    monkeypatch.setattr(ai_assistant.urllib_request, "urlopen", lambda *args, **kwargs: BytesIO(payload))
    result = ai_assistant.generate_explanation({"risk_score": 0.8, "rank": 1})
    assert result["mode"] == "远程接口不可用，已回退模板"


@pytest.mark.parametrize("fenced", [False, True])
def test_ai_remote_explanation_preserves_fields_and_excludes_label(monkeypatch, fenced):
    monkeypatch.setenv("CHURN_AI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("CHURN_AI_MODEL", "test-model")
    monkeypatch.setenv("CHURN_AI_API_KEY", "test-key")
    monkeypatch.setenv("CHURN_AI_TIMEOUT_SECONDS", "20")
    explanation = {
        "summary": "该客户的排序分数较高。",
        "evidence": ["在网时间较短"],
        "actions": ["人工确认近期服务体验。"],
        "limitations": "历史关联不代表因果关系。",
    }
    content = json.dumps(explanation, ensure_ascii=False)
    if fenced:
        content = f"```json\n{content}\n```"
    payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")

    def respond(request, timeout):
        assert request.full_url == "https://example.test/v1/chat/completions"
        assert request.get_header("Authorization") == "Bearer test-key"
        assert timeout == 20
        body = json.loads(request.data)
        assert body["model"] == "test-model"
        assert "actual_churn" not in body["messages"][1]["content"]
        assert "不应发送的真实标签" not in body["messages"][1]["content"]
        return BytesIO(payload)

    monkeypatch.setattr(ai_assistant.urllib_request, "urlopen", respond)
    result = ai_assistant.generate_explanation({
        "risk_score": 0.8, "rank": 1, "actual_churn": "不应发送的真实标签",
    })
    assert result == {"mode": "模型生成", **explanation}
