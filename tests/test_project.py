import json
import os
import shutil
import sys
import tempfile
from io import BytesIO
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from telco_churn import pipeline, validation
from optional_ai import assistant as ai_assistant
from telco_churn.pipeline import (
    analyze_contract_tenure, analyze_segment_associations, analyze_segments,
    evaluate_policy, holm_adjust, prepare_data, run_pipeline as _run_pipeline,
    select_f1_threshold, wilson_interval,
)


def run_pipeline(*args, **kwargs):
    """保留真实五折嵌套训练与报告，仅压缩测试运行预算。"""
    options = {
        "outer_splits": 5, "repeats": 1, "inner_splits": 2, "trials": 1,
        "n_bootstrap": 20, "backend": "sqlite",
    }
    options.update(kwargs)
    return _run_pipeline(*args, **options)


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


def use_fast_models(monkeypatch):
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    def tune(name, X, y, numeric_columns, seed, trials, inner_splits):
        # 真实预处理、内层 sigmoid 校准及外层验证照常运行；只跳过调参搜索。
        if name == "LightGBM":
            estimator = LGBMClassifier(
                n_estimators=8, max_depth=3, learning_rate=0.1, num_leaves=8,
                min_child_samples=2, random_state=seed, n_jobs=1, verbosity=-1,
            )
        elif name == "随机森林":
            estimator = RandomForestClassifier(
                n_estimators=8, max_depth=3, min_samples_leaf=2,
                random_state=seed, n_jobs=1,
            )
        else:
            assert name == "逻辑回归"
            estimator = LogisticRegression(max_iter=2000, random_state=seed)
        return pipeline.make_pipeline(estimator, numeric_columns)

    monkeypatch.setattr(validation, "tune_model", tune)


@pytest.fixture(autouse=True)
def fast_models_for_pipeline_tests(monkeypatch):
    use_fast_models(monkeypatch)


def test_prepare_data_converts_blank_total_charges():
    frame = sample_data()
    frame["TotalCharges"] = frame["TotalCharges"].astype(object)
    frame.loc[0, "TotalCharges"] = " "
    X, y, ids = prepare_data(frame)
    assert X.TotalCharges.isna().sum() == 1
    assert y.nunique() == 2
    assert ids.is_unique


def test_tenure_zero_total_charges_zero_filled():
    frame = sample_data()
    frame.loc[0, "tenure"] = 0
    frame["TotalCharges"] = frame["TotalCharges"].astype(object)
    frame.loc[0, "TotalCharges"] = " "
    X, _, _ = prepare_data(frame)
    assert X.loc[0, "TotalCharges"] == 0.0


def test_feature_label_no_spurious_match():
    from dashboard import feature_label
    assert "contract_month" not in feature_label("numeric__tenure_contract_month")
    assert feature_label("numeric__tenure") == "在网时长（月）"


def test_tune_lightgbm_integration():
    from lightgbm import LGBMClassifier
    from telco_churn.lightgbm_tuning import tune_lightgbm

    frame = sample_data(80)
    X, y, _ = prepare_data(frame)
    X = pipeline.engineer_features(X)
    params, _, _ = tune_lightgbm(
        X, y.to_numpy(), n_trials=2, n_jobs=1, n_splits=2, random_state=0,
        estimator_factory=lambda values: pipeline.make_pipeline(
            LGBMClassifier(**values), pipeline.ENGINEERED_NUMERIC),
    )
    model = pipeline.make_pipeline(LGBMClassifier(**params), pipeline.ENGINEERED_NUMERIC)
    model.fit(X, y)
    assert model.predict_proba(X).shape == (len(X), 2)


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
    use_fast_models(monkeypatch)
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        output = Path(directory) / "outputs"
        captured = []
        original_analyze = pipeline.analyze_training_segments

        def capture_training_segments(training_frame, backend="sqlite", include_pareto=False):
            captured.append((set(training_frame.customerID), backend))
            return original_analyze(training_frame, backend=backend, include_pareto=include_pareto)

        monkeypatch.setattr(pipeline, "analyze_training_segments", capture_training_segments)
        metrics = run_pipeline(sample_data(), output)
        assert set(metrics["模型"]) == {"逻辑回归", "随机森林", "LightGBM"}
        assert metrics["交叉验证"]["折数"] == 5
        assert metrics["交叉验证"]["重复次数"] == 1
        assert metrics["交叉验证"]["内层折数"] == 2
        assert metrics["交叉验证"]["Optuna轮数"] == 1
        assert all(len(scores) == 5 for scores in metrics["交叉验证"]["各折AP"].values())
        assert metrics["交叉验证"]["选择模型"] in metrics["模型"]
        assert (output / "metrics.json").is_file()
        predictions = pd.read_csv(output / "predictions.csv", dtype={"customerID": "string"})
        assert len(predictions) == 24
        assert predictions.customerID.isin(sample_data().customerID).all()
        assert len(captured) == 1
        training_ids, backend = captured[0]
        assert backend == "sqlite"
        assert len(training_ids) == metrics["训练集样本数"]
        assert training_ids.isdisjoint(set(predictions.customerID))
        assignments = pd.read_csv(output / "policy_assignments.csv", dtype={"customerID": "string"})
        assert assignments.groupby("策略").size().le(int(len(predictions) * 0.1)).all()
        assert len(pd.read_csv(output / "policy_metrics.csv")) == 5
        ai_context = pd.read_csv(output / "ai_customer_context.csv")
        assert len(ai_context) == int(len(predictions) * 0.1)
        assert "actual_churn" not in ai_context.columns
        assert ai_context["model_name"].iloc[0] == metrics["交叉验证"]["选择模型"]
        segments = pd.read_csv(output / "segment_summary.csv")
        assert segments["客户数"].sum() >= metrics["训练集样本数"]
        assert (output / "overall_summary.csv").is_file()
        assert (output / "segment_inference.csv").is_file()
        assert (output / "contract_tenure_segments.csv").is_file()
        assert (output / "segment_pareto.csv").is_file()
        assert (output / "model_comparison_test.csv").is_file()
        assert (output / "logistic_coefficients.csv").is_file()
        assert (output / "shap_summary.png").is_file()
        assert metrics["生存分析"]["状态"] == "默认禁用"
        assert not (output / "survival_curves.png").exists()
        assert not (output / "cox_results.csv").exists()
        folds = pd.read_csv(output / "nested_cv_folds.csv")
        assert len(folds) == 5 * 3
        assert folds.groupby("模型")["外层折"].nunique().eq(5).all()
        assert {"原始AP", "AP", "原始Brier", "校准Brier"}.issubset(folds.columns)
        assert np.isfinite(folds[["原始AP", "AP", "原始Brier", "校准Brier"]]).all().all()
        ablation = pd.read_csv(output / "ablation_folds.csv")
        assert len(ablation) == 5 * len(pipeline.FEATURE_GROUPS)
        assert set(ablation["特征组"]) == set(pipeline.FEATURE_GROUPS)
        assert ablation.groupby("特征组")["外层折"].nunique().eq(5).all()
        full_features = ablation.loc[ablation["特征组"].eq("全部衍生")].sort_values("外层折")
        logistic_folds = folds.loc[folds["模型"].eq("逻辑回归")].sort_values("外层折")
        np.testing.assert_allclose(full_features["AP"], logistic_folds["AP"])
        ablation_summary = pd.read_csv(output / "ablation_summary.csv")
        assert set(ablation_summary["特征组"]) == set(pipeline.FEATURE_GROUPS)
        calibration = pd.read_csv(output / "calibration_comparison.csv")
        assert set(zip(calibration["模型"], calibration["状态"])) == set(
            product(metrics["模型"], ["原始", "sigmoid"]))
        assert np.isfinite(calibration[["AP", "Brier"]]).all().all()
        for name in metrics["模型"]:
            state = "sigmoid" if metrics["校准决策"][name]["使用校准模型"] else "原始"
            row = calibration.loc[
                calibration["模型"].eq(name) & calibration["状态"].eq(state)
            ].iloc[0]
            assert row["AP"] == pytest.approx(metrics["模型"][name]["AP（平均精确率）"])
            assert row["Brier"] == pytest.approx(metrics["模型"][name]["Brier"])
        calibration_cv = pd.read_csv(output / "calibration_cv.csv").set_index("模型")
        pd.testing.assert_frame_equal(
            calibration_cv.sort_index(),
            folds.groupby("模型")[["原始Brier", "校准Brier", "原始AP", "AP"]].mean().sort_index(),
        )
        for name in ("calibration_pr.png", "calibration_bins.csv", "pr_curve.csv"):
            assert (output / name).is_file()
        training_oof = pd.read_csv(output / "training_oof.csv", dtype={"customerID": "string"})
        assert set(training_oof.customerID) == training_ids
        assert not training_oof.customerID.duplicated().any()
        scenarios = pd.read_csv(output / "business_scenarios.csv")
        assert len(scenarios) == 27
        assert set(scenarios[["触达成本", "假设成功率", "假设剩余月数"]].itertuples(
            index=False, name=None)) == set(product([20, 50, 100], [0.1, 0.2, 0.3], [6, 12, 24]))
        assert scenarios["训练OOF选择K"].between(0, 1).all()
        assert np.isfinite(scenarios["训练OOF情景净收益"]).all()
        training_budget = pd.read_csv(output / "budget_training_oof.csv")
        selected_k = training_budget.sort_values(
            ["期望净收益", "K比例"], ascending=[False, True]).iloc[0]["K比例"]
        assert metrics["名额敏感性"]["训练OOF选择名额比例"] == pytest.approx(selected_k)
        assert (output / "budget_sensitivity.csv").is_file()
        assert (output / "budget_curve.png").is_file()
        assert (output / "confusion_matrices.png").is_file()
        assert (output / "roc_curve.png").is_file()
        assert (output / "feature_importance.csv").is_file()
        saved = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        assert saved["测试集样本数"] == 24
        assert set(saved["阈值评估"]) == {"逻辑回归", "随机森林", "LightGBM"}
        assert "名额敏感性" in saved
        assert all("阈值测试集F1" in result for result in saved["模型"].values())
        segments = pd.read_csv(output / "segment_summary.csv")
        assert {"流失率下限", "流失率上限", "流失率排名"}.issubset(segments.columns)


def test_sql_segments_match_pandas():
    frame = sample_data(150)
    frame["Churn"] = frame["Churn"].map({"No": 0, "Yes": 1})
    _, segments = analyze_segments(frame)
    contract = segments.loc[segments["分群维度"].eq("合同类型")]
    assert "流失率排名" in segments.columns
    expected = frame.groupby("Contract").Churn.sum()
    actual = contract.set_index("分群")["流失人数"].sort_index()
    pd.testing.assert_series_equal(actual, expected.astype(int).sort_index(), check_names=False)


def test_wilson_interval_boundaries_and_single_observation():
    lower, upper = wilson_interval(0, 10)
    assert lower == 0 and 0 < upper < 1
    lower, upper = wilson_interval(10, 10)
    assert 0 < lower < 1 and upper == pytest.approx(1)
    assert all(np.isnan(value) for value in wilson_interval(0, 0))


def test_holm_adjustment_preserves_order_and_monotonicity():
    adjusted = holm_adjust([0.03, 0.01, np.nan, 0.04])
    assert adjusted[1] == pytest.approx(0.03)
    assert adjusted[0] == pytest.approx(0.06)
    assert np.isnan(adjusted[2])
    assert adjusted[3] == pytest.approx(0.06)


def test_segment_association_skips_sparse_expected_counts():
    frame = pd.DataFrame({
        "Contract": ["A"] * 40 + ["B"] * 40,
        "Churn": [0] * 20 + [1] * 20 + [0] * 20 + [1] * 20,
        "rare": ["small"] * 3 + ["large"] * 77,
    })
    result = analyze_segment_associations(frame, {"合同": "Contract", "稀疏维度": "rare"})
    assert result.loc[0, "检验状态"] == "已检验"
    assert result.loc[1, "检验状态"] == "期望频数小于5，未报告检验"
    assert result.loc[1, "p值"] != result.loc[1, "p值"]


def test_cross_segments_use_sql_and_include_wilson_intervals():
    frame = sample_data(180)
    frame["Churn"] = frame["Churn"].map({"No": 0, "Yes": 1})
    result = analyze_contract_tenure(frame)
    assert {"合同类型", "在网时长分组", "客户数", "流失率", "流失率下限", "流失率上限"}.issubset(result.columns)
    assert result["客户数"].sum() == len(frame)


def test_f1_threshold_uses_oof_inputs_and_resolves_ties_near_half():
    labels = np.array([0, 1, 1, 0])
    probabilities = np.array([0.2, 0.4, 0.6, 0.8])
    threshold = select_f1_threshold(labels, probabilities)
    assert threshold in probabilities or threshold in (0.0, 0.5, 1.0)
    assert threshold == select_f1_threshold(labels.copy(), probabilities.copy())


def test_selected_threshold_does_not_depend_on_test_labels(monkeypatch):
    use_fast_models(monkeypatch)
    frame = sample_data(150)
    split = (np.arange(120), np.arange(120, 150))
    monkeypatch.setattr(pipeline, "train_test_split", lambda *args, **kwargs: split)
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        first = Path(directory) / "first"
        second = Path(directory) / "second"
        original = run_pipeline(frame.copy(), first)
        changed = frame.copy()
        changed.loc[120:, "Churn"] = changed.loc[120:, "Churn"].map({"Yes": "No", "No": "Yes"})
        modified = run_pipeline(changed, second)
    assert original["交叉验证"]["选择模型"] == modified["交叉验证"]["选择模型"]
    assert original["交叉验证"]["各折AP"] == modified["交叉验证"]["各折AP"]
    assert original["名额敏感性"]["训练OOF选择名额比例"] == pytest.approx(
        modified["名额敏感性"]["训练OOF选择名额比例"]
    )
    for model in original["阈值评估"]:
        assert original["阈值评估"][model]["折外F1阈值"] == pytest.approx(
            modified["阈值评估"][model]["折外F1阈值"]
        )
        assert original["阈值评估"][model]["折外F1"] == pytest.approx(
            modified["阈值评估"][model]["折外F1"]
        )


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
        assert app.title[0].value == "电信客户流失预测与排序"
        assert app.sidebar.radio[0].value == "报告总览"
        assert not app.main.radio
        assert app.sidebar.radio[0].options == ["报告总览", "客户差异", "模型验证", "模型解释", "触达排序", "AI风险解读"]
        labels = [metric.label for metric in app.metric]
        assert labels == ["客户总量", "历史留出 AP", "优先名单命中率", "相对随机提升倍数"]
        assert app.expander[0].label == "数据来源与评估口径"
        app.sidebar.radio[0].set_value("模型验证").run(timeout=30)
        assert not app.exception and not app.error
        app.sidebar.radio[0].set_value("模型解释").run(timeout=30)
        assert not app.exception and not app.error
        app.sidebar.radio[0].set_value("触达排序").run(timeout=30)
        assert not app.exception and not app.error


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
    patch = pytest.MonkeyPatch()
    use_fast_models(patch)
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        root = Path(directory)
        shutil.copyfile(ROOT / "dashboard.py", root / "dashboard.py")
        current = root / "outputs" / "current"
        run_pipeline(sample_data(), current)
        legacy = root / "outputs" / "run_01"
        legacy.mkdir()
        shutil.copyfile(current / "metrics.json", legacy / "metrics.json")
        legacy_full = root / "outputs" / "legacy_full"
        shutil.copytree(current, legacy_full)
        (legacy_full / "segment_inference.csv").unlink()
        (legacy_full / "contract_tenure_segments.csv").unlink()
        legacy_metrics = json.loads((legacy_full / "metrics.json").read_text(encoding="utf-8"))
        legacy_metrics.pop("报告版本")
        legacy_metrics.pop("阈值评估")
        for model_metrics in legacy_metrics["模型"].values():
            for key in ("折外F1阈值", "阈值测试集查准率", "阈值测试集查全率", "阈值测试集F1"):
                model_metrics.pop(key)
        (legacy_full / "metrics.json").write_text(
            json.dumps(legacy_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        current_metrics = current / "metrics.json"
        current_stat = current_metrics.stat()
        os.utime(current_metrics, ns=(current_stat.st_atime_ns, current_stat.st_mtime_ns + 10_000_000_000))
        yield root
    patch.undo()


@pytest.mark.parametrize("arguments", [[], ["--output", "outputs/run_01"], ["--output=outputs/current"]])
def test_dashboard_selects_complete_report(report_project, monkeypatch, arguments):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(sys, "argv", ["dashboard.py", *arguments])
    monkeypatch.chdir(ROOT.parent)
    app = AppTest.from_file(str(report_project / "dashboard.py")).run(timeout=30)
    assert not app.exception and not app.error
    report_selector = next(item for item in app.selectbox if item.label == "分析报告")
    assert Path(report_selector.value) == report_project / "outputs" / "current"
    assert app.radio[0].options == ["报告总览", "客户差异", "模型验证", "模型解释", "触达排序", "AI风险解读"]
    if "outputs/run_01" in arguments:
        assert any("不包含完整" in item.value for item in app.info)
    assert "实习" not in app.caption[0].value
    assert "学习版" not in " ".join(item.value for item in app.caption)


def test_dashboard_pages_show_only_their_own_content(report_project, monkeypatch):
    from streamlit.testing.v1 import AppTest

    report = report_project / "outputs" / "current"
    metrics = json.loads((report / "metrics.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(report / "overall_summary.csv").iloc[0]
    model = metrics["交叉验证"]["选择模型"]
    policy = pd.read_csv(report / "policy_metrics.csv")
    selected_policy = policy.loc[policy["策略"].eq(model)].iloc[0]
    monkeypatch.setattr(sys, "argv", ["dashboard.py"])
    app = AppTest.from_file(str(report_project / "dashboard.py")).run(timeout=30)
    assert app.sidebar.radio[0].value == "报告总览"
    pages = [
        ("报告总览", "电信客户流失预测与排序", ["客户总量", "历史留出 AP", "优先名单命中率", "相对随机提升倍数"], None),
        ("客户差异", "训练集客户分群", ["训练集客户数", "训练集流失人数", "训练集流失率"], "选择分群维度"),
        ("模型验证", "模型验证", ["主模型", "外层 AP 均值", "留出集 AP", "留出集 ROC-AUC"], None),
        ("模型解释", "模型解释", [], "查看高风险客户解释"),
        ("触达排序", "固定 10% 名额下的策略对照", ["优先触达名额", "命中历史流失人数", "优先名单命中率", "相对随机提升倍数"], "查看策略的历史留出集示例"),
        ("AI风险解读", "AI 风险解读与触达建议", [], "选择主模型固定容量优先客户"),
    ]
    for page, title, metric_labels, selector in pages:
        app.sidebar.radio[0].set_value(page).run(timeout=30)
        assert not app.exception and not app.error
        assert [item.value for item in app.title] == [title]
        assert [item.label for item in app.metric] == metric_labels
        assert [item.label for item in app.main.selectbox] == ([selector] if selector else [])
        assert Path(app.sidebar.selectbox[0].value) == report
        expected_expanders = {
            "报告总览": ["数据来源与评估口径"], "AI风险解读": ["客户资料与分群统计"],
        }
        assert [item.label for item in app.expander] == expected_expanders.get(page, [])
        rendered = "\n".join(item.value for item in app.main.markdown)
        assert ('class="status-strip"' in rendered) == (page == "报告总览")
        assert ("分析客户流失差异" in rendered) == (page == "报告总览")
        assert "指标怎么读" not in [item.value for item in app.subheader]
        assert "报告目录" not in rendered and str(report) not in rendered
        if page == "客户差异":
            assert app.metric[0].value == f"{int(summary['客户数']):,}"
            assert app.metric[2].value == f"{summary['流失率']:.1%}"
            assert "分群与历史流失的关联检验" in [item.value for item in app.subheader]
            assert "合同类型 × 在网时长" in [item.value for item in app.subheader]
        elif page == "模型验证":
            assert app.metric[0].value == metrics["交叉验证"]["选择模型"]
            assert "历史留出集指标" in [item.value for item in app.subheader]
            assert "历史留出集 Bootstrap 区间" in [item.value for item in app.subheader]
            assert any("条件于已拟合模型" in item.value for item in app.caption)
            assert "训练集折外阈值对照" in [item.value for item in app.subheader]
        elif page == "报告总览":
            assert app.metric[0].value == f"{metrics['样本数']:,}"
            assert app.metric[1].value == f"{metrics['模型'][model]['AP（平均精确率）']:.3f}"
            assert app.metric[2].value == f"{selected_policy['Precision@K']:.1%}"
            assert app.metric[3].value == f"{selected_policy['Lift@K']:.2f}×"
            assert model in rendered and "固定触达容量" in rendered
            assert "不是独立盲测" in rendered
            assert str(int(selected_policy["命中流失"])) in app.info[0].value
    app.sidebar.radio[0].set_value("客户差异").run(timeout=30)
    assert not app.exception and app.title[0].value == "训练集客户分群"
    app.sidebar.radio[0].set_value("报告总览").run(timeout=30)
    assert not app.exception and app.title[0].value == "电信客户流失预测与排序"


def test_dashboard_legacy_report_explains_missing_analysis(report_project, monkeypatch):
    from streamlit.testing.v1 import AppTest

    legacy = report_project / "outputs" / "legacy_full"
    monkeypatch.setattr(sys, "argv", ["dashboard.py", "--output", str(legacy)])
    app = AppTest.from_file(str(report_project / "dashboard.py")).run(timeout=30)
    app.sidebar.radio[0].set_value("客户差异").run(timeout=30)
    assert not app.exception
    assert any("未包含统计检验结果" in item.value for item in app.info)
    assert any("未包含合同与在网时长交叉分群" in item.value for item in app.info)
    app.sidebar.radio[0].set_value("模型验证").run(timeout=30)
    assert not app.exception
    assert "外层 AP 均值" in [item.label for item in app.metric]
    assert any("未包含折外阈值评估" in item.value for item in app.info)


def test_ai_explanation_survives_navigation_and_stays_with_customer(report_project, monkeypatch):
    from streamlit.testing.v1 import AppTest

    current = report_project / "outputs" / "current"
    calls = []

    def generate(context, settings=None):
        calls.append(context["customerID"])
        return {
            "mode": "模型生成", "summary": f"客户 {context['customerID']} 的第 {len(calls)} 次解读",
            "evidence": ["在网时间较短"], "actions": ["人工核实服务情况"],
            "limitations": "分数只用于排序",
        }

    def summaries(app):
        return [item.value for item in app.markdown if 'class="ai-analysis-text"' in item.value]

    monkeypatch.setattr(ai_assistant, "generate_explanation", generate)
    monkeypatch.setattr(sys, "argv", ["dashboard.py", "--output", str(current)])
    with tempfile.TemporaryDirectory(dir=report_project / "outputs") as directory:
        other = Path(directory)
        shutil.copytree(current, other, dirs_exist_ok=True)
        app = AppTest.from_file(str(report_project / "dashboard.py")).run(timeout=30)
        app.sidebar.radio[0].set_value("AI风险解读").run(timeout=30)
        first, second = app.main.selectbox[0].options[:2]
        app.main.selectbox[0].set_value(second).run(timeout=30)
        app.button[0].click().run(timeout=30)
        original = summaries(app)
        assert original and calls == [second]

        for page in ("报告总览", "客户差异", "模型验证", "触达排序"):
            app.sidebar.radio[0].set_value(page).run(timeout=30)
            app.sidebar.radio[0].set_value("AI风险解读").run(timeout=30)
            assert not app.exception
            assert app.main.selectbox[0].value == second
            assert summaries(app) == original
            assert calls == [second]

        app.main.selectbox[0].set_value(first).run(timeout=30)
        assert not summaries(app) and calls == [second]
        app.button[0].click().run(timeout=30)
        first_result = summaries(app)
        assert first_result != original and calls == [second, first]

        app.sidebar.selectbox[0].set_value(other).run(timeout=30)
        assert not app.exception and not summaries(app)
        app.button[0].click().run(timeout=30)
        assert calls == [second, first, first]
        assert summaries(app) != first_result

        context_file = other / "ai_customer_context.csv"
        context = pd.read_csv(context_file)
        context.loc[context["customerID"].eq(first), "risk_evidence"] = "更新后的客户资料"
        context.to_csv(context_file, index=False)
        app.run(timeout=30)
        assert not app.exception and not summaries(app)
        assert len(calls) == 3

        app.sidebar.selectbox[0].set_value(current).run(timeout=30)
        assert summaries(app) == first_result
        app.main.selectbox[0].set_value(second).run(timeout=30)
        assert summaries(app) == original and len(calls) == 3
        app.button[0].click().run(timeout=30)
        assert not app.exception and summaries(app) != original
        assert calls == [second, first, first, second]


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


def test_ai_response_formats_paragraph_lists():
    content = {
        "summary": ["客户使用月付合同。", "联系前需核实服务情况。"],
        "evidence": ["在网时间为一个月。"],
        "actions": ["核对合同与账单。"],
        "limitations": ["分数只用于排序。", "分群差异不是因果结论。"],
    }
    payload = {"choices": [{"message": {"content": json.dumps(content)}}]}
    result = ai_assistant._parse_model_response(payload)
    assert result["summary"] == "客户使用月付合同。\n联系前需核实服务情况。"
    assert result["limitations"] == "分数只用于排序。\n分群差异不是因果结论。"
    assert result["mode"] == "模型生成"


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


def test_legacy_report_context_preserves_ranking_and_excludes_labels():
    assignments = [
        {"策略": "逻辑回归", "customerID": "C2", "排名": 2, "排序分数": 0.8, "actual_churn": 1},
        {"策略": "随机森林", "customerID": "C3", "排名": 1, "排序分数": 0.99},
        {"策略": "逻辑回归", "customerID": "C1", "排名": 1, "排序分数": 0.9, "actual_churn": 0},
    ]
    predictions = [
        {"customerID": "C1", "tenure": 5, "actual_churn": "不能发送的标签"},
        {"customerID": "C2", "tenure": 9, "actual_churn": "不能发送的标签"},
    ]
    context = ai_assistant.context_from_report(assignments, predictions, "逻辑回归")
    assert [row["customerID"] for row in context] == ["C1", "C2"]
    assert [row["risk_score"] for row in context] == [0.9, 0.8]
    assert [row["tenure"] for row in context] == [5, 9]
    assert all("actual_churn" not in row and "Contract" not in row for row in context)
    assert "不能发送的标签" not in ai_assistant.build_prompt(context[0])


def test_dashboard_legacy_report_ai_uses_cloud_secrets(report_project, monkeypatch):
    from streamlit.testing.v1 import AppTest

    for key in ("CHURN_AI_BASE_URL", "CHURN_AI_MODEL", "CHURN_AI_API_KEY", "CHURN_AI_TIMEOUT_SECONDS"):
        monkeypatch.delenv(key, raising=False)
    explanation = {
        "summary": "这是远程接口返回的解读。", "evidence": ["排序分数较高"],
        "actions": ["人工核实客户需求。"], "limitations": "排序分数未经校准。",
    }
    payload = json.dumps({"choices": [{"message": {"content": json.dumps(explanation)}}]}).encode()
    requests = []

    def respond(request, timeout):
        assert request.full_url == "https://example.test/v1/chat/completions"
        assert request.get_header("Authorization") == "Bearer cloud-test-key"
        body = json.loads(request.data)
        assert body["model"] == "cloud-test-model"
        assert "actual_churn" not in body["messages"][1]["content"]
        requests.append(body)
        return BytesIO(payload)

    monkeypatch.setattr(ai_assistant.urllib_request, "urlopen", respond)
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        report = Path(directory) / "legacy"
        shutil.copytree(report_project / "outputs" / "current", report,
                        ignore=shutil.ignore_patterns("ai_customer_context.csv"))
        monkeypatch.setattr(sys, "argv", ["dashboard.py", "--output", str(report)])
        app = AppTest.from_file(str(ROOT / "dashboard.py"))
        app.secrets.update({
            "CHURN_AI_BASE_URL": "https://example.test/v1",
            "CHURN_AI_MODEL": "cloud-test-model",
            "CHURN_AI_API_KEY": "cloud-test-key",
        })
        app.run(timeout=30)
        app.sidebar.radio[0].set_value("AI风险解读").run(timeout=30)
        assert not app.exception and app.button
        app.button[0].click().run(timeout=30)
        assert not app.exception and len(requests) == 1
        assert any("模型生成" in item.value for item in app.caption)
        assert any(explanation["summary"] in item.value for item in app.markdown)
        assert not (report / "ai_customer_context.csv").exists()


def test_chinese_chart_font_without_system_fonts(monkeypatch):
    import warnings
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from telco_churn.chart_font import configure_chart_font

    monkeypatch.setattr(font_manager.fontManager, "ttflist", [])
    font_manager.fontManager._findfont_cached.cache_clear()
    family = configure_chart_font()
    path = font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
    assert Path(path).is_relative_to(ROOT / "assets" / "fonts")
    characters = font_manager.get_font(path).get_charmap()
    assert all(ord(character) in characters for character in "客户流失率（月）逻辑回归随机森林")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Glyph .*missing from font")
        figure, axis = plt.subplots()
        try:
            axis.set_title("客户流失率（月）")
            figure.savefig(BytesIO(), format="png")
        finally:
            plt.close(figure)
    font_manager.fontManager._findfont_cached.cache_clear()


def test_deployment_demo_includes_matching_ai_context():
    report = ROOT / "outputs" / "run_final"
    metrics = json.loads((report / "metrics.json").read_text(encoding="utf-8"))
    model = metrics["交叉验证"]["选择模型"]
    context = pd.read_csv(report / "ai_customer_context.csv", dtype={"customerID": "string"})
    assignments = pd.read_csv(report / "policy_assignments.csv", dtype={"customerID": "string"})
    selected = assignments.loc[assignments["策略"].eq(model)].sort_values("排名")
    assert context["customerID"].tolist() == selected["customerID"].tolist()
    score_column = "排序分数" if "排序分数" in selected else "预测流失风险分数"
    np.testing.assert_allclose(context["risk_score"], selected[score_column])
    assert context[["Contract", "MonthlyCharges", "tenure"]].notna().all().all()
    assert "actual_churn" not in context.columns and "Churn" not in context.columns
