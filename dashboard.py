"""客户流失分析看板：按运行目录读取完整的离线报告。"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
REPORT_FILES = (
    "metrics.json", "overall_summary.csv", "segment_summary.csv", "policy_metrics.csv",
    "feature_importance.csv", "policy_assignments.csv", "confusion_matrices.png", "roc_curve.png",
)
FEATURE_LABELS = {
    "tenure": "在网时长（月）", "MonthlyCharges": "月费", "TotalCharges": "累计费用",
    "SeniorCitizen": "老年客户标记", "Contract": "合同类型", "InternetService": "互联网服务",
    "PaymentMethod": "付款方式", "OnlineSecurity": "在线安全服务", "OnlineBackup": "在线备份服务",
    "DeviceProtection": "设备保护服务", "TechSupport": "技术支持服务", "StreamingTV": "流媒体电视",
    "StreamingMovies": "流媒体电影", "PhoneService": "电话服务", "MultipleLines": "多线路服务",
    "Partner": "是否有伴侣", "Dependents": "是否有家属", "gender": "性别",
    "PaperlessBilling": "无纸化账单",
}
VALUE_LABELS = {
    "Month-to-month": "月付合同", "One year": "一年合同", "Two year": "两年合同",
    "Yes": "是", "No": "否", "Male": "男", "Female": "女",
    "Fiber optic": "光纤", "DSL": "数字用户线路（DSL）",
    "No internet service": "未开通互联网服务", "No phone service": "未开通电话服务",
    "Electronic check": "电子支票", "Mailed check": "邮寄支票",
    "Bank transfer (automatic)": "银行自动转账", "Credit card (automatic)": "信用卡自动扣款",
}


def output_dir():
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    parser.add_argument("--output", type=Path)
    args, _ = parser.parse_known_args(sys.argv[1:])
    if args.output is None:
        return None
    # 相对路径始终以项目目录为基准，避免从工作区启动时读错目录。
    return (ROOT / args.output).resolve()


def complete_report(path):
    return all((path / filename).is_file() for filename in REPORT_FILES)


def report_label(path):
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def feature_label(value):
    value = str(value)
    if "__" in value:
        _, value = value.split("__", 1)
    for name in sorted(FEATURE_LABELS, key=len, reverse=True):
        if value == name:
            return FEATURE_LABELS[name]
        if value.startswith(name + "_"):
            category = value[len(name) + 1:]
            return f"{FEATURE_LABELS[name]}：{VALUE_LABELS.get(category, category)}"
    return value


st.set_page_config(page_title="客户流失分析与用户排序", layout="wide")
st.title("电信客户流失分析与用户排序")
st.caption("客户分群洞察 · 流失风险建模 · 有限触达容量下的策略评估")

try:
    requested = output_dir()
except (argparse.ArgumentError, ValueError) as exc:
    st.error(f"结果目录参数有误：{exc}")
    st.code("python -m streamlit run dashboard.py", language="bash")
    st.stop()

outputs = ROOT / "outputs"
reports = [path for path in outputs.iterdir() if path.is_dir() and complete_report(path)] if outputs.is_dir() else []
if complete_report(outputs):
    reports.append(outputs)
if requested is not None and complete_report(requested) and requested not in reports:
    reports.append(requested)
reports.sort(key=lambda path: ((path / "metrics.json").stat().st_mtime_ns, str(path)), reverse=True)

if not reports:
    st.error("未找到完整的项目报告。请先运行以下命令生成报告，再刷新页面。")
    if requested is not None:
        missing = [name for name in REPORT_FILES if not (requested / name).is_file()]
        st.caption(f"指定目录 {report_label(requested)} 缺少：{'、'.join(missing)}")
    st.code("python run_project.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --output outputs/project_v1", language="bash")
    st.caption("请先将原始 CSV 放入 data/；若输出目录非空，请换一个新目录。看板只读取报告，不会自动训练。")
    st.stop()

default_index = reports.index(requested) if requested in reports else 0
path = st.selectbox("分析报告", reports, index=default_index, format_func=report_label)
if requested is not None and requested not in reports:
    st.info(f"指定目录 {report_label(requested)} 不包含完整的当前版报告；当前查看 {report_label(path)}。可在上方切换报告，原目录未修改。")

try:
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    if "交叉验证" not in metrics:
        raise ValueError("该报告缺少当前版交叉验证结果，请选择其他完整报告或重新生成")
    overall = pd.read_csv(path / "overall_summary.csv")
    segments = pd.read_csv(path / "segment_summary.csv")
    policy = pd.read_csv(path / "policy_metrics.csv")
    importance = pd.read_csv(path / "feature_importance.csv")
    assignments = pd.read_csv(path / "policy_assignments.csv", dtype={"customerID": "string"})
    if "预测流失风险分数" in assignments.columns and "排序分数" not in assignments.columns:
        assignments = assignments.rename(columns={"预测流失风险分数": "排序分数"})
except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError) as exc:
    st.error(f"无法读取项目报告：{exc}。请切换分析报告，或使用 run_project.py 生成新报告。")
    st.stop()

with st.container(horizontal=True):
    st.metric("客户数", f"{metrics['样本数']:,}", border=True)
    st.metric("训练集", f"{metrics['训练集样本数']:,}", border=True)
    st.metric("测试集", f"{metrics['测试集样本数']:,}", border=True)
    st.metric("样本流失率", f"{metrics['流失率']:.1%}", border=True)

with st.expander("数据来源与评估口径"):
    st.write("公开电信客户快照；分群使用训练集，模型依据训练集五折 AP 选择。测试结果为已参与开发分析的公开数据基准，不是独立盲测；历史流失识别不等于真实挽留收益。")
segments_tab, validation_tab, policy_tab = st.tabs(["客户差异", "模型验证", "触达排序"])

with segments_tab:
    st.subheader("训练集客户分群")
    overall_display = overall.copy()
    overall_display["流失率"] = overall_display["流失率"].map(lambda value: f"{value:.1%}")
    st.dataframe(overall_display, hide_index=True, width="stretch")
    dimension = st.selectbox("选择分群维度", segments["分群维度"].drop_duplicates().tolist())
    shown = segments.loc[segments["分群维度"].eq(dimension)].copy()
    contract_labels = {"Month-to-month": "月付合同", "One year": "一年合同", "Two year": "两年合同"}
    shown["分群"] = shown["分群"].replace(contract_labels)
    shown["流失率"] = shown["流失率"].map(lambda value: f"{value:.1%}")
    shown["小样本"] = shown["小样本"].map({True: "是", False: "否"})
    st.dataframe(shown, hide_index=True, width="stretch")
    chart = segments.loc[segments["分群维度"].eq(dimension), ["分群", "流失率"]].copy()
    chart["分群"] = chart["分群"].replace(contract_labels)
    chart = chart.set_index("分群")
    st.bar_chart(chart, y="流失率")
    st.caption("所有分群只使用训练集。差异是描述性关联，不说明合同或费用导致流失；客户数少于 30 的组应谨慎解读。")

with validation_tab:
    cv = metrics["交叉验证"]
    st.subheader("训练集五折交叉验证")
    st.caption(f"根据训练集五折 AP 均值选择模型：{cv['选择模型']}。测试集没有参与选择。")
    cv_frame = pd.DataFrame({
        "模型": list(cv["AP均值"]),
        "五折 AP 均值": list(cv["AP均值"].values()),
        "五折 AP 标准差": [cv["AP标准差"][name] for name in cv["AP均值"]],
    })
    st.dataframe(cv_frame, hide_index=True, width="stretch")
    fold_frame = pd.DataFrame(cv["各折AP"])
    fold_frame.index = [f"第 {index} 折" for index in range(1, len(fold_frame) + 1)]
    st.line_chart(fold_frame, y=list(fold_frame.columns))
    metric_frame = pd.DataFrame(metrics["模型"]).T.reset_index(names="模型")
    st.subheader("固定测试集指标")
    st.dataframe(metric_frame, hide_index=True, width="stretch")
    st.caption("AP（平均精确率）适合观察正类较少时的排序质量；Accuracy 不能单独代表模型好坏。分类阈值固定为 0.5。")
    left, right = st.columns(2)
    with left:
        st.image(str(path / "confusion_matrices.png"), caption="混淆矩阵：行是实际类别，列是预测类别")
    with right:
        st.image(str(path / "roc_curve.png"), caption="测试集 ROC 曲线")
    importance = importance.copy()
    importance["特征"] = importance.feature.map(feature_label)
    importance = importance.rename(columns={"importance": "重要性"})[["特征", "重要性"]].head(15)
    st.subheader("随机森林关注的特征")
    st.bar_chart(importance.set_index("特征"), y="重要性")
    st.caption("特征重要性描述模型使用方式，不代表因果关系。")

with policy_tab:
    st.subheader("固定 10% 名额下的策略对照")
    shown_policy = policy.copy()
    for column in ("Precision@K", "Recall@K"):
        shown_policy[column] = shown_policy[column].map(lambda value: f"{value:.1%}")
    shown_policy["Lift@K"] = shown_policy["Lift@K"].map(lambda value: f"{value:.2f}")
    shown_policy["命中流失"] = shown_policy["命中流失"].map(lambda value: f"{value:.1f}" if isinstance(value, float) and not value.is_integer() else f"{value:.0f}")
    st.dataframe(shown_policy, hide_index=True, width="stretch")
    st.caption("同一测试集、同一名额，向下取整；随机策略一行是数学期望，不是一份随机客户名单。入选名单只是历史测试样本的离线核验。")
    strategy = st.selectbox("查看策略的测试集示例", assignments["策略"].drop_duplicates().tolist())
    sample = assignments.loc[assignments["策略"].eq(strategy)].head(30).copy()
    sample = sample.rename(columns={
        "customerID": "客户编号", "actual_churn": "实际流失标签", "排名": "排名",
        "排序分数": "排序分数",
    })
    sample["实际流失标签"] = sample["实际流失标签"].map({0: "否", 1: "是"})
    sample = sample.drop(columns="策略")
    st.dataframe(sample, hide_index=True, width="stretch")
