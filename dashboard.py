"""客户流失分析看板：按运行目录读取完整的离线报告。"""
import argparse
import json
import os
import sys
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import auc, confusion_matrix, roc_curve


def configure_chart_font():
    """优先使用系统中文字体，避免中文轴标签被 Matplotlib 渲染成方框。"""
    font_root = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    candidates = ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc")
    for filename in candidates:
        font_path = font_root / filename
        if font_path.is_file():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False


configure_chart_font()

from ai_assistant import generate_explanation

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
AI_FIELD_LABELS = {
    "customerID": "客户编号", "model_name": "主模型", "rank": "风险排序名次",
    "risk_score": "排序分数", "tenure": "在网时长（月）", "Contract": "合同类型",
    "MonthlyCharges": "月费", "InternetService": "互联网服务",
    "PaymentMethod": "付款方式", "TechSupport": "技术支持",
    "OnlineSecurity": "在线安全服务", "contract_churn_rate": "合同类型历史流失率",
    "tenure_group_churn_rate": "在网时长分组历史流失率",
    "monthly_group_churn_rate": "月费分组历史流失率", "risk_evidence": "风险证据",
}


def inject_styles():
    """风险工作台视觉令牌：不改变 Streamlit 组件和业务计算。"""
    st.markdown(
        """
        <style>
        :root {
            --navy: #243b3a;
            --navy-deep: #fbf8f3;
            --teal: #176b63;
            --ink: #263238;
            --muted: #596560;
            --line: #ded6cb;
            --surface: #fffdf9;
            --canvas: #f5f1eb;
            --danger: #a94a3e;
            --warning: #a36a2f;
            --success: #3f7456;
        }
        .stApp { background: var(--canvas); color: var(--ink); }
        [data-testid="stHeader"] { background: rgba(245,241,235,.9); }
        .block-container { max-width: 1440px; padding-top: 4rem; padding-bottom: 3rem; }
        [data-testid="stSidebar"] { background: var(--navy-deep); border-right: 1px solid var(--line); }
        [data-testid="stSidebar"] section { padding-top: 1.6rem; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: var(--ink); }
        [data-testid="stSidebar"] p { color: #596560 !important; }
        [data-testid="stSelectbox"] label,
        [data-testid="stSelectbox"] label p,
        [data-testid="stSelectbox"] [data-testid="stWidgetLabel"] p { color:var(--ink) !important; }
        h1 { color: var(--navy); letter-spacing: 0; font-family: "Noto Serif SC", "Source Han Serif SC", "Microsoft YaHei", serif; font-weight: 800; margin-bottom: .2rem; }
        h2, h3 { color: var(--navy); letter-spacing: 0; }
        .eyebrow { color: var(--teal); font-size: .76rem; font-weight: 800; letter-spacing: .08em; margin-bottom: .35rem; }
        .hero-subtitle { color: var(--muted); font-size: 1rem; margin-top: 0; }
        .brand-mark { display:flex; align-items:center; gap:.65rem; margin-bottom:1.25rem; }
        .brand-icon { width:2.25rem; height:2.25rem; border-radius:50%; display:grid; place-items:center; background:#e7f0ec; color:var(--teal); border:1px solid #76a99e; font-weight:800; }
        .brand-name { color:var(--navy); font-size:1.05rem; font-weight:800; line-height:1.1; }
        .brand-caption { color:var(--muted); font-size:.72rem; margin-top:.18rem; }
        .sidebar-rule { border-top:1px solid var(--line); margin:1rem 0; }
        .sidebar-nav { color:var(--teal); font-size:.73rem; letter-spacing:.08em; margin:.8rem 0 .55rem; }
        .sidebar-item { color:var(--ink); padding:.55rem .65rem; border-radius:.5rem; margin:.18rem 0; font-size:.86rem; }
        .sidebar-item.active { background:#e7f0ec; color:var(--teal); font-weight:800; border-left:3px solid var(--teal); }
        .status-strip { display:flex; flex-wrap:wrap; gap:.65rem; align-items:center; margin:1rem 0 1.35rem; padding:.8rem 1rem; border:1px solid var(--line); border-left:4px solid var(--teal); border-radius:.65rem; background:var(--surface); color:var(--muted); font-size:.86rem; }
        .status-chip { display:inline-flex; align-items:center; gap:.35rem; padding:.25rem .55rem; border-radius:999px; background:#e7f0ec; color:var(--teal); font-weight:700; font-size:.74rem; }
        .section-kicker { color:var(--teal); font-size:.72rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; margin-bottom:.25rem; }
        .section-note { color:var(--muted); margin-top:-.35rem; margin-bottom:.8rem; }
        [data-testid="stMetric"] { background:var(--surface); border:1px solid var(--line); border-radius:.7rem; padding:1rem 1.05rem; box-shadow:0 3px 10px rgba(30,58,95,.045); }
        [data-testid="stMetricLabel"] { color:var(--muted); }
        [data-testid="stMetricValue"] { color:var(--navy); font-weight:800; }
        .table-wrap { width:100%; overflow-x:auto; border:1px solid var(--line); border-radius:.65rem; background:#fff; }
        .data-table { width:100%; border-collapse:collapse; color:var(--ink); background:#fff; font-size:.86rem; }
        .data-table th { background:#eef5f7; color:var(--navy); font-weight:800; text-align:left; white-space:nowrap; padding:.72rem .8rem; border-bottom:1px solid #cbdde3; }
        .data-table td { padding:.68rem .8rem; border-bottom:1px solid #e5edf2; white-space:nowrap; }
        .data-table tr:last-child td { border-bottom:0; }
        .data-table tr:hover td { background:#f5faf9; }
        .risk-card { border:1px solid var(--line); border-left:5px solid var(--warning); border-radius:.75rem; background:var(--surface); padding:1rem 1.1rem; margin:.8rem 0 1rem; }
        .risk-card.high { border-left-color:var(--danger); }
        .risk-card.low { border-left-color:var(--success); }
        .risk-label { color:var(--muted); font-size:.76rem; text-transform:uppercase; letter-spacing:.08em; }
        .risk-value { color:var(--navy); font-size:1.35rem; font-weight:800; margin:.15rem 0; }
        .risk-detail { color:var(--muted); font-size:.86rem; }
        [data-testid="stRadio"] > div { gap:.5rem; flex-wrap:wrap; }
        [data-testid="stRadio"] label { border:1px solid #cfc5b8; border-radius:.55rem; padding:.48rem .95rem; background:#fffdf9; color:var(--navy) !important; font-weight:700; }
        [data-testid="stRadio"] input[type="radio"] { accent-color:#b86b4b; }
        [data-testid="stRadio"] label p, [data-testid="stRadio"] label span { color:var(--navy) !important; }
        [data-testid="stRadio"] label:has(input:checked) { background:#e7f0ec; border-color:#76a99e; color:var(--teal) !important; }
        [data-testid="stRadio"] label:has(input:checked) p, [data-testid="stRadio"] label:has(input:checked) span { color:var(--teal) !important; }
        [data-testid="stSelectbox"] [data-baseweb="select"],
        [data-testid="stSelectbox"] [data-baseweb="select"] > div { background:#fffdf9 !important; border-color:#cfc5b8 !important; color:var(--ink) !important; }
        [data-testid="stSelectbox"] [data-baseweb="select"] div,
        [data-testid="stSelectbox"] [data-baseweb="select"] input,
        [data-testid="stSelectbox"] [data-baseweb="select"] span { background-color:transparent !important; color:var(--ink) !important; -webkit-text-fill-color:var(--ink) !important; }
        [data-testid="stSelectbox"] [data-baseweb="select"] svg { color:var(--ink) !important; fill:var(--ink) !important; }
        [data-baseweb="popover"] [role="listbox"] { background:#fffdf9 !important; border:1px solid var(--line) !important; }
        [data-baseweb="popover"] [role="option"],
        [data-baseweb="popover"] [role="option"] span { background:#fffdf9 !important; color:var(--ink) !important; -webkit-text-fill-color:var(--ink) !important; }
        [data-baseweb="popover"] [role="option"]:hover, [data-baseweb="popover"] [aria-selected="true"] { background:#e7f0ec !important; color:var(--teal) !important; }
        .stButton > button,
        .stButton > button[kind="primary"] { background:#e7f0ec !important; border:1px solid #76a99e !important; border-radius:.5rem; color:var(--teal) !important; font-weight:700; }
        .stButton > button p,
        .stButton > button span,
        .stButton > button[kind="primary"] p,
        .stButton > button[kind="primary"] span { color:var(--teal) !important; }
        .stButton > button:hover,
        .stButton > button[kind="primary"]:hover { background:#d8ebe4 !important; border-color:var(--teal) !important; color:var(--teal) !important; }
        .stButton > button:focus,
        .stButton > button[kind="primary"]:focus { box-shadow:0 0 0 .15rem rgba(23,107,99,.18) !important; color:var(--teal) !important; }
        [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:.6rem; overflow:hidden; }
        @media (max-width: 760px) {
            .block-container { padding:1.2rem .8rem 2rem; }
            h1 { font-size:1.85rem; }
            [data-testid="stSidebar"] { width:min(88vw, 320px); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_intro(kicker, title, note):
    st.markdown(f'<div class="section-kicker">{escape(kicker)}</div>', unsafe_allow_html=True)
    st.subheader(title)
    st.markdown(f'<div class="section-note">{escape(note)}</div>', unsafe_allow_html=True)


def risk_level(score):
    score = float(score)
    if score >= 0.7:
        return "高风险优先关注", "high"
    if score >= 0.4:
        return "中风险优先关注", ""
    return "低风险优先关注", "low"


def render_table(frame, percent_columns=(), decimal_columns=()):
    shown = frame.copy()
    for column in percent_columns:
        if column in shown:
            shown[column] = shown[column].map(lambda value: f"{float(value):.1%}")
    for column in decimal_columns:
        if column in shown:
            shown[column] = shown[column].map(lambda value: f"{float(value):.3f}")
    st.markdown(
        '<div class="table-wrap">' + shown.to_html(index=False, classes="data-table", escape=True) + '</div>',
        unsafe_allow_html=True,
    )


def _base_figure(size=(8, 3.6)):
    figure, axis = plt.subplots(figsize=size)
    figure.patch.set_facecolor("#ffffff")
    axis.set_facecolor("#ffffff")
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#cbd5e1")
    axis.tick_params(colors="#475569", labelsize=9)
    axis.grid(axis="y", color="#e2e8f0", linewidth=.8)
    axis.set_axisbelow(True)
    return figure, axis


def render_segment_chart(frame):
    figure, axis = _base_figure((8, 3.5))
    values = frame["流失率"].to_numpy(dtype=float)
    labels = frame["分群"].astype(str).tolist()
    colors = ["#0f766e" if value < values.mean() else "#c2410c" for value in values]
    bars = axis.barh(labels, values, color=colors, height=.52)
    axis.invert_yaxis()
    axis.set_xlim(0, max(values.max() * 1.22, .1))
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.set_xlabel("训练集历史流失率", color="#64748b")
    for bar, value in zip(bars, values):
        axis.text(value + .008, bar.get_y() + bar.get_height() / 2, f"{value:.1%}", va="center", color="#172033", fontsize=9)
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def render_cv_chart(cv):
    figure, axis = _base_figure((8, 3.2))
    names = list(cv["AP均值"])
    values = np.array([cv["AP均值"][name] for name in names])
    errors = np.array([cv["AP标准差"][name] for name in names])
    bars = axis.barh(names, values, xerr=errors, capsize=5, color=["#176b63", "#b86b4b"], height=.48, error_kw={"ecolor": "#9a938a", "elinewidth": 1.2})
    axis.set_xlim(0, max(values + errors) * 1.25)
    axis.set_xlabel("平均精确率 AP", color="#6f7774")
    axis.grid(axis="x", color="#e5ded4", linewidth=.8)
    axis.grid(axis="y", visible=False)
    for bar, value in zip(bars, values):
        axis.text(value + .012, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", color="#263238", fontsize=10, fontweight="bold")
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def render_feature_chart(frame):
    figure, axis = _base_figure((8, 4.4))
    shown = frame.head(10).sort_values("重要性")
    bars = axis.barh(shown["特征"], shown["重要性"], color="#0f766e", height=.58)
    axis.set_xlabel("模型特征重要性", color="#64748b")
    for bar, value in zip(bars, shown["重要性"]):
        axis.text(value + shown["重要性"].max() * .015, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8, color="#172033")
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def render_policy_chart(frame):
    figure, axis = _base_figure((8, 3.8))
    colors = ["#1e3a5f" if value == frame["Precision@K"].max() else "#0f766e" for value in frame["Precision@K"]]
    colors = ["#b86b4b" if value == frame["Precision@K"].max() else "#176b63" for value in frame["Precision@K"]]
    bars = axis.barh(frame["策略"], frame["Precision@K"], color=colors, height=.52)
    axis.set_xlim(0, max(frame["Precision@K"].max() * 1.25, .1))
    axis.set_xlabel("Precision@K", color="#6f7774")
    axis.grid(axis="x", color="#e5ded4", linewidth=.8)
    axis.grid(axis="y", visible=False)
    for bar, value in zip(bars, frame["Precision@K"]):
        axis.text(value + .012, bar.get_y() + bar.get_height() / 2, f"{value:.1%}", va="center", fontsize=9, color="#263238")
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def render_model_charts(predictions, metrics):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    figure.patch.set_facecolor("#ffffff")
    model_colors = {"逻辑回归": "#1e3a5f", "随机森林": "#0f766e"}
    for axis, model in zip(axes, metrics["模型"]):
        labels = predictions["actual_churn"].to_numpy(dtype=int)
        predicted = predictions[f"{model}_prediction"].to_numpy(dtype=int)
        matrix = confusion_matrix(labels, predicted)
        image = axis.imshow(matrix, cmap="GnBu", vmin=0, vmax=max(matrix.max(), 1))
        axis.set_title(model, color="#172033", fontweight="bold")
        axis.set_xticks([0, 1], ["未流失", "已流失"])
        axis.set_yticks([0, 1], ["未流失", "已流失"])
        axis.set_xlabel("预测结果", color="#64748b")
        axis.set_ylabel("实际结果", color="#64748b")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="#172033", fontsize=12, fontweight="bold")
        axis.spines[:].set_visible(False)
    figure.suptitle("测试集混淆矩阵", color="#172033", fontsize=14, fontweight="bold")
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)

    figure, axis = _base_figure((8, 4.3))
    for model in metrics["模型"]:
        false_positive, true_positive, _ = roc_curve(predictions["actual_churn"], predictions[f"{model}_probability"])
        axis.plot(false_positive, true_positive, color=model_colors[model], linewidth=2.4, label=f"{model}（AUC {auc(false_positive, true_positive):.3f}）")
    axis.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=1.2, label="随机基线")
    axis.set_xlabel("误报率", color="#64748b")
    axis.set_ylabel("召回率", color="#64748b")
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)


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


def ai_display_frame(row):
    shown = pd.DataFrame([row]).rename(columns=AI_FIELD_LABELS)
    for column in ("合同类型", "互联网服务", "付款方式", "技术支持", "在线安全服务"):
        if column in shown:
            shown[column] = shown[column].replace(VALUE_LABELS)
    for column in ("合同类型历史流失率", "在网时长分组历史流失率", "月费分组历史流失率"):
        if column in shown:
            shown[column] = shown[column].map(lambda value: f"{float(value):.1%}")
    if "排序分数" in shown:
        shown["排序分数"] = shown["排序分数"].map(lambda value: f"{float(value):.4f}")
    # 转置后各行原本可能混合数字和字符串；统一为字符串，避免 Streamlit/Arrow
    # 在渲染客户详情时进行隐式类型修复，也让 Demo 表格的展示口径一致。
    return shown.T.rename(columns={0: "当前客户"}).astype(str).reset_index(names="字段")


st.set_page_config(
    page_title="客户流失风险工作台",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="auto",
)
inject_styles()
st.markdown('<div class="eyebrow">客户留存分析 · 风险工作台</div>', unsafe_allow_html=True)
st.title("电信客户流失分析与用户排序")
st.markdown('<p class="hero-subtitle">从客户差异到风险排序，再到人工触达建议的一站式离线分析工作台。</p>', unsafe_allow_html=True)

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
with st.sidebar:
    st.markdown(
        '<div class="brand-mark"><div class="brand-icon">R</div><div><div class="brand-name">风险工作台</div><div class="brand-caption">客户流失分析 · 离线评估</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-rule"></div><div class="sidebar-nav">当前项目</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-item active">客户流失风险</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-nav">分析模块</div>', unsafe_allow_html=True)
    st.caption("请使用主区域页签切换分析模块。")
    st.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)
    path = st.selectbox("分析报告", reports, index=default_index, format_func=report_label)
    st.caption("看板只读取已生成报告，不会在页面内训练模型。")
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
    predictions = pd.read_csv(path / "predictions.csv")
    assignments = pd.read_csv(path / "policy_assignments.csv", dtype={"customerID": "string"})
    context_file = path / "ai_customer_context.csv"
    ai_context = pd.read_csv(context_file, dtype={"customerID": "string"}) if context_file.is_file() else pd.DataFrame()
    if "预测流失风险分数" in assignments.columns and "排序分数" not in assignments.columns:
        assignments = assignments.rename(columns={"预测流失风险分数": "排序分数"})
except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError) as exc:
    st.error(f"无法读取项目报告：{exc}。请切换分析报告，或使用 run_project.py 生成新报告。")
    st.stop()

selected_model = metrics["交叉验证"]["选择模型"]
selected_model_metrics = metrics["模型"][selected_model]
selected_policy = policy.loc[policy["策略"].eq(selected_model)].iloc[0]
st.markdown(
    f'<div class="status-strip"><span class="status-chip">● 离线评估</span><span>主模型：<strong>{escape(selected_model)}</strong></span><span>测试集：<strong>{metrics["测试集样本数"]:,}</strong> 名客户</span><span>固定触达容量：<strong>{metrics["名额比例"]:.0%}</strong></span><span>数据口径：公开快照，不代表线上收益</span></div>',
    unsafe_allow_html=True,
)
with st.container(horizontal=True):
    st.metric("客户总量", f"{metrics['样本数']:,}", border=True)
    st.metric("测试集 AP", f"{selected_model_metrics['AP（平均精确率）']:.3f}", border=True)
    st.metric("Precision@K", f"{selected_policy['Precision@K']:.1%}", border=True)
    st.metric("Lift@K", f"{selected_policy['Lift@K']:.2f}×", border=True)

with st.expander("数据来源与评估口径"):
    st.write("公开电信客户快照；分群使用训练集，模型依据训练集五折 AP 选择。测试结果为已参与开发分析的公开数据基准，不是独立盲测；历史流失识别不等于真实挽留收益。")
selected_page = st.radio(
    "分析模块",
    ["客户差异", "模型验证", "触达排序", "AI风险解读"],
    horizontal=True,
    label_visibility="collapsed",
)

if selected_page == "客户差异":
    section_intro("客户分群", "训练集客户分群", "先看客户结构差异，再决定模型排序应该服务哪类触达策略。")
    overall_display = overall.copy()
    render_table(overall_display, percent_columns=("流失率",))
    dimension = st.selectbox("选择分群维度", segments["分群维度"].drop_duplicates().tolist())
    chart = segments.loc[segments["分群维度"].eq(dimension), ["分群", "流失率"]].copy()
    contract_labels = {"Month-to-month": "月付合同", "One year": "一年合同", "Two year": "两年合同"}
    chart["分群"] = chart["分群"].replace(contract_labels)
    shown = segments.loc[segments["分群维度"].eq(dimension)].copy()
    shown["分群"] = shown["分群"].replace(contract_labels)
    shown["小样本"] = shown["小样本"].map({True: "是", False: "否"})
    render_table(shown, percent_columns=("流失率",))
    render_segment_chart(chart)
    st.caption("所有分群只使用训练集。差异是描述性关联，不说明合同或费用导致流失；客户数少于 30 的组应谨慎解读。")

elif selected_page == "模型验证":
    cv = metrics["交叉验证"]
    section_intro("模型验证", "模型验证", "用训练集五折 AP 选择主模型，再用固定测试集做一次性离线评估。")
    st.caption(f"根据训练集五折 AP 均值选择模型：{cv['选择模型']}。测试集没有参与选择。")
    cv_frame = pd.DataFrame({
        "模型": list(cv["AP均值"]),
        "五折 AP 均值": list(cv["AP均值"].values()),
        "五折 AP 标准差": [cv["AP标准差"][name] for name in cv["AP均值"]],
    })
    render_table(cv_frame, decimal_columns=("五折 AP 均值", "五折 AP 标准差"))
    render_cv_chart(cv)
    metric_frame = pd.DataFrame(metrics["模型"]).T.reset_index(names="模型")
    metric_frame = metric_frame.rename(columns={
        "Accuracy（准确率）": "准确率", "Precision（查准率）": "查准率",
        "Recall（查全率）": "查全率", "ROC-AUC": "ROC-AUC",
        "AP（平均精确率）": "平均精确率 AP",
    })
    st.markdown('<div class="section-kicker">测试集证据</div>', unsafe_allow_html=True)
    st.subheader("固定测试集指标")
    render_table(metric_frame, decimal_columns=("准确率", "查准率", "查全率", "F1", "ROC-AUC", "平均精确率 AP"))
    st.caption("AP（平均精确率）适合观察正类较少时的排序质量；Accuracy 不能单独代表模型好坏。分类阈值固定为 0.5。")
    render_model_charts(predictions, metrics)
    importance = importance.copy()
    importance["特征"] = importance.feature.map(feature_label)
    importance = importance.rename(columns={"importance": "重要性"})[["特征", "重要性"]].head(15)
    st.subheader("随机森林关注的特征")
    render_feature_chart(importance)
    st.caption("特征重要性描述模型使用方式，不代表因果关系。")

elif selected_page == "触达排序":
    section_intro("触达策略", "固定 10% 名额下的策略对照", "把模型分数转换成有限人力下的优先触达顺序，并与规则基线比较。")
    render_policy_chart(policy)
    render_table(policy, percent_columns=("Precision@K", "Recall@K"), decimal_columns=("Lift@K",))
    st.caption("同一测试集、同一名额，向下取整；随机策略一行是数学期望，不是一份随机客户名单。入选名单只是历史测试样本的离线核验。")
    strategy = st.selectbox("查看策略的测试集示例", assignments["策略"].drop_duplicates().tolist())
    sample = assignments.loc[assignments["策略"].eq(strategy)].head(30).copy()
    sample = sample.rename(columns={
        "customerID": "客户编号", "actual_churn": "实际流失标签", "排名": "排名",
        "排序分数": "排序分数",
    })
    sample["实际流失标签"] = sample["实际流失标签"].map({0: "否", 1: "是"})
    sample = sample.drop(columns="策略")
    render_table(sample, decimal_columns=("排序分数",))

else:
    section_intro("AI 辅助", "AI 风险解读与触达建议", "统计模型负责排序，AI 只把白名单证据整理为可复核的中文解释。")
    st.caption("AI 不修改模型分数、阈值或客户名单；没有远程接口时使用离线模板。")
    if ai_context.empty:
        st.info("当前报告没有 AI 客户上下文。请用 run_project.py 重新生成一份报告；旧报告仍可查看前三个页面。")
    else:
        customer_ids = ai_context["customerID"].astype("string").tolist()
        selected_customer = st.selectbox("选择主模型固定容量优先客户", customer_ids)
        selected_row = ai_context.loc[
            ai_context["customerID"].eq(selected_customer)
        ].iloc[0].to_dict()
        level, level_class = risk_level(selected_row["risk_score"])
        st.markdown(
            f'<div class="risk-card {level_class}"><div class="risk-label">当前客户风险等级</div><div class="risk-value">{escape(level)}</div><div class="risk-detail">排序名次 #{int(selected_row["rank"])} · 排序分数 {float(selected_row["risk_score"]):.4f} · 主模型 {escape(str(selected_row["model_name"]))}</div></div>',
            unsafe_allow_html=True,
        )
        render_table(ai_display_frame(selected_row))
        st.caption("上下文不包含测试集真实流失标签；风险分数用于排序，分组流失率只表示历史关联。")
        if st.button("生成 AI 风险解读", type="primary"):
            explanation = generate_explanation(selected_row)
            st.caption(f"生成模式：{explanation['mode']}")
            st.markdown(f"**风险摘要**\n\n{explanation['summary']}")
            left, right = st.columns(2)
            with left:
                st.markdown("**风险证据**")
                for item in explanation["evidence"]:
                    st.write(f"• {item}")
            with right:
                st.markdown("**建议动作**")
                for item in explanation["actions"]:
                    st.write(f"• {item}")
            st.caption(f"使用边界：{explanation['limitations']}")
