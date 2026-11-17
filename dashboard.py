"""客户流失分析看板：按运行目录读取完整的离线报告。"""
import argparse
import json
import os
import sys
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import auc, confusion_matrix, roc_curve

from telco_churn.chart_font import configure_chart_font
from optional_ai.assistant import context_from_report, generate_explanation


configure_chart_font()

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
    "charges_deviation": "费用偏差（实缴 vs 月费）",
    "has_high_charges": "高月费标记（≥70元）",
    "service_count": "增值服务开通数",
    "has_internet": "是否开通互联网",
    "full_protection": "全套安全防护标记",
    "is_new_customer": "新客户标记（≤6月）",
    "is_loyal_customer": "忠诚客户标记（≥36月）",
    "tenure_contract_month": "月付在网时长",
    "auto_payment": "自动扣款标记",
    "paperless_risky": "电子账单+电子支票",
    "has_family": "家庭客户标记",
    "single_senior": "独居老年客户标记",
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
POLICY_METRIC_LABELS = {
    "Precision@K": "优先名单命中率",
    "Recall@K": "流失客户覆盖率",
    "Lift@K": "相对随机提升倍数",
}
MODEL_COLORS = {"逻辑回归": "#006b40", "随机森林": "#bf754e", "LightGBM": "#2465ad"}


def inject_styles():
    """沿用暖白与青绿，统一工作台的展示层样式。"""
    st.markdown(
        """
        <style>
        :root {
            --heading:#243b3a; --sidebar:#fbf8f3; --teal:#176b63;
            --ink:#263238; --muted:#596560; --line:#ded6cb;
            --surface:#fffdf9; --canvas:#f5f1eb; --accent:#e7f0ec;
            --danger:#a94a3e; --warning:#a36a2f; --success:#3f7456;
        }
        .stApp { background:var(--canvas); color:var(--ink);
            font-family:"Microsoft YaHei","PingFang SC","Noto Sans SC",system-ui,sans-serif; }
        .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"] {
            font-family:"Microsoft YaHei","PingFang SC","Noto Sans SC",system-ui,sans-serif; }
        .stApp input, .stApp button, .stApp select, .stApp textarea { font-family:inherit; }
        [data-testid="stHeader"] { background:var(--canvas); }
        [data-testid="stAppDeployButton"], [data-testid="stMainMenu"] { display:none; }
        .block-container { max-width:1440px; padding:2.5rem 3rem 3.5rem; }
        [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] { gap:1.15rem; }
        [data-testid="stSidebar"] { background:var(--sidebar); border-right:1px solid var(--line); }
        [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top:1.5rem; }
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color:var(--muted) !important; opacity:1 !important; font-size:.81rem; line-height:1.75; }
        h1 { color:var(--heading); font-family:"Noto Serif SC","Source Han Serif SC","Microsoft YaHei",serif;
            font-size:2.25rem !important; font-weight:800; letter-spacing:-.035em; line-height:1.35 !important; padding-bottom:.3rem; }
        h2, h3 { color:var(--heading); letter-spacing:-.025em; }
        h2 { font-size:1.35rem !important; padding-top:.75rem; }
        h3 { font-size:1.1rem !important; }
        .hero-subtitle { color:var(--muted); font-size:.95rem; line-height:1.8; margin:0; }
        .brand-mark { display:flex; align-items:center; gap:.7rem; margin-bottom:1rem; }
        .brand-icon { width:2.4rem; height:2.4rem; border-radius:.7rem; display:grid; place-items:center;
            background:var(--teal); color:var(--surface); font-weight:800; font-size:1.1rem; }
        .brand-name { color:var(--heading); font-size:1.08rem; font-weight:800; line-height:1.4; }
        .brand-caption { color:var(--muted); font-size:.72rem; margin-top:.15rem; }
        .sidebar-rule { border-top:1px solid var(--line); margin:1.1rem 0; }
        .sidebar-nav { color:var(--muted); font-size:.72rem; font-weight:600; letter-spacing:.1em; margin:1rem 0 .55rem; }
        .sidebar-item { color:var(--heading); font-size:.9rem; font-weight:650; padding:.25rem 0; }
        .st-key-analysis_navigation [role="radiogroup"] { flex-direction:column; gap:.25rem; }
        .st-key-analysis_navigation label { box-sizing:border-box; width:100%; min-height:44px;
            padding:.6rem .75rem; border:1px solid transparent; border-left:3px solid transparent;
            border-radius:.5rem; background:transparent; transition:background .15s; }
        .st-key-analysis_navigation label:hover { background:#f0ebe2; }
        .st-key-analysis_navigation label:has(input:checked) { background:var(--accent); border-left-color:var(--teal); }
        .st-key-analysis_navigation label:has(input:checked) p { color:var(--teal); font-weight:750; }
        .st-key-analysis_navigation label:has(input:focus-visible) { outline:2px solid var(--teal); outline-offset:2px; }
        .section-kicker { color:var(--teal); font-size:.73rem; font-weight:750; letter-spacing:.12em; margin-bottom:.2rem; }
        .status-strip { display:flex; flex-wrap:wrap; gap:.6rem 1.25rem; align-items:center;
            padding:.8rem 1rem; border:1px solid var(--line); border-radius:.6rem;
            background:var(--surface); color:var(--muted); font-size:.8rem; line-height:1.8; }
        .status-chip { display:inline-flex; align-items:center; padding:.15rem .6rem; border-radius:999px;
            background:var(--accent); color:var(--teal); font-weight:700; font-size:.72rem; }
        .st-key-kpi-row { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.85rem; }
        .st-key-segment-kpis { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.85rem; }
        .st-key-kpi-row > div, .st-key-segment-kpis > div { width:100% !important; min-width:0 !important; }
        [data-testid="stMetric"] { height:100%; box-sizing:border-box; min-height:116px; background:var(--surface); border:1px solid var(--line);
            border-radius:.7rem; padding:1.05rem 1.1rem; box-shadow:0 2px 7px rgba(36,59,58,.025); }
        [data-testid="stMetricLabel"] { color:var(--muted); font-size:.8rem; }
        [data-testid="stMetricLabel"] p { white-space:normal; font-size:.8rem; }
        [data-testid="stMetricValue"] { color:var(--heading); font-size:1.9rem; font-weight:750; letter-spacing:-.035em;
            font-variant-numeric:tabular-nums; }
        .st-key-overview-evidence, .st-key-segment-evidence, .st-key-model-evidence, .st-key-policy-evidence {
            background:var(--surface); border-color:var(--line) !important; border-radius:.75rem; padding:.25rem; }
        [data-testid="stImage"] { background:var(--surface); border-radius:.5rem; overflow:hidden; }
        .table-wrap { max-height:460px; overflow:auto; width:100%; border:1px solid var(--line); border-radius:.6rem;
            background:var(--surface); scrollbar-color:#c3c7bc var(--surface); }
        .table-wrap:focus-visible { outline:2px solid var(--teal); outline-offset:3px; }
        .data-table { width:100%; border-collapse:collapse; color:var(--ink); background:var(--surface); font-size:.82rem; }
        .data-table th { position:sticky; top:0; z-index:1; background:#eeeae1; color:var(--heading); font-weight:700;
            text-align:left; white-space:nowrap; padding:.7rem .85rem; border-bottom:1px solid var(--line); }
        .data-table td { padding:.75rem .85rem; border-bottom:1px solid #ebe5db; white-space:nowrap; }
        .data-table th.numeric, .data-table td.numeric { text-align:right; font-variant-numeric:tabular-nums; }
        .data-table tr:last-child td { border-bottom:0; }
        .data-table tr:hover td { background:#eff4ed; }
        [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:.6rem; overflow:hidden; }
        .risk-card { border:1px solid var(--line); border-left:4px solid var(--warning); border-radius:.75rem;
            background:var(--surface); padding:1.2rem 1.3rem; margin:.2rem 0; }
        .risk-card.high { border-left-color:var(--danger); }
        .risk-card.low { border-left-color:var(--success); }
        .risk-label { color:var(--muted); font-size:.78rem; }
        .risk-value { color:var(--heading); font-size:1.4rem; font-weight:750; margin:.4rem 0; }
        .risk-detail { color:var(--muted); font-size:.85rem; line-height:1.8; font-variant-numeric:tabular-nums; }
        .st-key-ai-summary, .st-key-ai-evidence, .st-key-ai-actions, .st-key-ai-limitations {
            background:var(--surface); border-color:var(--line) !important; border-radius:.75rem; }
        .st-key-ai-summary { border-left:4px solid var(--teal) !important; }
        .st-key-ai-actions { background:#eef5f1; }
        .st-key-ai-limitations { background:#f8f4ec; }
        .st-key-ai-analysis .ai-analysis-title { color:var(--heading); font-family:inherit; font-size:1rem !important;
            font-weight:750; line-height:1.5; margin:0 0 .65rem; padding:0; }
        .ai-analysis-text, .ai-analysis-list { color:var(--ink); font-family:inherit; font-size:.95rem; line-height:1.85; overflow-wrap:anywhere; }
        .ai-analysis-text { margin:0; white-space:pre-line; }
        .ai-analysis-list { width:100%; box-sizing:border-box; margin:0; padding-left:1.25rem; }
        .ai-analysis-list li { padding:.65rem .1rem; border-bottom:1px solid var(--line); white-space:pre-line; }
        .ai-analysis-list li:first-child { padding-top:0; }
        .ai-analysis-list li:last-child { border-bottom:0; padding-bottom:0; }
        .ai-analysis-list li::marker { color:var(--teal); }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div { background:var(--surface); border-color:var(--line); min-height:44px; }
        .stButton > button { min-height:44px; border-radius:.5rem; font-weight:650; }
        .stButton > button[kind="primary"] { background:var(--teal); border-color:var(--teal); color:#fffdf9; }
        .stButton > button[kind="primary"]:hover { background:#12564f; border-color:#12564f; color:#fffdf9; }
        .stButton > button:focus-visible, [data-testid="stExpander"] summary:focus-visible {
            outline:2px solid var(--teal); outline-offset:3px; }
        [data-testid="stAlert"] { border:1px solid var(--line); border-radius:.6rem; }
        [data-testid="stAlert"][data-baseweb="notification"] { font-size:.88rem; }
        [data-testid="stExpander"] { border-color:var(--line); border-radius:.6rem; }
        @media (max-width:1100px) {
            .block-container { padding-left:1.5rem; padding-right:1.5rem; }
            .st-key-kpi-row { grid-template-columns:repeat(2,minmax(0,1fr)); }
        }
        @media (max-width:760px) {
            .block-container { padding:1.4rem 1rem 2rem; }
            h1 { font-size:1.75rem !important; letter-spacing:-.04em; }
            h2 { font-size:1.2rem !important; }
            [data-testid="stSidebar"] { width:min(88vw,320px); }
            .st-key-segment-kpis { grid-template-columns:repeat(2,minmax(0,1fr)); }
            [data-testid="stMetric"] { min-height:110px; padding:.85rem; }
            [data-testid="stMetricLabel"] { min-height:2.2rem; }
            [data-testid="stMetricValue"] { font-size:1.65rem; }
            .status-strip { gap:.4rem .8rem; padding:.7rem .8rem; }
            .data-table { font-size:.8rem; }
            [data-testid="stSelectbox"], [data-testid="stVerticalBlock"] { min-width:0; }
        }
        @media (max-width:340px) {
            .st-key-kpi-row, .st-key-segment-kpis { grid-template-columns:1fr; }
        }
        @media (prefers-reduced-motion:reduce) { .st-key-analysis_navigation label { transition:none; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title, description):
    st.markdown('<div class="section-kicker">客户留存分析 · 离线评估</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(description)


def risk_level(score):
    score = float(score)
    if score >= 0.7:
        return "高风险优先关注", "high"
    if score >= 0.4:
        return "中风险优先关注", ""
    return "低风险优先关注", "low"


def render_table(frame, percent_columns=(), decimal_columns=()):
    """小表直接阅读，长表使用原生排序、搜索和下载；格式化不改源数据。"""
    if len(frame) > 12:
        formats = {
            POLICY_METRIC_LABELS.get(column, column): st.column_config.NumberColumn(format="percent")
            for column in percent_columns if column in frame
        }
        formats.update({
            POLICY_METRIC_LABELS.get(column, column): st.column_config.NumberColumn(format="%.3f")
            for column in decimal_columns if column in frame
        })
        st.dataframe(
            frame.rename(columns=POLICY_METRIC_LABELS), hide_index=True, width="stretch",
            height=min(440, 36 * (len(frame) + 1) + 3), column_config=formats,
        )
        st.caption(f"共 {len(frame):,} 行 · 点击列名排序；表格工具栏支持搜索、全屏与下载。")
        return
    shown = frame.copy()
    numeric_columns = set(shown.select_dtypes(include="number").columns)
    for column in percent_columns:
        if column in shown:
            shown[column] = shown[column].map(
                lambda value: "—" if pd.isna(value) else f"{float(value):.1%}"
            )
    for column in decimal_columns:
        if column in shown:
            shown[column] = shown[column].map(
                lambda value: "—" if pd.isna(value) else f"{float(value):.3f}"
            )
    for column in shown.select_dtypes(include=["object", "string"]).columns:
        shown[column] = shown[column].fillna("—")
    for column in numeric_columns - set(percent_columns) - set(decimal_columns):
        shown[column] = shown[column].map(
            lambda value: "—" if pd.isna(value) else f"{value:,}"
        )
    headers = "".join(
        f'<th scope="col" class="{"numeric" if column in numeric_columns else ""}">'
        f'{escape(POLICY_METRIC_LABELS.get(column, str(column)))}</th>' for column in shown
    )
    rows = "".join(
        "<tr>" + "".join(
            f'<td class="{"numeric" if column in numeric_columns else ""}">{escape(str(value))}</td>'
            for column, value in zip(shown.columns, row)
        ) + "</tr>" for row in shown.itertuples(index=False, name=None)
    )
    st.markdown(
        '<div class="table-wrap" tabindex="0" role="region" aria-label="数据明细，可横向滚动">'
        f'<table class="data-table"><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def _base_figure(size=(8, 3.6)):
    figure, axis = plt.subplots(figsize=size)
    figure.patch.set_facecolor("#fffdf9")
    axis.set_facecolor("#fffdf9")
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#ded6cb")
    axis.tick_params(colors="#596560", labelsize=10, length=0, pad=8)
    axis.grid(axis="x", color="#e5ded4", linewidth=.7)
    axis.set_axisbelow(True)
    return figure, axis


def render_segment_chart(frame):
    figure, axis = _base_figure((8, 3.5))
    values = frame["流失率"].to_numpy(dtype=float)
    labels = frame["分群"].astype(str).tolist()
    errors = None
    if {"流失率下限", "流失率上限"}.issubset(frame.columns):
        errors = np.array([values - frame["流失率下限"], frame["流失率上限"] - values])
    bars = axis.barh(labels, values, color="#176b63", height=.45, xerr=errors,
                     capsize=4, error_kw={"ecolor": "#596560", "elinewidth": 1.2})
    axis.invert_yaxis()
    upper = frame["流失率上限"].max() if errors is not None else values.max()
    axis.set_xlim(0, max(upper * 1.22, .1))
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.set_xlabel("训练集历史流失率", color="#596560", labelpad=10)
    right = frame["流失率上限"].to_numpy() if errors is not None else values
    for bar, value, endpoint in zip(bars, values, right):
        axis.text(endpoint + .008, bar.get_y() + bar.get_height() / 2, f"{value:.1%}", va="center", color="#243b3a", fontsize=10)
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def render_cv_chart(cv):
    figure, axis = _base_figure((8, 3.2))
    names = list(cv["AP均值"])
    values = np.array([cv["AP均值"][name] for name in names])
    errors = np.array([cv["AP标准差"][name] for name in names])
    bars = axis.barh(names, values, xerr=errors, capsize=4,
                    color=[MODEL_COLORS.get(name, "#176b63") for name in names],
                    height=.43, error_kw={"ecolor": "#596560", "elinewidth": 1.2})
    axis.set_xlim(0, max(values + errors) * 1.25)
    axis.set_xlabel("平均精确率 AP（误差线为折间标准差）", color="#596560", labelpad=10)
    axis.grid(axis="x", color="#e5ded4", linewidth=.8)
    axis.grid(axis="y", visible=False)
    for bar, value, error in zip(bars, values, errors):
        axis.text(value + error + .012, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", color="#263238", fontsize=10, fontweight="bold")
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def render_feature_chart(frame):
    figure, axis = _base_figure((8, 4.4))
    shown = frame.head(10).sort_values("重要性")
    bars = axis.barh(shown["特征"], shown["重要性"], color="#0f766e", height=.58)
    axis.set_xlabel("模型特征重要性", color="#596560")
    for bar, value in zip(bars, shown["重要性"]):
        axis.text(value + shown["重要性"].max() * .015, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8, color="#243b3a")
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def render_policy_chart(frame):
    figure, axis = _base_figure((8, 3.8))
    colors = [MODEL_COLORS.get(name, "#8a9386") for name in frame["策略"]]
    bars = axis.barh(frame["策略"], frame["Precision@K"], color=colors, height=.52)
    axis.set_xlim(0, max(frame["Precision@K"].max() * 1.25, .1))
    axis.set_xlabel(POLICY_METRIC_LABELS["Precision@K"], color="#596560", labelpad=10)
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.grid(axis="x", color="#e5ded4", linewidth=.8)
    axis.grid(axis="y", visible=False)
    for bar, value in zip(bars, frame["Precision@K"]):
        axis.text(value + .012, bar.get_y() + bar.get_height() / 2, f"{value:.1%}", va="center", fontsize=9, color="#263238")
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def render_model_charts(predictions, metrics):
    model_names = list(metrics["模型"])
    st.subheader("历史留出集混淆矩阵（训练OOF阈值）" if metrics.get("报告版本", 0) >= 3 else "历史留出集混淆矩阵（0.5阈值）")
    with st.container(horizontal=True):
        for model in model_names:
            with st.container(width=280, border=True):
                figure, axis = _base_figure((4, 3.8))
                labels = predictions["actual_churn"].to_numpy(dtype=int)
                if metrics.get("报告版本", 0) >= 3:
                    predicted = (predictions[f"{model}_probability"].to_numpy() >= metrics["模型"][model]["折外F1阈值"]).astype(int)
                else:
                    predicted = predictions[f"{model}_prediction"].to_numpy(dtype=int)
                matrix = confusion_matrix(labels, predicted, labels=[0, 1])
                axis.grid(False)
                axis.imshow(matrix, cmap="Greens", vmin=0, vmax=max(matrix.max(), 1))
                axis.set_title(model, color="#243b3a", fontweight="bold")
                axis.set_xticks([0, 1], ["未流失", "已流失"])
                axis.set_yticks([0, 1], ["未流失", "已流失"])
                axis.set_xlabel("预测结果", color="#596560")
                axis.set_ylabel("实际结果", color="#596560")
                for row in range(2):
                    for column in range(2):
                        color = "#ffffff" if matrix[row, column] > matrix.max() * .5 else "#243b3a"
                        axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color=color, fontsize=15, fontweight="bold")
                axis.spines[:].set_visible(False)
                figure.tight_layout()
                st.pyplot(figure, width="stretch")
                plt.close(figure)

    st.subheader("历史留出集 ROC 曲线")
    figure, axis = _base_figure((8, 4.3))
    for index, model in enumerate(metrics["模型"]):
        false_positive, true_positive, _ = roc_curve(predictions["actual_churn"], predictions[f"{model}_probability"])
        axis.plot(false_positive, true_positive, color=MODEL_COLORS.get(model, "#176b63"),
                  linestyle=["-", "--", "-."][index % 3], linewidth=2,
                  label=f"{model}（AUC {auc(false_positive, true_positive):.3f}）")
    axis.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=1.2, label="随机基线")
    axis.set_xlabel("误报率", color="#596560")
    axis.set_ylabel("召回率", color="#596560")
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
    prefix, separator, bare = value.partition("__")
    if not separator:
        bare = value
        prefix = ""
    if bare in FEATURE_LABELS:
        return FEATURE_LABELS[bare]
    if prefix == "categorical":
        for name in sorted(FEATURE_LABELS, key=len, reverse=True):
            if bare.startswith(name + "_"):
                category = bare[len(name) + 1:]
                return f"{FEATURE_LABELS[name]}：{VALUE_LABELS.get(category, category)}"
    return bare


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
    # 转置后的同一列包含数字和文字，按文本展示。
    return shown.T.rename(columns={0: "当前客户"}).astype(str).reset_index(names="字段")


def ai_settings():
    try:
        secrets = st.secrets.to_dict()
    except FileNotFoundError:
        secrets = {}
    defaults = {
        "CHURN_AI_BASE_URL": "", "CHURN_AI_MODEL": "", "CHURN_AI_API_KEY": "",
        "CHURN_AI_TIMEOUT_SECONDS": "20",
    }
    return {key: os.environ.get(key, str(secrets.get(key, default))) for key, default in defaults.items()}


def render_ai_explanation(explanation):
    st.caption(f"生成模式：{explanation['mode']}")
    with st.container(border=True, key="ai-summary"):
        st.markdown('<h3 class="ai-analysis-title">风险摘要</h3>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="ai-analysis-text">{escape(explanation["summary"])}</p>',
            unsafe_allow_html=True,
        )
    left, right = st.columns(2, gap="small")
    for column, title, field, key in (
        (left, "风险证据", "evidence", "ai-evidence"),
        (right, "建议动作", "actions", "ai-actions"),
    ):
        with column:
            with st.container(border=True, height="stretch", key=key):
                st.markdown(f'<h3 class="ai-analysis-title">{title}</h3>', unsafe_allow_html=True)
                items = "".join(f"<li>{escape(item)}</li>" for item in explanation[field])
                st.markdown(f'<ul class="ai-analysis-list">{items}</ul>', unsafe_allow_html=True)
    with st.container(border=True, key="ai-limitations"):
        st.markdown('<h3 class="ai-analysis-title">解读说明</h3>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="ai-analysis-text">{escape(explanation["limitations"])}</p>',
            unsafe_allow_html=True,
        )


st.set_page_config(
    page_title="客户流失风险工作台",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="auto",
)
inject_styles()

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
    st.markdown('<div class="sidebar-item">客户流失风险</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-nav">分析模块</div>', unsafe_allow_html=True)
    selected_page = st.radio(
        "分析模块",
        ["报告总览", "客户差异", "模型验证", "模型解释", "触达排序", "AI风险解读"],
        label_visibility="collapsed",
        key="analysis_navigation",
        width="stretch",
    )
    st.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)
    path = st.selectbox("分析报告", reports, index=default_index, format_func=lambda report: report.name)
    st.caption("切换报告，查看对应的分析成果。")
if requested is not None and requested not in reports:
    st.info(f"指定目录 {report_label(requested)} 不包含完整的当前版报告；当前查看 {report_label(path)}。可在侧栏切换报告，原目录未修改。")

try:
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    if "交叉验证" not in metrics:
        raise ValueError("该报告缺少当前版交叉验证结果，请选择其他完整报告或重新生成")
    overall = pd.read_csv(path / "overall_summary.csv")
    segments = pd.read_csv(path / "segment_summary.csv")
    policy = pd.read_csv(path / "policy_metrics.csv")
    importance = pd.read_csv(path / "feature_importance.csv")
    predictions = pd.read_csv(path / "predictions.csv", dtype={"customerID": "string"})
    assignments = pd.read_csv(path / "policy_assignments.csv", dtype={"customerID": "string"})
    context_file = path / "ai_customer_context.csv"
    ai_context = pd.read_csv(context_file, dtype={"customerID": "string"}) if context_file.is_file() else pd.DataFrame()
    inference_file = path / "segment_inference.csv"
    inference = pd.read_csv(inference_file) if inference_file.is_file() else pd.DataFrame()
    cross_segments_file = path / "contract_tenure_segments.csv"
    cross_segments = pd.read_csv(cross_segments_file) if cross_segments_file.is_file() else pd.DataFrame()
    if "预测流失风险分数" in assignments.columns and "排序分数" not in assignments.columns:
        assignments = assignments.rename(columns={"预测流失风险分数": "排序分数"})
    context_recovered = ai_context.empty
    if context_recovered:
        ai_context = pd.DataFrame(context_from_report(
            assignments.to_dict("records"), predictions.to_dict("records"),
            metrics["交叉验证"]["选择模型"],
        ))
except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError) as exc:
    st.error(f"无法读取项目报告：{exc}。请切换分析报告，或使用 run_project.py 生成新报告。")
    st.stop()

selected_model = metrics["交叉验证"]["选择模型"]
selected_model_metrics = metrics["模型"][selected_model]
selected_policy = policy.loc[policy["策略"].eq(selected_model)].iloc[0]
current_method = metrics.get("报告版本", 0) >= 3
if not current_method:
    st.warning("历史报告：使用旧调参/检验流程，不能视为嵌套验证结果。旧生存与留出集最优名额不作为业务结论。")

if selected_page == "报告总览":
    st.markdown('<div class="section-kicker">客户留存分析 · 风险工作台</div>', unsafe_allow_html=True)
    st.title("电信客户流失预测与排序")
    st.markdown('<p class="hero-subtitle">分析客户流失差异，比较模型在有限联系名额下的排序效果。</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="status-strip"><span class="status-chip">● 离线评估</span>'
        f'<span>主模型：<strong>{escape(selected_model)}</strong></span>'
        f'<span>历史留出集：<strong>{metrics["测试集样本数"]:,}</strong> 名客户</span>'
        f'<span>固定触达容量：<strong>{metrics["名额比例"]:.0%}</strong></span>'
        '<span>公开数据 · 离线结果</span></div>',
        unsafe_allow_html=True,
    )
    with st.container(horizontal=True, key="kpi-row"):
        st.metric("客户总量", f"{metrics['样本数']:,}", border=True)
        st.metric("历史留出 AP", f"{selected_model_metrics['AP（平均精确率）']:.3f}", border=True)
        st.metric(
            POLICY_METRIC_LABELS["Precision@K"], f"{selected_policy['Precision@K']:.1%}", border=True,
            help="优先名单中历史流失客户所占比例。",
        )
        st.metric(
            POLICY_METRIC_LABELS["Lift@K"], f"{selected_policy['Lift@K']:.2f}×", border=True,
            help="优先名单命中率相对于随机选择的提升倍数。",
        )
    st.info(
        f"在历史留出集 {metrics['测试集样本数']:,} 名客户中，主模型将 {int(selected_policy['名额']):,} 人列为优先触达对象，"
        f"其中 {int(selected_policy['命中流失']):,} 人的历史标签为流失。"
    )
    with st.container(border=True, key="overview-evidence"):
        st.subheader(f"同样联系 {int(selected_policy['名额']):,} 人，排序表现如何？")
        st.caption("同一历史留出集 · 固定容量对照 · 数值越高，名单中历史流失客户占比越高")
        render_policy_chart(policy)
    st.caption("主模型按训练集外层 AP 均值选择；本图为历史留出集描述，不据此重新选模。命中历史流失不代表挽留成功。")
    with st.expander("数据来源与评估口径"):
        st.write(
            f"使用公开电信客户数据。分群统计和 {metrics['交叉验证']['折数']} 折模型选择均在训练集完成。"
            "公开数据已用于开发分析，测试结果不是独立盲测。名单命中率不能当作实际挽留成功率。"
        )

elif selected_page == "客户差异":
    page_header("训练集客户分群", "按合同、在网时长和月费比较客户规模与历史流失率。")
    summary = overall.iloc[0]
    with st.container(horizontal=True, key="segment-kpis"):
        st.metric("训练集客户数", f"{int(summary['客户数']):,}", border=True)
        st.metric("训练集流失人数", f"{int(summary['流失人数']):,}", border=True)
        st.metric("训练集流失率", f"{summary['流失率']:.1%}", border=True)
    dimension = st.selectbox("选择分群维度", segments["分群维度"].drop_duplicates().tolist())
    chart = segments.loc[segments["分群维度"].eq(dimension), [
        column for column in ("分群", "流失率", "流失率下限", "流失率上限") if column in segments
    ]].copy()
    contract_labels = {"Month-to-month": "月付合同", "One year": "一年合同", "Two year": "两年合同"}
    chart["分群"] = chart["分群"].replace(contract_labels)
    shown = segments.loc[segments["分群维度"].eq(dimension)].copy()
    shown["分群"] = shown["分群"].replace(contract_labels)
    shown["小样本"] = shown["小样本"].map({True: "是", False: "否"})
    with st.container(border=True, key="segment-evidence"):
        st.subheader(f"{dimension}与历史流失率")
        render_segment_chart(chart)
    st.caption("分群仅统计训练集；区间为 95% Wilson 区间。少于 30 人的组已标记为小样本，差异不代表因果关系。")
    render_table(shown, percent_columns=("流失率", "流失率下限", "流失率上限"))
    if not inference.empty:
        st.subheader("分群与历史流失的关联检验")
        inference_shown = inference.rename(columns={
            "p值": "原始 p 值", "校正后p值": "Holm 校正 p 值",
        })
        render_table(inference_shown, decimal_columns=(
            "卡方统计量", "Cramér's V", "原始 p 值", "Holm 校正 p 值",
        ))
        st.caption("卡方检验衡量分群与历史流失标签的关联；Cramér’s V 表示关联强度，不说明因果。")
    else:
        st.info("此旧版报告未包含统计检验结果。重新运行项目可生成。")
    if not cross_segments.empty:
        st.subheader("合同类型 × 在网时长")
        render_table(cross_segments, percent_columns=("流失率", "流失率下限", "流失率上限"))
        st.caption("二维分群用于描述客户结构；小样本组合的流失率不宜单独解读。")
    else:
        st.info("此旧版报告未包含合同与在网时长交叉分群。重新运行项目可生成。")

elif selected_page == "模型验证":
    cv = metrics["交叉验证"]
    page_header("模型验证", "比较嵌套交叉验证结果，并查看历史留出集描述性指标。")
    with st.container(horizontal=True, key="kpi-row"):
        st.metric("主模型", selected_model, border=True)
        st.metric("外层 AP 均值", f"{cv['AP均值'][selected_model]:.3f}", border=True)
        st.metric("留出集 AP", f"{selected_model_metrics['AP（平均精确率）']:.3f}", border=True)
        st.metric("留出集 ROC-AUC", f"{selected_model_metrics['ROC-AUC']:.3f}", border=True)
    st.caption(f"训练集 {cv['折数']} 折 × {cv.get('重复次数', 1)} 次重复；"
               f"流程：{cv.get('方法', '历史普通CV')}。按外层平均 AP 选择：{cv['选择模型']}。"
               "当前流程按外层 AP 选模；折间标准差不是均值置信区间。")
    cv_frame = pd.DataFrame({
        "模型": list(cv["AP均值"]),
        "外层 AP 均值": list(cv["AP均值"].values()),
        "外层 AP 标准差": [cv["AP标准差"][name] for name in cv["AP均值"]],
    })
    with st.container(border=True, key="model-evidence"):
        st.subheader("外层交叉验证 · 模型排序能力")
        render_cv_chart(cv)
    render_table(cv_frame, decimal_columns=("外层 AP 均值", "外层 AP 标准差"))
    full_metric_frame = pd.DataFrame(metrics["模型"]).T.reset_index(names="模型")
    metric_frame = full_metric_frame.rename(columns={
        "Accuracy（准确率）": "准确率", "Precision（查准率）": "查准率",
        "Recall（查全率）": "查全率",
        "AP（平均精确率）": "平均精确率 AP",
    })
    metric_frame = metric_frame[[
        "模型", "准确率", "查准率", "查全率",
        "F1", "ROC-AUC", "平均精确率 AP",
    ]]
    st.subheader("历史留出集指标")
    render_table(metric_frame, decimal_columns=("准确率", "查准率", "查全率", "F1", "ROC-AUC", "平均精确率 AP"))
    st.caption("分类指标使用 0.5 阈值；AP 和 ROC-AUC 根据模型分数计算。该留出集曾用于开发，结果仅作历史描述。")
    ci_rows = []
    budget_ratio = metrics.get("名额比例", 0.10)
    ci_labels = {
        "AP": "平均精确率 AP", "ROC-AUC": "ROC-AUC", "Brier": "Brier 分数",
        "Precision": "查准率@0.5", "Recall": "查全率@0.5", "F1": "F1@0.5",
        "Precision@K": f"Precision@{budget_ratio:.0%}",
        "Recall@K": f"Recall@{budget_ratio:.0%}",
    }
    for model_name, values in metrics["模型"].items():
        intervals = values.get("Bootstrap置信区间", {})
        for metric_name, label in ci_labels.items():
            interval = intervals.get(metric_name)
            if interval:
                ci_rows.append({
                    "模型": model_name, "指标": label,
                    "均值": interval["mean"], "95%区间下限": interval["ci_lower"],
                    "95%区间上限": interval["ci_upper"],
                })
    if ci_rows:
        st.subheader("历史留出集 Bootstrap 区间")
        render_table(pd.DataFrame(ci_rows), decimal_columns=("均值", "95%区间下限", "95%区间上限"))
        st.caption("区间条件于已拟合模型，反映当前留出样本的抽样波动；不包含训练与选模不确定性，也不能使其成为盲测。")
    comparison_file = path / "model_comparison_test.csv"
    if comparison_file.is_file():
        st.subheader("交叉验证 AP 配对检验")
        render_table(pd.read_csv(comparison_file), decimal_columns=(
            "模型1均值", "模型2均值", "均值差异", "t统计量", "p值", "校正后p值",
            "差值区间下限", "差值区间上限",
        ))
        if current_method:
            st.caption("新版为 Nadeau–Bengio 近似修正 + Holm 多重校正；差值区间为逐比较近似区间，未检出差异不代表等效。")
        else:
            st.caption("历史普通配对检验未修正相关折与多重比较，不据此判断显著性。")
    for filename, title in [("ablation_summary.csv", "LR 特征消融（共用外层划分）"),
                            ("calibration_cv.csv", "训练集外层校准对照"),
                            ("calibration_comparison.csv", "历史留出集校准对照")]:
        if (path / filename).is_file():
            st.subheader(title)
            st.dataframe(pd.read_csv(path / filename), hide_index=True, width="stretch")
    if (path / "calibration_pr.png").is_file():
        st.image(str(path / "calibration_pr.png"), width="stretch")
    threshold_columns = [
        "模型", "折外F1阈值", "阈值测试集查准率", "阈值测试集查全率", "阈值测试集F1",
    ]
    if all(column in full_metric_frame.columns for column in threshold_columns[1:]):
        threshold_frame = full_metric_frame[threshold_columns].rename(columns={
            "折外F1阈值": "训练集折外选择阈值",
            "阈值测试集查准率": "留出集查准率",
            "阈值测试集查全率": "留出集查全率",
            "阈值测试集F1": "留出集F1",
        })
        st.subheader("训练集折外阈值对照")
        render_table(threshold_frame, decimal_columns=(
            "训练集折外选择阈值", "留出集查准率", "留出集查全率", "留出集F1",
        ))
        st.caption("阈值在训练集折外预测中按 F1 选择，再固定到历史留出集评估；它不代表业务成本最优。")
    else:
        st.info("此旧版报告未包含折外阈值评估。重新运行项目可生成。")
    render_model_charts(predictions, metrics)
    importance["特征"] = importance.feature.map(feature_label)
    importance = importance.rename(columns={"importance": "重要性"})[["特征", "重要性"]].head(15)
    st.subheader("随机森林关注的特征")
    render_feature_chart(importance)
    st.caption("此图来自随机森林，不能用来解释逻辑回归；重要性也不代表因果关系。")

elif selected_page == "模型解释":
    page_header("模型解释", "查看逻辑回归的描述性系数和树模型 SHAP；静态快照不作可靠生存推断。")
    st.caption("新版系数区间固定训练集预处理坐标，仅描述惩罚 LR 的不确定性；数值系数按训练集标准差计，不是因果效应。")
    coefficient_file = path / "logistic_coefficients.csv"
    shap_file = path / "shap_summary.png"
    if coefficient_file.is_file():
        coefficients = pd.read_csv(coefficient_file).rename(columns={
            "feature": "特征", "coefficient": "逻辑回归系数",
            "odds_ratio": "优势比 OR", "ci_lower": "OR 区间下限", "ci_upper": "OR 区间上限",
        })
        coefficients["特征"] = coefficients["特征"].map(feature_label)
        st.subheader("逻辑回归系数与 OR")
        render_table(coefficients.head(30), decimal_columns=(
            "逻辑回归系数", "优势比 OR", "OR 区间下限", "OR 区间上限",
        ))
    else:
        st.info("此演示报告尚无逻辑回归系数结果；重新运行当前训练流程后可查看。")
    if shap_file.is_file():
        st.subheader("树模型 SHAP 汇总")
        st.caption(f"解释对象：{metrics.get('SHAP模型', '历史树模型，未注明')}。不一定是主模型，不包含校准映射。")
        st.image(str(shap_file), caption="SHAP 值描述各特征对模型输出的贡献，不表示因果效应。", width="stretch")
        examples = sorted((path / "shap_examples").glob("waterfall_top*.png"))
        if examples:
            selected_example = st.selectbox("查看高风险客户解释", examples, format_func=lambda item: item.stem)
            st.image(str(selected_example), width="stretch")
    else:
        st.info("此演示报告尚无 SHAP 结果；重新运行当前训练流程后可查看。")
    survival_file = path / "survival_curves.png"
    cox_file = path / "cox_results.csv"
    if survival_file.is_file():
        st.warning("以下为历史探索产物：缺少事件日期、入组与删失机制，不能据此推断未来留存或合同因果效应。")
        st.subheader("按合同类型分层的 Kaplan-Meier 曲线")
        st.image(str(survival_file), width="stretch")
    if cox_file.is_file():
        cox = pd.read_csv(cox_file).rename(columns={
            "covariate": "因素", "Unnamed: 0": "因素", "系数": "Cox 系数", "风险比HR": "风险比 HR",
            "HR下限": "HR 区间下限", "HR上限": "HR 区间上限",
        })
        st.subheader("Cox 比例风险模型")
        render_table(cox, decimal_columns=("Cox 系数", "风险比 HR", "HR 区间下限", "HR 区间上限", "p值"))
        st.caption("HR 描述训练集协变量与流失事件风险的条件关联；仍需满足模型假设，不代表因果效应。")

elif selected_page == "触达排序":
    page_header(f"固定 {metrics['名额比例']:.0%} 名额下的策略对照", "联系人数相同时，哪种排序能找到更多历史流失客户？")
    st.caption(f"以下指标对应主模型：{selected_model}；表格与示例名单可用于比较其他策略。")
    with st.container(horizontal=True, key="kpi-row"):
        st.metric("优先触达名额", f"{int(selected_policy['名额']):,}", border=True)
        st.metric("命中历史流失人数", f"{int(selected_policy['命中流失']):,}", border=True)
        st.metric(
            POLICY_METRIC_LABELS["Precision@K"], f"{selected_policy['Precision@K']:.1%}", border=True,
            help="优先名单中历史流失客户所占比例（Precision@K）。",
        )
        st.metric(
            POLICY_METRIC_LABELS["Lift@K"], f"{selected_policy['Lift@K']:.2f}×", border=True,
            help="优先名单命中率 ÷ 历史留出集整体流失率（Lift@K）。",
        )
    with st.container(border=True, key="policy-evidence"):
        st.subheader("同容量策略对照")
        render_policy_chart(policy)
    shown_policy = policy.copy()
    shown_policy["命中流失"] = shown_policy["命中流失"].map(
        lambda value: f"{value:.0f}" if float(value).is_integer() else f"{value:.2f}"
    )
    render_table(shown_policy, percent_columns=("Precision@K", "Recall@K"), decimal_columns=("Lift@K",))
    st.caption("各策略使用同一历史留出集和相同名额。随机策略为期望值，因此命中人数可能有小数。")
    strategy = st.selectbox("查看策略的历史留出集示例", assignments["策略"].drop_duplicates().tolist())
    sample = assignments.loc[assignments["策略"].eq(strategy)].head(30).copy()
    sample = sample.rename(columns={
        "customerID": "客户编号", "actual_churn": "实际流失标签",
    })
    sample["实际流失标签"] = sample["实际流失标签"].map({0: "否", 1: "是"})
    sample = sample.drop(columns="策略")
    render_table(sample, decimal_columns=("排序分数",))
    if current_method:
        budget = metrics.get("名额敏感性", {})
        st.info(f"情景名额仅在训练 OOF 选择：{budget.get('训练OOF选择名额比例', 0):.0%}，"
                "冻结后才在历史留出集评估；与上方预设10%名额对照是两项不同评估。")
        if (path / "business_scenarios.csv").is_file():
            st.subheader("27 组训练折外业务假设情景")
            st.dataframe(pd.read_csv(path / "business_scenarios.csv"), hide_index=True, width="stretch")
    sensitivity_file = path / "budget_sensitivity.csv"
    curve_file = path / "budget_curve.png"
    if sensitivity_file.is_file():
        st.subheader("联系名额敏感性")
        sensitivity = pd.read_csv(sensitivity_file)
        render_table(sensitivity, percent_columns=("K比例", "Precision@K", "Recall@K"), decimal_columns=("Lift@K",))
        st.caption("净收益按示例成本、客户价值、剩余生命周期和挽留成功率假设估算，不是实测业务收益。")
    if curve_file.is_file():
        st.image(str(curve_file), width="stretch")
    value_sensitivity_file = path / "budget_sensitivity_value.csv"
    value_curve_file = path / "budget_curve_value.png"
    if value_sensitivity_file.is_file():
        st.subheader("期望价值排序对照")
        render_table(pd.read_csv(value_sensitivity_file),
                     percent_columns=("K比例", "Precision@K", "Recall@K"),
                     decimal_columns=("Lift@K",))
        st.caption("此排序将流失风险分数与月费价值代理相乘，用于比较净收益目标；不是命中人数指标。")
    if value_curve_file.is_file():
        st.image(str(value_curve_file), width="stretch")

elif selected_page == "AI风险解读":
    page_header("AI 风险解读与触达建议", "查看优先名单中的客户资料，生成联系前需要核实的事项。")
    settings = ai_settings()
    if settings["CHURN_AI_BASE_URL"] and settings["CHURN_AI_MODEL"]:
        st.caption(f"远程模型已配置：{settings['CHURN_AI_MODEL']}；点击按钮后才发起调用。")
    else:
        st.info("尚未配置远程接口，点击后将生成模板解读。接口配置方法见 README。")
    if ai_context.empty:
        st.info("当前主模型没有入选客户，暂时无法生成逐客解读。请检查报告中的触达名额。")
    else:
        if context_recovered:
            st.info("这份报告缺少完整客户资料，解读仅使用已有排名、分数和在网时长。")
        customer_ids = ai_context["customerID"].astype("string").tolist()
        customer_choices = st.session_state.setdefault("ai_customer_choices", {})
        previous_customer = customer_choices.get(str(path))
        selected_customer = st.selectbox(
            "选择主模型固定容量优先客户", customer_ids,
            index=customer_ids.index(previous_customer) if previous_customer in customer_ids else 0,
            key=f"ai_customer_{path}",
        )
        customer_choices[str(path)] = selected_customer
        selected_row = ai_context.loc[
            ai_context["customerID"].eq(selected_customer)
        ].iloc[0].to_dict()
        level, level_class = risk_level(selected_row["risk_score"])
        st.markdown(
            f'<div class="risk-card {level_class}"><div class="risk-label">当前客户风险等级</div>'
            f'<div class="risk-value">{escape(level)}</div><div class="risk-detail">'
            f'排序名次 #{int(selected_row["rank"])} · 排序分数 {float(selected_row["risk_score"]):.4f} · '
            f'主模型 {escape(str(selected_row["model_name"]))}</div></div>',
            unsafe_allow_html=True,
        )
        with st.expander("客户资料与分群统计"):
            render_table(ai_display_frame(selected_row))
        selected_calibration = metrics.get("校准决策", {}).get(selected_model, {})
        if selected_calibration.get("使用校准模型"):
            score_note = "分数采用训练内 sigmoid 校准，但不代表未来流失或实际干预成功概率。"
        elif "使用校准模型" in selected_calibration:
            score_note = "外层 CV 判定校准未改善 Brier，当前使用原始概率；不代表未来流失或实际干预成功概率。"
        else:
            score_note = "历史报告排序分数未经概率校准。"
        st.caption("发送给 AI 的资料不含真实流失标签。" + score_note)
        explanations = st.session_state.setdefault("ai_explanations", {})
        # 客户资料变化后，不再展示旧资料对应的解读。
        explanation_key = (str(path), json.dumps(selected_row, sort_keys=True, ensure_ascii=False))
        if st.button("生成 AI 风险解读", type="primary"):
            with st.spinner("正在生成解读…"):
                explanations[explanation_key] = generate_explanation(selected_row, settings=settings)
        if explanation_key in explanations:
            with st.container(key="ai-analysis"):
                render_ai_explanation(explanations[explanation_key])
