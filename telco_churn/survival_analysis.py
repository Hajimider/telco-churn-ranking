"""生存分析模块：Kaplan-Meier 曲线、Log-rank 检验和 Cox 比例风险模型。"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test


def survival_analysis(train_frame, output_dir):
    """
    使用训练集进行生存分析。

    - tenure 作为生存时间
    - Churn 作为事件（1=流失，0=删失）
    - 按 Contract 分层绘制 KM 曲线
    - Log-rank 检验比较不同合同类型的生存曲线
    - Cox 模型识别风险因素

    返回：
    - km_results: 各组的中位生存时间
    - logrank_result: Log-rank 检验结果
    - cox_results: Cox 模型系数、HR 及置信区间
    """
    # 准备生存数据
    survival_data = train_frame[["tenure", "Churn", "Contract"]].copy()
    survival_data = survival_data.rename(columns={"tenure": "T", "Churn": "E"})

    # Kaplan-Meier 曲线（按合同类型分层）
    fig, ax = plt.subplots(figsize=(10, 6))
    kmf = KaplanMeierFitter()

    km_results = []
    for contract_type in sorted(survival_data["Contract"].unique()):
        subset = survival_data[survival_data["Contract"] == contract_type]
        kmf.fit(subset["T"], subset["E"], label=contract_type)
        kmf.plot_survival_function(ax=ax, ci_show=True)

        median_survival = kmf.median_survival_time_
        km_results.append({
            "Contract": contract_type,
            "样本数": len(subset),
            "中位生存时间": median_survival if np.isfinite(median_survival) else None
        })

    ax.set_xlabel("在网时长（月）")
    ax.set_ylabel("留存率")
    ax.set_title("按合同类型分层的 Kaplan-Meier 生存曲线")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / "survival_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    km_results_df = pd.DataFrame(km_results)

    # Log-rank 检验
    logrank_result = multivariate_logrank_test(
        survival_data["T"],
        survival_data["Contract"],
        survival_data["E"]
    )

    logrank_summary = {
        "检验统计量": float(logrank_result.test_statistic),
        "p值": float(logrank_result.p_value),
        "自由度": int(logrank_result.degrees_of_freedom)
    }

    # Cox 比例风险模型
    # 准备 Cox 模型数据（添加其他协变量）
    cox_data = train_frame[[
        "tenure", "Churn", "Contract", "MonthlyCharges",
        "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity"
    ]].copy()

    cox_data = cox_data.rename(columns={"tenure": "T", "Churn": "E"})

    # One-hot 编码类别变量
    cox_data = pd.get_dummies(cox_data, columns=[
        "Contract", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity"
    ], drop_first=True)

    # 拟合 Cox 模型
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_data, duration_col="T", event_col="E")

    # 提取结果
    cox_summary = cph.summary.copy()
    cox_summary = cox_summary.rename(columns={
        "coef": "系数",
        "exp(coef)": "风险比HR",
        "se(coef)": "标准误",
        "coef lower 95%": "系数下限",
        "coef upper 95%": "系数上限",
        "exp(coef) lower 95%": "HR下限",
        "exp(coef) upper 95%": "HR上限",
        "z": "z统计量",
        "p": "p值",
        "-log2(p)": "-log2(p)"
    })

    cox_summary = cox_summary[[
        "系数", "风险比HR", "HR下限", "HR上限", "p值"
    ]].sort_values("风险比HR", ascending=False)

    cox_summary.to_csv(output_dir / "cox_results.csv", encoding="utf-8-sig")

    return km_results_df, logrank_summary, cox_summary
