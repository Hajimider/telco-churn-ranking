"""历史标签上的名额与收益情景，不估计干预增量或实际收入。"""
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def budget_table(labels, scores, ids, monthly_values, k_range=None,
                 cost_per_contact=50, avg_remaining_months=12, retention_success_rate=0.3,
                 rank_by_value=False):
    y, score = np.asarray(labels), np.asarray(scores, dtype=float)
    ids, values = np.asarray(ids, dtype=str), np.asarray(monthly_values, dtype=float)
    if y.ndim != 1 or not len(y) or any(a.shape != y.shape for a in (score, ids, values)):
        raise ValueError("标签、分数、客户编号、月值须为等长非空一维数组。")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(score).all():
        raise ValueError("标签须二值，排序分数须有限。")
    if len(set(ids)) != len(ids) or any(not item.strip() for item in ids):
        raise ValueError("客户编号须唯一且非空。")
    assumptions = np.asarray([cost_per_contact, avg_remaining_months, retention_success_rate], float)
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.isfinite(assumptions).all() or (assumptions < 0).any()
            or retention_success_rate > 1):
        raise ValueError("月值及业务假设须有限非负，成功率须在 [0,1]。")
    ratios = np.asarray(np.linspace(0, 1, 21) if k_range is None else k_range, float)
    if ratios.ndim != 1 or not len(ratios) or not np.isfinite(ratios).all() or ((ratios < 0) | (ratios > 1)).any():
        raise ValueError("名额比例须为 [0,1] 内的非空一维序列。")
    ratios = np.round(np.unique(np.r_[0.0, ratios, 1.0]), 4)
    sort_key = score * values if rank_by_value else score
    order = np.lexsort((ids, -sort_key))
    cumulative_hits = np.r_[0, np.cumsum(y[order])]
    cumulative_value = np.r_[0, np.cumsum(y[order] * values[order])]
    prevalence = float(y.mean())
    rows = []
    for ratio in ratios:
        count = int(len(y) * ratio)
        hits = int(cumulative_hits[count])
        precision = hits / count if count else 0.0
        value = float(cumulative_value[count] * avg_remaining_months * retention_success_rate)
        cost = float(count * cost_per_contact)
        rows.append({"K比例": float(ratio), "K人数": count, "命中流失": hits,
                     "Precision@K": precision, "Recall@K": hits / y.sum() if y.sum() else 0.0,
                     "Lift@K": precision / prevalence if prevalence else 0.0,
                     "挽留成本": cost, "期望价值": value, "期望净收益": value - cost})
    return pd.DataFrame(rows)


def select_budget(table):
    """只对训练折外情景调用；并列收益优先更少联系。"""
    row = table.sort_values(["期望净收益", "K比例"], ascending=[False, True]).iloc[0]
    return float(row["K比例"])


def business_scenarios(labels, scores, ids, monthly_values, rank_by_value=False):
    rows = []
    for cost, success, months in product([20, 50, 100], [0.1, 0.2, 0.3], [6, 12, 24]):
        table = budget_table(labels, scores, ids, monthly_values, cost_per_contact=cost,
                             retention_success_rate=success, avg_remaining_months=months,
                             rank_by_value=rank_by_value)
        ratio = select_budget(table)
        row = table.loc[table["K比例"].eq(ratio)].iloc[0]
        rows.append({"触达成本": cost, "假设成功率": success, "假设剩余月数": months,
                     "训练OOF选择K": ratio, "训练OOF情景净收益": float(row["期望净收益"])})
    return pd.DataFrame(rows)


def budget_sensitivity_analysis(y_test, scores, test_ids, output_dir,
                                k_range=None, cost_per_contact=50, avg_monthly_value=60,
                                avg_remaining_months=12, retention_success_rate=0.3,
                                monthly_values=None, selected_k=None, rank_by_value=False,
                                output_name="budget_sensitivity.csv", curve_name="budget_curve.png"):
    """评估冻结 K；缺少训练选择结果时只输出描述曲线，不从测试标签选 K。"""
    values = np.full(len(y_test), avg_monthly_value) if monthly_values is None else monthly_values
    table = budget_table(y_test, scores, test_ids, values, k_range, cost_per_contact,
                         avg_remaining_months, retention_success_rate, rank_by_value=rank_by_value)
    table.to_csv(output_dir / output_name, index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(table["K比例"], table["Precision@K"], color="black", label="Precision@K")
    axes[0].plot(table["K比例"], table["Recall@K"], color="black", linestyle="--", label="Recall@K")
    axes[0].set(ylabel="历史标签命中比例", title="测试集描述曲线（不选名额）")
    axes[0].legend()
    axes[1].plot(table["K比例"], table["期望净收益"], color="black",
                 label="期望价值排序" if rank_by_value else "纯风险排序")
    axes[1].axhline(0, color="gray", linestyle=":")
    axes[1].set(ylabel="假设情景净收益（非实测，元）", title="历史标签收益情景")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("联系名额比例")
        if selected_k is not None:
            ax.axvline(selected_k, color="gray", linestyle="--", label="训练OOF冻结K")
        ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_dir / curve_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return table, selected_k
