"""名额敏感性分析和收益模型。"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def budget_sensitivity_analysis(y_test, scores, test_ids, output_dir,
                                 k_range=None, cost_per_contact=50,
                                 avg_monthly_value=60, avg_remaining_months=12,
                                 retention_success_rate=0.3):
    """
    计算不同联系名额下的 Precision@K、Recall@K、Lift@K 和期望净收益。

    参数:
    - y_test: 测试集真实标签
    - scores: 模型排序分数
    - test_ids: 客户 ID
    - k_range: K 的比例范围，默认 [0.05, 0.10, 0.15, ..., 0.50]
    - cost_per_contact: 每人挽留成本（元）
    - avg_monthly_value: 客户月均价值（元）
    - avg_remaining_months: 平均剩余生命周期（月）
    - retention_success_rate: 挽留成功率

    返回:
    - sensitivity_df: 不同 K 下的指标表
    - optimal_k: 期望净收益最大的 K
    """
    if k_range is None:
        k_range = np.arange(0.05, 0.51, 0.05)

    y_test = np.asarray(y_test, dtype=int)
    scores = np.asarray(scores, dtype=float)
    test_ids = np.asarray(test_ids, dtype=str)

    # 按分数和 ID 排序
    order = np.lexsort((test_ids, -scores))
    y_sorted = y_test[order]

    n_test = len(y_test)
    n_positives = int(y_test.sum())
    prevalence = n_positives / n_test if n_test > 0 else 0

    results = []
    for k_ratio in k_range:
        k_count = int(n_test * k_ratio)
        if k_count == 0:
            continue

        hits = int(y_sorted[:k_count].sum())
        precision_k = hits / k_count
        recall_k = hits / n_positives if n_positives > 0 else 0
        lift_k = precision_k / prevalence if prevalence > 0 else 0

        # 收益计算
        # 假设挽留成功的客户继续留存，带来的价值是月均价值 × 剩余月数
        total_cost = cost_per_contact * k_count
        expected_retained = hits * retention_success_rate
        expected_value = expected_retained * avg_monthly_value * avg_remaining_months
        net_profit = expected_value - total_cost

        results.append({
            "K比例": k_ratio,
            "K人数": k_count,
            "命中流失": hits,
            "Precision@K": precision_k,
            "Recall@K": recall_k,
            "Lift@K": lift_k,
            "挽留成本": total_cost,
            "期望价值": expected_value,
            "期望净收益": net_profit
        })

    sensitivity_df = pd.DataFrame(results)
    sensitivity_df.to_csv(output_dir / "budget_sensitivity.csv", index=False, encoding="utf-8-sig")

    # 找到最优 K
    optimal_row = sensitivity_df.loc[sensitivity_df["期望净收益"].idxmax()]
    optimal_k = optimal_row["K比例"]

    # 绘制敏感性曲线
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Precision-Recall-K
    ax = axes[0, 0]
    ax.plot(sensitivity_df["K比例"], sensitivity_df["Precision@K"], marker="o", label="Precision@K")
    ax.plot(sensitivity_df["K比例"], sensitivity_df["Recall@K"], marker="s", label="Recall@K")
    ax.axvline(optimal_k, color="red", linestyle="--", label=f"最优K={optimal_k:.2f}")
    ax.set_xlabel("联系名额比例 K")
    ax.set_ylabel("指标值")
    ax.set_title("Precision 和 Recall 随名额变化")
    ax.legend()
    ax.grid(alpha=0.3)

    # Lift@K
    ax = axes[0, 1]
    ax.plot(sensitivity_df["K比例"], sensitivity_df["Lift@K"], marker="o", color="green")
    ax.axvline(optimal_k, color="red", linestyle="--", label=f"最优K={optimal_k:.2f}")
    ax.set_xlabel("联系名额比例 K")
    ax.set_ylabel("Lift@K")
    ax.set_title("提升倍数随名额变化")
    ax.legend()
    ax.grid(alpha=0.3)

    # 净收益曲线
    ax = axes[1, 0]
    ax.plot(sensitivity_df["K比例"], sensitivity_df["期望净收益"] / 10000, marker="o", color="purple")
    ax.axvline(optimal_k, color="red", linestyle="--", label=f"最优K={optimal_k:.2f}")
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("联系名额比例 K")
    ax.set_ylabel("期望净收益（万元）")
    ax.set_title("期望净收益随名额变化")
    ax.legend()
    ax.grid(alpha=0.3)

    # 成本与价值对比
    ax = axes[1, 1]
    ax.plot(sensitivity_df["K比例"], sensitivity_df["挽留成本"] / 10000, marker="o", label="挽留成本")
    ax.plot(sensitivity_df["K比例"], sensitivity_df["期望价值"] / 10000, marker="s", label="期望价值")
    ax.axvline(optimal_k, color="red", linestyle="--", label=f"最优K={optimal_k:.2f}")
    ax.set_xlabel("联系名额比例 K")
    ax.set_ylabel("金额（万元）")
    ax.set_title("成本与价值对比")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("联系名额敏感性分析", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "budget_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return sensitivity_df, optimal_k
