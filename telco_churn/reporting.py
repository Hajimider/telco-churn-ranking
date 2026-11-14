"""报告诊断图、校准对照与运行溯源。"""
import hashlib
import importlib.metadata
import platform
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve


def run_metadata(frame, seed):
    from .pipeline import ROOT
    packages = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "lightgbm", "optuna"]
    result = {"python": sys.version, "platform": platform.platform(), "seed": seed,
              "versions": {name: importlib.metadata.version(name) for name in packages},
              "数据SHA256": hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()}
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, timeout=5, stderr=subprocess.DEVNULL,
            text=True).strip()
        result["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, timeout=5,
            stderr=subprocess.DEVNULL, text=True).strip())
    except (OSError, subprocess.SubprocessError):
        result["git_commit"] = "unavailable"
        result["git_dirty"] = None
    return result


def render_diagnostics(calibration, pr, comparison, prevalence, output):
    """从报告底层表重绘，不重新训练、不改变指标。"""
    from matplotlib.ticker import PercentFormatter
    from .chart_font import configure_chart_font

    configure_chart_font()
    canvas, surface = "#f5f1eb", "#fffdf9"
    ink, muted, border = "#243b3a", "#596560", "#ded6cb"
    colors = {"逻辑回归": "#006b40", "随机森林": "#bf754e", "LightGBM": "#2465ad"}
    styles = {"逻辑回归": "-", "随机森林": "--", "LightGBM": "-."}
    names = [name for name in colors if name in set(comparison["模型"])]
    with plt.rc_context({"font.size": 12, "text.color": ink, "axes.labelcolor": muted,
                         "xtick.color": muted, "ytick.color": muted,
                         "savefig.facecolor": canvas}):
        fig = plt.figure(figsize=(14, 9.2), facecolor=canvas)
        grid = fig.add_gridspec(2, 6, left=.065, right=.97, top=.805, bottom=.105,
                               wspace=.9, hspace=.6, height_ratios=[1, 1.12])
        fig.text(.065, .957, "模型验证 / 概率质量与排序能力", fontsize=24, weight="bold", color=ink)
        fig.text(.065, .914, "固定测试集 · 校准对照 + Precision–Recall · 同一份报告数据", fontsize=14, color=muted)
        fig.text(.065, .867, "01  校准曲线", fontsize=15, weight="bold")
        fig.text(.27, .867, "越接近对角线越好；虚线为原始分数，实线为 sigmoid 校准", fontsize=12, color=muted)

        def setup(ax):
            ax.set_facecolor(surface)
            ax.set_axisbelow(True)
            ax.grid(color=border, alpha=.5, linewidth=.65)
            ax.spines[["top", "right"]].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(border)
            ax.tick_params(length=0, pad=7, labelsize=11)
            ax.set(xlim=(0, 1), ylim=(0, 1.03))
            ax.set_xticks([0, .25, .5, .75, 1])
            ax.set_yticks([0, .25, .5, .75, 1])
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

        for index, name in enumerate(names):
            ax = fig.add_subplot(grid[0, index * 2:index * 2 + 2])
            setup(ax)
            ax.plot([0, 1], [0, 1], color="#a7aaa2", linestyle=":", linewidth=1.1, zorder=1)
            data = calibration.loc[calibration["模型"].eq(name)]
            for state, linestyle, marker in [("原始", "--", "o"), ("sigmoid", "-", "s")]:
                rows = data.loc[data["状态"].eq(state)].sort_values("平均预测")
                ax.plot(rows["平均预测"], rows["实际比例"], color=colors[name],
                        linestyle=linestyle, linewidth=1.4 if state == "原始" else 2.2,
                        marker=marker, markersize=4.8,
                        markerfacecolor=surface if state == "原始" else colors[name],
                        markeredgewidth=1, alpha=.65 if state == "原始" else 1,
                        label="原始" if state == "原始" else "sigmoid 校准")
            ax.set_title(name, loc="left", fontsize=15, weight="bold", color=ink, pad=12)
            ax.set_xlabel("平均预测分数", labelpad=8)
            if index == 0:
                ax.set_ylabel("实际流失比例", labelpad=8)
            ax.legend(loc="upper left", frameon=False, fontsize=10.5)

        ax = fig.add_subplot(grid[1, :4])
        setup(ax)
        ax.set_title("02  PR 曲线 / 排序表现", loc="left", fontsize=15, weight="bold", color=ink, pad=15)
        for name in names:
            rows = pr.loc[pr["模型"].eq(name)]
            ap = comparison.loc[comparison["模型"].eq(name) & comparison["状态"].eq("sigmoid"), "AP"].iloc[0]
            ax.plot(rows["Recall"], rows["Precision"], color=colors[name],
                    linestyle=styles[name], linewidth=1.8, label=f"{name}  AP {ap:.3f}")
        ax.axhline(prevalence, color="#899087", linewidth=1.1, linestyle=":")
        ax.annotate(f"随机基线 {prevalence:.1%}", (.99, prevalence), xytext=(-4, 7),
                    textcoords="offset points", ha="right", color=muted, fontsize=11,
                    bbox={"facecolor": surface, "edgecolor": "none", "pad": 2})
        ax.set(xlabel="查全率 Recall", ylabel="查准率 Precision")
        ax.legend(loc="upper right", frameon=False, fontsize=11)

        table_ax = fig.add_subplot(grid[1, 4:])
        table_ax.set_facecolor(surface)
        table_ax.set_xticks([])
        table_ax.set_yticks([])
        for spine in table_ax.spines.values():
            spine.set_color(border)
        table_ax.text(.07, .91, "03  Brier 分数", fontsize=15, weight="bold", transform=table_ax.transAxes)
        table_ax.text(.07, .81, "越低越好 · 校准不一定改善结果", fontsize=11, color=muted, transform=table_ax.transAxes)
        for x, label in [(.07, "模型"), (.51, "原始"), (.75, "校准后")]:
            table_ax.text(x, .66, label, fontsize=11, color=muted, transform=table_ax.transAxes)
        for index, name in enumerate(names):
            y = .52 - index * .16
            rows = comparison.loc[comparison["模型"].eq(name)].set_index("状态")
            table_ax.text(.07, y, name, fontsize=12, transform=table_ax.transAxes)
            table_ax.text(.51, y, f"{rows.loc['原始', 'Brier']:.4f}", fontsize=12, transform=table_ax.transAxes)
            table_ax.text(.75, y, f"{rows.loc['sigmoid', 'Brier']:.4f}", fontsize=12, transform=table_ax.transAxes)
            table_ax.plot([.07, .93], [y - .055, y - .055], color=border, linewidth=.6, transform=table_ax.transAxes)
        fig.text(.065, .039, "读图提示：AP 衡量排序，Brier 衡量概率误差；两者不能相互替代。仅评估历史标签，不表示因果或实际挽留收益。",
                 fontsize=11, color=muted)
        fig.savefig(output / "calibration_pr.png", dpi=180, facecolor=canvas)
        plt.close(fig)


def redraw_report(output):
    """使用既有 CSV 更新图表，保留原始模型、预测及统计结果。"""
    from pathlib import Path
    output = Path(output)
    predictions = pd.read_csv(output / "predictions.csv")
    render_diagnostics(pd.read_csv(output / "calibration_bins.csv"),
                       pd.read_csv(output / "pr_curve.csv"),
                       pd.read_csv(output / "calibration_comparison.csv"),
                       float(predictions["actual_churn"].mean()), output)


def export_diagnostics(validation, X_test, y_test, raw_models, calibrated_models, output):
    """图形采用线型/标记区分，所有数值同步保存 CSV。"""
    calibration_rows, curve_rows, summary = [], [], []
    for name, model in calibrated_models.items():
        p = model.predict_proba(X_test)[:, 1]
        raw = raw_models[name].predict_proba(X_test)[:, 1]
        for state, scores in [("原始", raw), ("sigmoid", p)]:
            summary.append({"模型": name, "状态": state,
                            "AP": average_precision_score(y_test, scores),
                            "Brier": brier_score_loss(y_test, scores)})
            observed, predicted = calibration_curve(y_test, scores, n_bins=10, strategy="quantile")
            calibration_rows.extend({"模型": name, "状态": state, "平均预测": a, "实际比例": b}
                                    for a, b in zip(predicted, observed))
        precision, recall, _ = precision_recall_curve(y_test, p)
        curve_rows.extend({"模型": name, "Recall": r, "Precision": v} for r, v in zip(recall, precision))
    pd.DataFrame(summary).to_csv(output / "calibration_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(calibration_rows).to_csv(output / "calibration_bins.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curve_rows).to_csv(output / "pr_curve.csv", index=False, encoding="utf-8-sig")
    render_diagnostics(pd.DataFrame(calibration_rows), pd.DataFrame(curve_rows),
                       pd.DataFrame(summary), float(np.mean(y_test)), output)
    validation["folds"].groupby("模型")[["原始Brier", "校准Brier", "原始AP", "AP"]].mean().to_csv(
        output / "calibration_cv.csv", encoding="utf-8-sig")
