"""Capture the running current dashboard; never launch a server or contact remote AI.

Start the project preview server first, then run this file from any directory.
Requires the user's existing Playwright/Chromium installation.
"""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PAGES = [
    ("报告总览", "demo-v3-overview.png"),
    ("客户差异", "demo-v3-segments.png"),
    ("模型验证", "demo-v3-validation.png"),
    ("模型解释", "demo-v3-explanation.png"),
    ("触达排序", "demo-v3-policy.png"),
    ("AI风险解读", "demo-v3-ai.png"),
]


def wait_for_render(page):
    """桌面和窄屏使用相同的完成条件，避免保存半加载画面。"""
    page.wait_for_function(
        """() => !document.querySelector('[data-testid="stSpinner"], [data-stale="true"]') &&
        Array.from(document.images).every(image => image.complete && image.naturalWidth > 0)""",
        timeout=60000,
    )
    page.evaluate("document.fonts.ready")
    errors = page.locator('[data-testid="stException"], [data-testid="stAlertContentError"]')
    if errors.count():
        raise RuntimeError(f"Dashboard error: {errors.all_text_contents()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", help="Optional path to an already installed Chromium executable")
    parser.add_argument("--report", default="run_final", help="Report directory name selected in the dashboard")
    parser.add_argument("--url", default="http://127.0.0.1:8501", help="Dashboard URL")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs", help="Screenshot destination")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.browser)
        page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
        page.goto(args.url, wait_until="domcontentloaded")
        page.get_by_role("heading", name="电信客户流失预测与排序", exact=True).wait_for(timeout=60000)
        report = page.get_by_role("combobox", name="分析报告")
        report.click()
        page.get_by_role("option", name=args.report, exact=True).click()
        selected_value = report.input_value()
        # 新版下拉框把选中值放在相邻元素，输入框只负责搜索。
        selected_text = report.locator('xpath=ancestor::*[@data-testid="stSelectbox"]').inner_text()
        if selected_value != args.report and args.report not in selected_text:
            raise RuntimeError(f"Selected report is not {args.report}")
        page.evaluate("document.fonts.ready")
        for label, filename in PAGES:
            page.set_viewport_size({"width": 1440, "height": 1100})
            page.locator("label", has_text=label).first.click()
            title = {"报告总览": "电信客户流失预测与排序", "客户差异": "训练集客户分群",
                     "触达排序": "固定 10% 名额下的策略对照",
                     "AI风险解读": "AI 风险解读与触达建议"}.get(label, label)
            page.get_by_role("heading", name=title, exact=True).wait_for(timeout=60000)
            bottom_marker = {
                "报告总览": "数据来源与评估口径",
                "客户差异": "合同类型 × 在网时长",
                "模型验证": "随机森林关注的特征",
                "模型解释": "树模型 SHAP 汇总",
                "触达排序": "期望价值排序对照",
            }.get(label)
            if bottom_marker:
                page.get_by_text(bottom_marker, exact=True).wait_for(timeout=60000)
            page.wait_for_timeout(500)
            wait_for_render(page)
            # Do not click the AI generation button: configuration may target a remote service.
            page.screenshot(path=str(args.output_dir / filename), full_page=True)
            print(f"Captured {label}: {filename}")
        page.locator("label", has_text="报告总览").first.click()
        page.get_by_role("heading", name="电信客户流失预测与排序", exact=True).wait_for(timeout=60000)
        page.get_by_text("数据来源与评估口径", exact=True).wait_for(timeout=60000)
        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(500)
        wait_for_render(page)
        page.screenshot(path=str(args.output_dir / "demo-mobile.png"), full_page=True)
        browser.close()


if __name__ == "__main__":
    main()
