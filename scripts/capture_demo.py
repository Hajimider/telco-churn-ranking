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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", help="Optional path to an already installed Chromium executable")
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.browser)
        page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
        page.goto("http://127.0.0.1:8501", wait_until="domcontentloaded")
        page.get_by_role("heading", name="电信客户流失预测与排序", exact=True).wait_for(timeout=60000)
        report = page.get_by_role("combobox", name="分析报告")
        report.click()
        page.get_by_role("option", name="run_validated", exact=True).click()
        page.get_by_text("0.669", exact=True).wait_for(timeout=60000)
        page.evaluate("document.fonts.ready")
        for label, filename in PAGES:
            height = {"报告总览": 800, "客户差异": 1200, "模型验证": 1500,
                      "模型解释": 1400, "触达排序": 1300, "AI风险解读": 850}[label]
            page.set_viewport_size({"width": 1440, "height": height})
            page.get_by_test_id("stRadioOption").filter(has_text=label).click()
            title = {"报告总览": "电信客户流失预测与排序", "客户差异": "训练集客户分群",
                     "触达排序": "固定 10% 名额下的策略对照",
                     "AI风险解读": "AI 风险解读与触达建议"}.get(label, label)
            page.get_by_role("heading", name=title, exact=True).wait_for(timeout=60000)
            page.wait_for_timeout(1200)
            page.evaluate("document.fonts.ready")
            # Do not click the AI generation button: configuration may target a remote service.
            errors = page.locator('[data-testid="stException"], [data-testid="stAlertContentError"]')
            if errors.count():
                raise RuntimeError(f"Dashboard error on {label}: {errors.all_text_contents()}")
            if not report.input_value() and "run_validated" not in report.locator("..").inner_text():
                # BaseWeb may render the selected option beside an empty input.
                if "run_validated" not in page.locator('[data-testid="stSidebar"]').inner_text():
                    raise RuntimeError("Selected report is not run_validated")
            page.screenshot(path=str(ROOT / "docs" / filename), full_page=True)
            print(f"Captured {label}: {filename}")
        browser.close()


if __name__ == "__main__":
    main()
