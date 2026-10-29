"""为 Matplotlib 注册项目内的中文字体。"""
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager


def configure_chart_font():
    font_path = Path(__file__).resolve().parent / "assets" / "fonts" / "NotoSansSC-Regular.otf"
    if not any(Path(font.fname) == font_path for font in font_manager.fontManager.ttflist):
        font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=str(font_path)).get_name()
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [family]
    mpl.rcParams["axes.unicode_minus"] = False
    return family
