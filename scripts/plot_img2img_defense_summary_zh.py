# -*- coding: utf-8 -*-
"""绘制 clean img2img diffusion 防御诊断汇总图。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制 img2img defense summary。")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def setup_chinese_font() -> font_manager.FontProperties:
    for path in [
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]:
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            prop = font_manager.FontProperties(fname=str(path))
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["font.sans-serif"] = [prop.get_name()]
            plt.rcParams["axes.unicode_minus"] = False
            return prop
    raise FileNotFoundError("没有找到可用中文字体。")


def main() -> None:
    args = parse_args()
    prop = setup_chinese_font()
    csv_path = Path(args.csv).resolve()
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: (row["mode"], int(row["start_timestep"])))
    output = Path(args.output).resolve() if args.output else csv_path.with_suffix(".png")

    metrics = [
        ("generated_nonzero_mean", "平均生成非零包数"),
        ("overhead_ratio_mean", "平均额外包比例"),
        ("original_flipped_mean", "原始位置平均方向翻转数"),
        ("full_sequence_l1_units_mean", "全序列平均 L1"),
    ]
    colors = {"condition_clean": "#1f77b4", "zero_condition": "#d62728"}
    labels = {"condition_clean": "条件 c=clean", "zero_condition": "零条件 c=0"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2))
    fig.subplots_adjust(top=0.88, bottom=0.1, wspace=0.28, hspace=0.35)
    fig.suptitle("Clean x0 -> q(x_t|x0) -> diffusion 恢复诊断（256 条 test clean）", fontproperties=prop, fontsize=14)
    for ax, (key, title) in zip(axes.ravel(), metrics):
        for mode in ["condition_clean", "zero_condition"]:
            sub = [row for row in rows if row["mode"] == mode]
            xs = [int(row["start_timestep"]) for row in sub]
            ys = [float(row[key]) for row in sub]
            ax.plot(xs, ys, marker="o", label=labels[mode], color=colors[mode])
        ax.set_title(title, fontproperties=prop)
        ax.set_xlabel("起始加噪时间步 t", fontproperties=prop)
        ax.grid(alpha=0.25)
        ax.legend(prop=prop)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(prop)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
