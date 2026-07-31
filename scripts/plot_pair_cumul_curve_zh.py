# -*- coding: utf-8 -*-
"""按 CUMUL 方法绘制单条流量的累计表示曲线。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.data.cw import stored_npy_from_npz

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制 Panchenko et al. CUMUL 累计表示。")
    parser.add_argument("--run-dir", required=True, help="purifier run 目录。")
    parser.add_argument("--clean-path", default="D:/learning/TOR/datasets/CW/CW.npz")
    parser.add_argument("--row-index", type=int, default=0, help="purified manifest 中的行号。")
    parser.add_argument("--samples", type=int, default=100, help="CUMUL 等距采样点数量。")
    parser.add_argument("--output-dir", default="", help="默认写入 <run-dir>/visualizations/pair_curves。")
    parser.add_argument(
        "--sign-convention",
        choices=["user", "paper"],
        default="user",
        help="user: 沿用当前数据约定，上行/raw>0 为正；paper: 按论文入站为正、出站为负，等价于对当前数据取反。",
    )
    return parser.parse_args()


def setup_chinese_font() -> font_manager.FontProperties:
    candidates = [
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for path in candidates:
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            prop = font_manager.FontProperties(fname=str(path))
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["font.sans-serif"] = [prop.get_name()]
            plt.rcParams["axes.unicode_minus"] = False
            return prop
    raise FileNotFoundError("没有找到可用中文字体。")


def read_manifest_row(path: Path, row_index: int) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if index == int(row_index):
                return row
    raise IndexError(f"manifest 行号越界: {row_index}")


def load_clean(clean_path: Path, clean_index: int) -> tuple[np.ndarray, int | None]:
    x_map = stored_npy_from_npz(clean_path, "X")
    y_map = stored_npy_from_npz(clean_path, "y")
    if x_map is not None:
        clean = np.asarray(x_map[int(clean_index)], dtype=np.float32).reshape(-1)
        label = int(np.asarray(y_map[int(clean_index)], dtype=np.int64)) if y_map is not None else None
        return clean, label
    with np.load(clean_path, allow_pickle=False) as payload:
        clean = np.asarray(payload["X"][int(clean_index)], dtype=np.float32).reshape(-1)
        label = int(payload["y"][int(clean_index)]) if "y" in payload.files else None
        return clean, label


def load_defended(row: dict[str, str]) -> np.ndarray:
    with np.load(row["defended_path"], allow_pickle=False) as payload:
        flat = np.asarray(payload["flat"], dtype=np.float32)
        offsets = np.asarray(payload["offsets"], dtype=np.int64)
        local = int(row["defended_local_index"])
        return flat[int(offsets[local]) : int(offsets[local + 1])]


def load_purified(row: dict[str, str]) -> np.ndarray:
    with np.load(row["purified_path"], allow_pickle=False) as payload:
        return np.asarray(payload["X"][int(row["purified_index"])], dtype=np.float32).reshape(-1)


def cumul_curve(values: np.ndarray, *, samples: int, sign_multiplier: float) -> dict[str, np.ndarray | int | float]:
    nonzero = np.asarray(values, dtype=np.float32).reshape(-1)
    nonzero = nonzero[nonzero != 0]
    # 当前数据只有方向符号有意义；按 Tor cell 固定大小场景，将每个包大小归一为 1。
    p = np.sign(nonzero).astype(np.float32) * float(sign_multiplier)
    a = np.concatenate([[0.0], np.cumsum(np.abs(p), dtype=np.float32)])
    c = np.concatenate([[0.0], np.cumsum(p, dtype=np.float32)])
    if len(a) > 1:
        sampled_a = np.linspace(0.0, float(a[-1]), int(samples), dtype=np.float32)
        sampled_c = np.interp(sampled_a, a, c).astype(np.float32)
    else:
        sampled_a = np.zeros(int(samples), dtype=np.float32)
        sampled_c = np.zeros(int(samples), dtype=np.float32)
    return {
        "a": a,
        "c": c,
        "sampled_a": sampled_a,
        "sampled_c": sampled_c,
        "pos": int(np.sum(p > 0)),
        "neg": int(np.sum(p < 0)),
        "nonzero": int(len(p)),
        "final_cum": float(c[-1]) if len(c) else 0.0,
        "max_a": float(a[-1]) if len(a) else 0.0,
    }


def apply_axis_style(ax, title: str, xlabel: str, font_prop: font_manager.FontProperties) -> None:
    ax.set_title(title, fontproperties=font_prop, fontsize=12)
    ax.set_xlabel(xlabel, fontproperties=font_prop)
    ax.set_ylabel("累计带符号包大小 c_i", fontproperties=font_prop)
    ax.axhline(0, color="black", lw=0.8, alpha=0.35)
    ax.grid(True, alpha=0.25)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)


def plot_sampled(ax, item: dict[str, np.ndarray | int | float], color: str, label: str) -> None:
    ax.plot(item["a"], item["c"], color=color, lw=1.25, alpha=0.55)
    ax.scatter(item["sampled_a"], item["sampled_c"], color=color, s=14, label=label, alpha=0.85, zorder=3)


def summary_item(item: dict[str, np.ndarray | int | float]) -> dict[str, int | float]:
    return {
        "positive_packets_under_plot_convention": int(item["pos"]),
        "negative_packets_under_plot_convention": int(item["neg"]),
        "nonzero_packets": int(item["nonzero"]),
        "final_cumulative_value": float(item["final_cum"]),
        "max_cumulative_abs_packet_size": float(item["max_a"]),
    }


def main() -> None:
    args = parse_args()
    font_prop = setup_chinese_font()
    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "manifests" / "purified_dataset_manifest.csv"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "visualizations" / "pair_curves"
    output_dir.mkdir(parents=True, exist_ok=True)

    row = read_manifest_row(manifest_path, int(args.row_index))
    clean_index = int(row["clean_index"])
    source_id = int(row["source_id"])
    variant_id = int(row["variant_id"])
    class_id = int(row["class_id"])
    purified_index = int(row["purified_index"])
    sign_multiplier = -1.0 if args.sign_convention == "paper" else 1.0
    sign_note = "当前数据约定：上行为正，下行为负" if args.sign_convention == "user" else "论文约定：入站为正，出站为负"

    clean, clean_label = load_clean(Path(args.clean_path), clean_index)
    defended = load_defended(row)
    purified = load_purified(row)
    clean_c = cumul_curve(clean, samples=int(args.samples), sign_multiplier=sign_multiplier)
    defended_c = cumul_curve(defended, samples=int(args.samples), sign_multiplier=sign_multiplier)
    purified_c = cumul_curve(purified, samples=int(args.samples), sign_multiplier=sign_multiplier)

    summary = {
        "method": "CUMUL",
        "definition": "C(T)=((0,0),(a_i,c_i)), a_i=sum_j<=i |p_j|, c_i=sum_j<=i p_j; sampled by piecewise-linear interpolation.",
        "sign_convention": args.sign_convention,
        "sign_note": sign_note,
        "source_id": source_id,
        "variant_id": variant_id,
        "class_id": class_id,
        "clean_index": clean_index,
        "clean_label": clean_label,
        "defended_index": int(row["defended_index"]),
        "defended_local_index": int(row["defended_local_index"]),
        "purified_index": purified_index,
        "representation": row["representation"],
        "output_length_policy": row["output_length_policy"],
        "sample_count": int(args.samples),
        "clean": summary_item(clean_c),
        "defended": summary_item(defended_c),
        "purified": summary_item(purified_c),
    }

    blue = "#1f77b4"
    red = "#d62728"
    gray = "#666666"
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5))
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.06, right=0.98, wspace=0.24, hspace=0.38)
    fig.suptitle(
        f"CUMUL 累计表示：source_id={source_id}，variant={variant_id}，类别={class_id}\n"
        f"a_i=Σ|p_i|，c_i=Σp_i；{sign_note}；采样点 n={int(args.samples)}",
        fontsize=14,
        fontproperties=font_prop,
    )

    ax = axes[0, 0]
    ax.plot(clean_c["a"], clean_c["c"], label=f"干净流量（包数={clean_c['nonzero']}）", lw=2.0, color=blue)
    ax.plot(purified_c["a"], purified_c["c"], label=f"净化后流量（包数={purified_c['nonzero']}）", lw=1.2, color=red, alpha=0.9)
    apply_axis_style(ax, "完整 CUMUL 曲线", "累计绝对包大小 a_i（固定 cell 下等价于累计包数）", font_prop)
    ax.legend(loc="best", prop=font_prop)

    ax = axes[0, 1]
    zoom_x = min(max(float(clean_c["max_a"]) * 2.0, 800.0), float(purified_c["max_a"]))
    ax.plot(clean_c["a"], clean_c["c"], label="干净流量", lw=2.0, color=blue)
    mask = purified_c["a"] <= zoom_x
    ax.plot(purified_c["a"][mask], purified_c["c"][mask], label=f"净化后前 {int(zoom_x)} 个包", lw=1.2, color=red, alpha=0.9)
    ax.set_xlim(0, zoom_x)
    apply_axis_style(ax, "CUMUL 局部放大", "累计绝对包大小 a_i", font_prop)
    ax.legend(loc="best", prop=font_prop)

    ax = axes[1, 0]
    plot_sampled(ax, clean_c, blue, f"干净流量采样点 n={int(args.samples)}")
    plot_sampled(ax, purified_c, red, f"净化后采样点 n={int(args.samples)}")
    apply_axis_style(ax, "CUMUL 等距采样点", "等距采样位置 a_i", font_prop)
    ax.legend(loc="best", prop=font_prop)

    ax = axes[1, 1]
    ax.plot(clean_c["a"], clean_c["c"], label=f"干净流量（包数={clean_c['nonzero']}）", lw=2.0, color=blue)
    ax.plot(defended_c["a"], defended_c["c"], label=f"防御后流量（包数={defended_c['nonzero']}）", lw=1.5, color=gray, alpha=0.65)
    ax.plot(purified_c["a"], purified_c["c"], label=f"净化后流量（包数={purified_c['nonzero']}）", lw=1.0, color=red, alpha=0.75)
    apply_axis_style(ax, "干净 / 防御后 / 净化后 CUMUL 对照", "累计绝对包大小 a_i", font_prop)
    ax.legend(loc="best", prop=font_prop)

    stats_text = (
        f"干净：正向 {clean_c['pos']}，负向 {clean_c['neg']}，包数 {clean_c['nonzero']}，终点 {clean_c['final_cum']:.0f}\n"
        f"防御：正向 {defended_c['pos']}，负向 {defended_c['neg']}，包数 {defended_c['nonzero']}，终点 {defended_c['final_cum']:.0f}\n"
        f"净化：正向 {purified_c['pos']}，负向 {purified_c['neg']}，包数 {purified_c['nonzero']}，终点 {purified_c['final_cum']:.0f}"
    )
    fig.text(0.5, 0.035, stats_text, ha="center", va="bottom", fontsize=10, fontproperties=font_prop)

    suffix = "paper" if args.sign_convention == "paper" else "user"
    png_path = output_dir / f"pair_source{source_id}_variant{variant_id}_cumul_{suffix}_zh.png"
    summary_path = output_dir / f"pair_source{source_id}_variant{variant_id}_cumul_{suffix}_summary_zh.json"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"image": str(png_path), "summary": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
