# -*- coding: utf-8 -*-
"""绘制单条 clean/defended/purified 流量的累计方向曲线。"""

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
    parser = argparse.ArgumentParser(description="绘制累计方向曲线。")
    parser.add_argument("--run-dir", required=True, help="purifier run 目录。")
    parser.add_argument("--clean-path", default="D:/learning/TOR/datasets/CW/CW.npz")
    parser.add_argument("--row-index", type=int, default=0, help="purified manifest 中的行号。")
    parser.add_argument("--output-dir", default="", help="默认写入 <run-dir>/visualizations/pair_curves。")
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


def curve(values: np.ndarray) -> dict[str, np.ndarray | int | float]:
    nonzero = np.asarray(values, dtype=np.float32).reshape(-1)
    nonzero = nonzero[nonzero != 0]
    signs = np.sign(nonzero).astype(np.float32)
    event_x = np.arange(1, len(nonzero) + 1, dtype=np.int64)
    event_y = np.cumsum(signs)
    times = np.abs(nonzero).astype(np.float32)
    order = np.argsort(times, kind="stable")
    return {
        "event_x": event_x,
        "event_y": event_y,
        "time_x": times[order],
        "time_y": np.cumsum(signs[order]),
        "pos": int(np.sum(nonzero > 0)),
        "neg": int(np.sum(nonzero < 0)),
        "nonzero": int(len(nonzero)),
        "final_cum": float(event_y[-1]) if len(event_y) else 0.0,
        "max_time": float(times.max()) if times.size else 0.0,
    }


def apply_axis_style(ax, title: str, xlabel: str, font_prop: font_manager.FontProperties) -> None:
    ax.set_title(title, fontproperties=font_prop, fontsize=12)
    ax.set_xlabel(xlabel, fontproperties=font_prop)
    ax.set_ylabel("累计方向值", fontproperties=font_prop)
    ax.axhline(0, color="black", lw=0.8, alpha=0.35)
    ax.grid(True, alpha=0.25)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)


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

    clean, clean_label = load_clean(Path(args.clean_path), clean_index)
    defended = load_defended(row)
    purified = load_purified(row)
    clean_c = curve(clean)
    defended_c = curve(defended)
    purified_c = curve(purified)

    summary = {
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
        "clean": {key: value for key, value in clean_c.items() if not key.endswith("_x") and not key.endswith("_y")},
        "defended": {key: value for key, value in defended_c.items() if not key.endswith("_x") and not key.endswith("_y")},
        "purified": {key: value for key, value in purified_c.items() if not key.endswith("_x") and not key.endswith("_y")},
    }

    blue = "#1f77b4"
    red = "#d62728"
    gray = "#666666"
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5))
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.06, right=0.98, wspace=0.24, hspace=0.38)
    fig.suptitle(
        f"单条流量累计方向曲线：source_id={source_id}，variant={variant_id}，类别={class_id}\n"
        f"累计规则：上行包 +1，下行包 -1；clean_index={clean_index}，purified_index={purified_index}",
        fontsize=14,
        fontproperties=font_prop,
    )

    ax = axes[0, 0]
    ax.plot(clean_c["event_x"], clean_c["event_y"], label=f"干净流量（非零包={clean_c['nonzero']}）", lw=2.0, color=blue)
    ax.plot(purified_c["event_x"], purified_c["event_y"], label=f"净化后流量（非零包={purified_c['nonzero']}）", lw=1.2, color=red, alpha=0.9)
    apply_axis_style(ax, "按包序号绘制的完整累计曲线", "包序号 / 事件序号", font_prop)
    ax.legend(loc="best", prop=font_prop)

    ax = axes[0, 1]
    zoom_n = int(min(max(clean_c["nonzero"] * 2, 800), purified_c["nonzero"]))
    ax.plot(clean_c["event_x"], clean_c["event_y"], label="干净流量", lw=2.0, color=blue)
    mask = clean_c["event_x"] <= zoom_n
    ax.plot(clean_c["event_x"][mask], clean_c["event_y"][mask], lw=2.0, color=blue)
    mask = purified_c["event_x"] <= zoom_n
    ax.plot(purified_c["event_x"][mask], purified_c["event_y"][mask], label=f"净化后前 {zoom_n} 个包", lw=1.2, color=red, alpha=0.9)
    ax.set_xlim(0, zoom_n)
    apply_axis_style(ax, "按包序号局部放大", "包序号 / 事件序号", font_prop)
    ax.legend(loc="best", prop=font_prop)

    ax = axes[1, 0]
    ax.plot(clean_c["time_x"], clean_c["time_y"], label="干净流量", lw=2.0, color=blue)
    ax.plot(purified_c["time_x"], purified_c["time_y"], label="净化后流量", lw=1.2, color=red, alpha=0.9)
    apply_axis_style(ax, "按时间戳绝对值排序后的累计曲线", "时间戳绝对值", font_prop)
    ax.legend(loc="best", prop=font_prop)

    ax = axes[1, 1]
    ax.plot(clean_c["event_x"], clean_c["event_y"], label=f"干净流量（非零包={clean_c['nonzero']}）", lw=2.0, color=blue)
    ax.plot(defended_c["event_x"], defended_c["event_y"], label=f"防御后流量（非零包={defended_c['nonzero']}）", lw=1.5, color=gray, alpha=0.65)
    ax.plot(purified_c["event_x"], purified_c["event_y"], label=f"净化后流量（非零包={purified_c['nonzero']}）", lw=1.0, color=red, alpha=0.75)
    apply_axis_style(ax, "干净 / 防御后 / 净化后 对照", "包序号 / 事件序号", font_prop)
    ax.legend(loc="best", prop=font_prop)

    stats_text = (
        f"干净：上行 {clean_c['pos']}，下行 {clean_c['neg']}，非零包 {clean_c['nonzero']}，终点 {clean_c['final_cum']:.0f}\n"
        f"防御：上行 {defended_c['pos']}，下行 {defended_c['neg']}，非零包 {defended_c['nonzero']}，终点 {defended_c['final_cum']:.0f}\n"
        f"净化：上行 {purified_c['pos']}，下行 {purified_c['neg']}，非零包 {purified_c['nonzero']}，终点 {purified_c['final_cum']:.0f}"
    )
    fig.text(0.5, 0.035, stats_text, ha="center", va="bottom", fontsize=10, fontproperties=font_prop)

    png_path = output_dir / f"pair_source{source_id}_variant{variant_id}_cumulative_direction_zh.png"
    summary_path = output_dir / f"pair_source{source_id}_variant{variant_id}_summary_zh.json"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"image": str(png_path), "summary": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
