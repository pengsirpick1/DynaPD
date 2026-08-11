"""
Streaming state machine 版 DynaPD-RT (严格因果) — 审计修复版 v2
修复 5 个审计点:
  1. 统一 burst 定义: tr > 0 (正方向, 对齐 fullcw 主线)
  2. 统一 renderer 参数: coordinate='absolute', strategy='priority' (对齐 fullcw)
  3. buffer/release-time audit: 只 delay 已到达(时间戳<=决策bin)未释放的包
  4. 最后 burst 动作单独统计 (tail action)
  5. 带宽用 renderer 返回的 raw_bandwidth, 不用截断后 nonzero count
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
WFLIB_ROOT = ROOT / "wflib_copy"
if WFLIB_ROOT.exists() and str(WFLIB_ROOT) not in sys.path:
    sys.path.insert(0, str(WFLIB_ROOT))
from scripts.stage_b_run_dual_actuator import _render_dummy

# ---------------- 常量 (对齐 fullcw 主线) ----------------
BINS = 1800
GAP_THRESH = 4                     # bin 间隔 > 4 视为 burst 结束
RHO = 0.25                         # running budget 比例
MAX_DELAY = 64                     # bounded delay (bins)
DELAY_WIN = 16                     # delay 窗口宽度 (bins)
USE_DELAY = True
SL = 5000

# renderer 参数: 与 fullcw_gen_rt_mp.py 主线完全一致
def _make_args(seed=0):
    from types import SimpleNamespace
    return SimpleNamespace(rf_num_slots=1800, max_trace_length=SL, max_delay=0, rounds=8,
                           delay_length=64, delay_rho=1.0, max_load_time=80.0,
                           algorithm='priority', seed=seed, renderer_strategy='priority',
                           renderer_coordinate='absolute', ratio=0.10, max_windows=8)

# ---------------- utility 表 ----------------
def _load_utility() -> dict[tuple[str, str], dict[int, float]]:
    """Load the public compact phase/direction/dose utility table."""
    table_path = ROOT / "configs" / "dynapd_rt_utility.json"
    with table_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return {
        (phase, direction): {int(dose): float(value) for dose, value in doses.items()}
        for phase, directions in raw["utility"].items()
        for direction, doses in directions.items()
    }


utility = _load_utility()

# tail action ablation 开关: True=最后 burst 也防御 (含), False=不防御 (靠 timeout 结束)
TAIL_ACTION = True

# ---------------- 流式防御 ----------------
def phase_of_bin(bin_idx):
    if bin_idx < 600:
        return 'early'
    if bin_idx < 1200:
        return 'mid'
    return 'late'

def best_dose(phase, budget_left):
    row = utility.get((phase, 'out'), {})
    best, best_ratio = 1, -1e9
    for dose in [1, 2, 4, 8]:
        if dose <= budget_left and dose in row:
            ratio = abs(row[dose]) / dose
            if ratio > best_ratio:
                best_ratio, best = ratio, dose
    return best

def _extract_all_packets(tr):
    """提取全部包 (含正负方向): 用于预算累积 (对齐 fullcw clean_total=(tr!=0))"""
    nz = np.flatnonzero(tr != 0)
    if len(nz) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
    times = np.abs(tr[nz])
    slots = np.floor(times * (1799.0 / 80.0)).astype(int)
    slots = np.clip(slots, 0, 1799)
    dirs = np.sign(tr[nz])
    order = np.argsort(slots)
    return times[order], slots[order], dirs[order], nz[order]

def defend_stream(tr, seed=0, rho=RHO, debug=False):
    """
    严格 streaming: 逐包扫描 (只用已观测), burst 结束立即防御.
    两套状态:
      obs_all:  所有已观测包 (tr!=0) → running budget (对齐 fullcw 主线 clean_total)
      outgoing: 正方向包 (tr>0)     → burst 检测/触发动作
    buffering proxy 模型: 包到达后被代理缓存, delay = 推迟释放时间.
    audit: 只 delay 时间戳 <= 决策 bin 的包 (已到达未释放), 且只后移.
    """
    rng = np.random.default_rng(seed)
    times, slots, dirs, _ = _extract_all_packets(tr)
    n_pkts = len(times)

    obs_all = 0                   # 所有已观测包 (预算累积)
    used_dummy = 0
    cur_b_start = cur_b_end = None   # 正方向 burst 状态
    cur_b_n = 0
    burst_id = 0
    injections = []
    delay_windows = []            # (decision_bin, ws, we)
    n_bursts_detected = 0
    n_tail_actions = 0
    audit = {'delay_past_packet': 0, 'delay_future_window': 0}

    def close_burst(is_tail=False):
        nonlocal used_dummy, burst_id, n_bursts_detected, n_tail_actions
        if cur_b_end is None:
            return
        n_bursts_detected += 1
        if is_tail:
            n_tail_actions += 1
        decision_bin = int(cur_b_end)                    # 决策时刻 = burst 尾 bin
        phase = phase_of_bin(decision_bin)
        # token bucket: 按 ALL 已观测包累积 (rho * obs_all), 对齐 fullcw budget=0.25*total
        token = rho * obs_all - used_dummy              # 可用额度
        if token > 0:
            seen = burst_id + 1
            # per-burst 份额: 总预算均分到已见 burst 数 (近似 batch per_b = budget/n_b)
            per_burst = max(1, int(rho * obs_all / seen))
            dose = best_dose(phase, int(token))
            dose = max(dose, min(int(per_burst * 0.7), int(token)))
            dose = min(dose, int(token))
            if dose > 0:
                dummy_bin = min(BINS - 1, cur_b_end + 1)
                injections.append((dummy_bin, dose, phase))
                used_dummy += dose
                # delay 窗口: 只覆盖已到达区域 [decision_bin-DELAY_WIN, decision_bin]
                ws = max(0, decision_bin - DELAY_WIN)
                we = decision_bin + 1
                delay_windows.append((decision_bin, ws, we))

    i = 0
    while i < n_pkts:
        t = times[i]; s = int(slots[i]); d = dirs[i]
        obs_all += 1                      # 所有包累积预算 (对齐 fullcw clean_total)
        if d > 0:                         # 只用正方向包做 burst 检测/触发
            if cur_b_start is None:
                cur_b_start = cur_b_end = s
                cur_b_n = 1
            elif s - cur_b_end <= GAP_THRESH:
                cur_b_end = s
                cur_b_n += 1
            else:
                close_burst(is_tail=False)
                burst_id += 1
                cur_b_start = cur_b_end = s
                cur_b_n = 1
        i += 1
    # 最后 burst: 真实在线靠 timeout 判定
    if TAIL_ACTION:
        close_burst(is_tail=True)

    # ---- 渲染 (与 fullcw 主线同 renderer) ----
    counts = np.zeros((2, BINS), dtype=np.int32)
    for (dummy_bin, dose, phase) in injections:
        for b in range(dummy_bin, min(BINS, dummy_bin + dose)):
            counts[0, b] += 1
    tr_d, _tam, stats = _render_dummy(base_trace=tr, counts=counts, args=_make_args(seed))

    # ---- causal delay + buffer audit ----
    if delay_windows and USE_DELAY:
        nz2 = np.flatnonzero(tr_d != 0)
        if len(nz2):
            t2 = np.abs(tr_d[nz2]); s2 = np.sign(tr_d[nz2])
            sl2 = np.floor(t2 * (1799.0/80.0)).astype(int); sl2 = np.clip(sl2, 0, 1799)
            for (decision_bin, ws_, we_) in delay_windows:
                in_win = (sl2 >= ws_) & (sl2 < we_)
                if in_win.any():
                    # audit: 窗口内包必须已到达 (时间戳 <= decision_bin), 且只后移
                    past = sl2[in_win] <= decision_bin
                    audit['delay_past_packet'] += int(past.sum())
                    audit['delay_future_window'] += int((~past).sum())
                    db = rng.integers(1, MAX_DELAY + 1, size=in_win.sum())
                    t2[in_win] += db * (80.0/1800.0)
            tr_d[nz2] = s2 * t2
            order2 = np.argsort(np.abs(tr_d)); ts = tr_d[order2]
            tr_d = np.pad(ts, (0, max(0, SL-len(ts))), mode='constant')[:SL]

    if debug:
        bw = float(stats.get('raw_bandwidth', 0.0))
        return tr_d, {'n_bursts': n_bursts_detected, 'n_inj': len(injections),
                      'inj_total': int(sum(d for _, d, _ in injections)),
                      'n_delay': len(delay_windows), 'n_tail': n_tail_actions,
                      'audit': audit, 'raw_bw': bw}
    return tr_d

# ---------------- 评估 ----------------
def main():
    from WFlib import models as wm
    from dynapd.evaluation.attack_models import build_rf_tam_input
    from dynapd.stage_a.faithfulness import predict_probabilities
    from dynapd.stage_a.modeling import load_stage_a_attacker

    d = np.load('wflib_copy/datasets/CW/test.npz')
    X, y = d['X'][:512].astype(np.float32), d['y'][:512]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rf = load_stage_a_attacker('models/attacks/fixed_rf_checkpoint.pt', attacker='rf', device=device)
    df_model = wm.DF(95).to(device); df_model.load_state_dict(torch.load('wflib_copy/checkpoints/CW/DF/dynapd_clean_seed0.pth', map_location=device, weights_only=True)); df_model.eval()
    tf_model = wm.TF(95).to(device); tf_model.load_state_dict(torch.load('wflib_copy/checkpoints/CW/TF/dynapd_clean_seed0.pth', map_location=device, weights_only=True)); tf_model.eval()
    awf_model = wm.AWF(95).to(device); awf_model.load_state_dict(torch.load('wflib_copy/checkpoints/CW/AWF/dynapd_clean_seed0.pth', map_location=device, weights_only=True)); awf_model.eval()
    vc_model = wm.VarCNN(95).to(device); vc_model.load_state_dict(torch.load('wflib_copy/checkpoints/CW/VarCNN/dynapd_clean_seed0.pth', map_location=device, weights_only=True)); vc_model.eval()

    def eval_batch(Xb):
        res = {}
        tam = build_rf_tam_input(Xb, max_len=5000, max_load_time=80.0, num_slots=1800)
        p = predict_probabilities(rf, tam, device=device, batch_size=256)
        res['RF'] = float(np.mean(np.argmax(p, 1) == y[:len(Xb)]))
        from scripts.stage_b_run_ensemble_oracle_e2b_completion import _predict_wflib
        res['DF'] = float(np.mean(np.argmax(_predict_wflib(df_model, list(Xb), feature='DIR', device=device, batch_size=256, seq_len=5000), 1) == y[:len(Xb)]))
        res['TF'] = float(np.mean(np.argmax(_predict_wflib(tf_model, list(Xb), feature='DIR', device=device, batch_size=256, seq_len=5000), 1) == y[:len(Xb)]))
        res['AWF'] = float(np.mean(np.argmax(_predict_wflib(awf_model, list(Xb), feature='DIR', device=device, batch_size=256, seq_len=3000), 1) == y[:len(Xb)]))
        def dt2_feature(traces, sl=3000):
            x = np.asarray(traces)[:, :sl]
            x_dir = np.sign(x)
            x_time = np.abs(x)
            x_time = np.diff(x_time, axis=1)
            x_time[x_time < 0] = 0.0
            x_time = np.pad(x_time, ((0, 0), (0, 1)), mode='constant')[:, :sl]
            return np.stack([x_dir, x_time], axis=1).astype(np.float32)
        xb = torch.tensor(dt2_feature(Xb), dtype=torch.float32).to(device)
        od, _ = vc_model(xb)
        res['VarCNN'] = float(np.mean(torch.argmax(od, 1).cpu().numpy() == y[:len(Xb)]))
        return res

    # batch 对照: 与 streaming 同 renderer/同 burst 定义 (tr>0), 完整 trace 信息
    def defend_batch(tr, seed):
        """batch 对照: 预算按全部包 (tr!=0), 对齐 fullcw 主线 clean_total"""
        clean_total = float((tr != 0).sum())       # ← 修复: 全部包
        budget = int(clean_total * 0.25)
        # burst 提取仍用正方向包 (tr>0)
        nz = np.flatnonzero(tr > 0)
        times2 = np.abs(tr[nz])
        slots2 = np.floor(times2 * (1799.0 / 80.0)).astype(int)
        slots2 = np.clip(slots2, 0, 1799)
        order2 = np.argsort(slots2); slots2 = slots2[order2]
        bursts = []
        cs, ce, cn = slots2[0], slots2[0], 1
        for s in slots2[1:]:
            if s - ce <= GAP_THRESH:
                ce = s; cn += 1
            else:
                bursts.append((cs, ce, cn)); cs = ce = s; cn = 1
        bursts.append((cs, ce, cn))
        n_b = len(bursts)
        rng = np.random.default_rng(seed)
        counts = np.zeros((2, BINS), dtype=np.int32)
        used = 0
        delay_windows = []
        per_b = max(1, int(budget / max(n_b, 1)))
        for bi, (s, e, bn) in enumerate(bursts):
            if used >= budget:
                break
            phase = 'early' if bi < n_b/3 else ('mid' if bi < 2*n_b/3 else 'late')
            dose = best_dose(phase, per_b)
            dose = min(max(dose, int(per_b * 0.7)), budget - used)
            for bb in range(e, min(BINS, e + dose)):
                if used >= budget:
                    break
                counts[0, bb] += 1; used += 1
            delay_windows.append((int(e), max(0, int(e)-DELAY_WIN), min(BINS, int(e)+16)))
        tr_d, _tam, stats = _render_dummy(base_trace=tr, counts=counts, args=_make_args(seed))
        if delay_windows and USE_DELAY:
            nz2 = np.flatnonzero(tr_d != 0)
            if len(nz2):
                t2 = np.abs(tr_d[nz2]); s2 = np.sign(tr_d[nz2])
                sl2 = np.floor(t2 * (1799.0/80.0)).astype(int); sl2 = np.clip(sl2, 0, 1799)
                for (db_, ws_, we_) in delay_windows:
                    in_win = (sl2 >= ws_) & (sl2 < we_)
                    if in_win.any():
                        dd = rng.integers(1, MAX_DELAY + 1, size=in_win.sum())
                        t2[in_win] += dd * (80.0/1800.0)
                tr_d[nz2] = s2 * t2
                order2 = np.argsort(np.abs(tr_d)); ts = tr_d[order2]
                tr_d = np.pad(ts, (0, max(0, SL-len(ts))), mode='constant')[:SL]
        return tr_d, float(stats.get('raw_bandwidth', 0.0))

    # ---- streaming (delay on) + tail ablation ----
    global USE_DELAY, TAIL_ACTION
    for tail_on in [True, False]:
        TAIL_ACTION = tail_on
        USE_DELAY = True
        t0 = time.time()
        dbg_list = []; def_stream = []; bws = []
        for i in range(512):
            tr_d, dbg = defend_stream(X[i], i, debug=True)
            dbg_list.append(dbg); bws.append(dbg['raw_bw'])
            def_stream.append(np.pad(tr_d, (0, SL), mode='constant')[:SL])
        def_stream = np.stack(def_stream)
        t_ms = (time.time() - t0) / 512 * 1000
        r = eval_batch(def_stream)
        nb = np.mean([d['n_bursts'] for d in dbg_list]); inj = np.mean([d['inj_total'] for d in dbg_list])
        tail = np.mean([d['n_tail'] for d in dbg_list]); dl = np.mean([d['n_delay'] for d in dbg_list])
        past = sum(d['audit']['delay_past_packet'] for d in dbg_list)
        fut = sum(d['audit']['delay_future_window'] for d in dbg_list)
        bw = float(np.mean(bws))
        tag = f'streaming_tail{int(tail_on)}'
        print(f'[STR] {tag}: WC={max(r.values()):.4f} | ' + ' '.join(f'{k}={v:.4f}' for k, v in r.items()) +
              f' | BW={bw:.4f} | gen={t_ms:.1f}ms | bursts={nb:.1f} inj={inj:.0f} tail={tail:.1f} delay_win={dl:.1f} | audit_past={past} future={fut}', flush=True)

    # ---- batch 对照 ----
    t0 = time.time()
    def_batch = []; bws = []
    for i in range(512):
        tr_d, bw = defend_batch(X[i], i)
        bws.append(bw)
        def_batch.append(np.pad(tr_d, (0, SL), mode='constant')[:SL])
    def_batch = np.stack(def_batch)
    t_ms = (time.time() - t0) / 512 * 1000
    r = eval_batch(def_batch)
    bw = float(np.mean(bws))
    print(f'[STR] batch_fullinfo: WC={max(r.values()):.4f} | ' + ' '.join(f'{k}={v:.4f}' for k, v in r.items()) +
          f' | BW(renderer)={bw:.4f} | gen={t_ms:.1f}ms', flush=True)
    print('[STR] DONE', flush=True)

if __name__ == '__main__':
    main()
