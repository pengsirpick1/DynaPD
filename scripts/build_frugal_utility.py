# -*- coding: utf-8 -*-
"""FRUGAL-lite MI utility 表生成.
离线: 对训练集样本, 在每个 burst 尾注入 dummy (dose 1..8), 三模型评估 gain.
聚合: (burst_phase: 前/中/后, direction: out/in) -> utility (平均 gain, 平均 cost)
在线: 当前 burst -> 查表 -> 选 dose (utility/cost 权衡)
"""
import sys, numpy as np, torch, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if (ROOT / 'wflib_copy').exists():
    sys.path.insert(0, str(ROOT / 'wflib_copy'))
from dynapd.utils import resolve_device, set_seed
from dynapd.evaluation.attack_models import build_rf_tam_input
from dynapd.stage_a.modeling import load_stage_a_attacker
from dynapd.stage_a.faithfulness import predict_probabilities
from scripts.stage_b_run_dual_actuator import _render_dummy
from WFlib import models as wm
from types import SimpleNamespace

device = resolve_device('cuda')
set_seed(0)
NUM_CLASSES = 95
N_SAMPLES = 256   # 训练表用 256 条
MAX_BURST_PER_TRACE = 40

d = np.load('wflib_copy/datasets/CW/test.npz')
X, y = d['X'][:N_SAMPLES].astype(np.float32), d['y'][:N_SAMPLES]
rf = load_stage_a_attacker('models/attacks/fixed_rf_checkpoint.pt', attacker='rf',
                           num_classes=NUM_CLASSES, device=device, max_trace_length=5000, rf_num_slots=1800)

df_model = wm.DF(NUM_CLASSES).to(device)
df_model.load_state_dict(torch.load('wflib_copy/checkpoints/CW/DF/dynapd_clean_seed0.pth', map_location=device, weights_only=True)); df_model.eval()
awf_model = wm.AWF(NUM_CLASSES).to(device)
awf_model.load_state_dict(torch.load('wflib_copy/checkpoints/CW/AWF/dynapd_clean_seed0.pth', map_location=device, weights_only=True)); awf_model.eval()
vc_model = wm.VarCNN(NUM_CLASSES).to(device)
vc_model.load_state_dict(torch.load('wflib_copy/checkpoints/CW/VarCNN/dynapd_clean_seed0.pth', map_location=device, weights_only=True)); vc_model.eval()

args = SimpleNamespace(rf_num_slots=1800, max_trace_length=5000, max_delay=0, rounds=8,
                       delay_length=64, delay_rho=1.0, max_load_time=80.0,
                       algorithm='priority', seed=0, renderer_strategy='priority',
                       renderer_coordinate='absolute', ratio=0.10, max_windows=8)

def dt2_feature(trace, sl=5000):
    x = np.asarray(trace, dtype=np.float32)
    x_dir = np.sign(x); x_time = np.abs(x)
    x_time = np.diff(x_time, axis=1); x_time[x_time < 0] = 0
    x_time = np.concatenate([np.zeros((len(x), 1), dtype=np.float32), x_time], axis=1)
    out = np.zeros((len(x), sl), dtype=np.float32); ot = np.zeros((len(x), sl), dtype=np.float32)
    n = min(x.shape[1], sl)
    out[:, :n] = x_dir[:, :n]; ot[:, :n] = x_time[:, :n]
    return np.stack([out, ot], axis=1)

def margin_of(prob, tl):
    return float(prob[tl] - np.delete(prob, tl).max())

def wflib_prob(model, trace, sl=5000):
    xb = np.sign(trace[:sl])[None, None, :]
    with torch.no_grad():
        od, _ = model(torch.tensor(xb, dtype=torch.float32).to(device))
    return torch.softmax(od, dim=1).cpu().numpy()[0]

def varcnn_prob(trace, sl=5000):
    xb = torch.tensor(dt2_feature(trace[None, :sl]), dtype=torch.float32).to(device)
    with torch.no_grad():
        od, _ = vc_model(xb)
    return torch.softmax(od, dim=1).cpu().numpy()[0]

def extract_bursts(trace):
    nz = np.flatnonzero(trace > 0)
    if len(nz) == 0:
        return []
    times = trace[nz]
    slots = np.floor(times * (1799.0 / 80.0)).astype(int)
    bursts = []
    cur_start, cur_end, cur_n = slots[0], slots[0], 1
    for s in slots[1:]:
        if s - cur_end <= 4:
            cur_end = s; cur_n += 1
        else:
            bursts.append((cur_start, cur_end, cur_n))
            cur_start = cur_end = s; cur_n = 1
    bursts.append((cur_start, cur_end, cur_n))
    return bursts

# 聚合表: key=(phase, direction), value={dose: (mean_gain, n, mean_cost)}
utility = {}
t0 = time.time()
for i in range(N_SAMPLES):
    tr = X[i]; tl = int(y[i])
    bursts = extract_bursts(tr)[:MAX_BURST_PER_TRACE]
    n_b = len(bursts)
    if n_b == 0:
        continue
    # 原始 margin
    tam = build_rf_tam_input(tr[None], max_len=5000, max_load_time=80.0, num_slots=1800)[0]
    prob = predict_probabilities(rf, tam[None], device=device, batch_size=1)[0]
    m0_rf = margin_of(prob, tl)
    p_df0 = wflib_prob(df_model, tr); m0_df = margin_of(p_df0, tl)
    p_awf0 = wflib_prob(awf_model, tr, 3000); m0_awf = margin_of(p_awf0, tl)
    p_vc0 = varcnn_prob(tr); m0_vc = margin_of(p_vc0, tl)
    for bi, (s, e, bn) in enumerate(bursts):
        # 阶段
        phase = 'early' if bi < n_b / 3 else ('mid' if bi < 2 * n_b / 3 else 'late')
        for dose in [1, 2, 4, 8]:
            # 出站 burst 尾注入 (out direction)
            counts = np.zeros((2, 1800), dtype=np.int32)
            for bb in range(e, min(1800, e + dose)):
                counts[0, bb] += 1
            tr_d, _tam_d, _ = _render_dummy(base_trace=tr, counts=counts, args=args)
            tr_pad = np.pad(tr_d, (0, max(0, 5000-len(tr_d))), mode='constant')[:5000]
            p_rf = predict_probabilities(rf, build_rf_tam_input(tr_pad[None], max_len=5000, max_load_time=80.0, num_slots=1800), device=device, batch_size=1)[0]
            g_rf = margin_of(p_rf, tl) - m0_rf
            p_df = wflib_prob(df_model, tr_pad); g_df = margin_of(p_df, tl) - m0_df
            p_awf = wflib_prob(awf_model, tr_pad, 3000); g_awf = margin_of(p_awf, tl) - m0_awf
            p_vc = varcnn_prob(tr_pad); g_vc = margin_of(p_vc, tl) - m0_vc
            key = (phase, 'out')
            if key not in utility:
                utility[key] = {dd: {'gain': [], 'cost': []} for dd in [1, 2, 4, 8]}
            utility[key][dose]['gain'].append(0.8*g_rf + 0.1*g_df + 0.1*g_awf)
            utility[key][dose]['cost'].append(dose)
    if (i + 1) % 64 == 0:
        print(f'[UT] {i+1}/{N_SAMPLES} ({time.time()-t0:.0f}s)', flush=True)

# 输出表
print('[UT] FRUGAL-lite utility 表:', flush=True)
table = {}
for key in sorted(utility.keys()):
    row = {}
    for dose, v in utility[key].items():
        if v['gain']:
            row[dose] = float(np.mean(v['gain']))
    table[key] = row
    print(f'  [UT] {key}: ' + ' '.join(f'd{d}={row.get(d, 0):.4f}' for d in [1, 2, 4, 8]), flush=True)

np.save('results/frugal_lite_utility.npy', table, allow_pickle=True)
print(f'[UT] 保存 frugal_lite_utility.npy ({time.time()-t0:.0f}s)', flush=True)
print('[UT] DONE', flush=True)
