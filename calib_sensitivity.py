"""
N_calib 敏感性实验: 验证 small-calibration claim
- 生成 N_calib = 32/64/128/256/512/1024 的 utility 表 (从 test.npz 前 N_calib 条)
- 每个表在固定测试集 test[1024:10564] 上跑 streaming tail0, 子采样 N_EVAL 条
- 看 WC/BW 是否 128/256 接近饱和
"""
import sys, numpy as np, torch, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
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
import streaming_state_machine as sm

device = resolve_device('cuda')
set_seed(0)
NUM_CLASSES = 95

def make_args():
    return SimpleNamespace(rf_num_slots=1800, max_trace_length=5000, max_delay=0, rounds=8,
                           delay_length=64, delay_rho=1.0, max_load_time=80.0,
                           algorithm='priority', seed=0, renderer_strategy='priority',
                           renderer_coordinate='absolute', ratio=0.10, max_windows=8)

# ---- 加载模型 ----
rf = load_stage_a_attacker('models/attacks/fixed_rf_checkpoint.pt', attacker='rf', device=device)
df_model = wm.DF(NUM_CLASSES).to(device)
df_model.load_state_dict(torch.load('wflib_copy/checkpoints/CW/DF/dynapd_clean_seed0.pth', map_location=device, weights_only=True)); df_model.eval()
awf_model = wm.AWF(NUM_CLASSES).to(device)
awf_model.load_state_dict(torch.load('wflib_copy/checkpoints/CW/AWF/dynapd_clean_seed0.pth', map_location=device, weights_only=True)); awf_model.eval()
vc_model = wm.VarCNN(NUM_CLASSES).to(device)
vc_model.load_state_dict(torch.load('wflib_copy/checkpoints/CW/VarCNN/dynapd_clean_seed0.pth', map_location=device, weights_only=True)); vc_model.eval()

def margin_of(prob, tl):
    return float(prob[tl] - np.delete(prob, tl).max())

def wflib_prob(model, trace, sl=5000):
    xb = np.sign(trace[:sl])[None, None, :]
    with torch.no_grad():
        od, _ = model(torch.tensor(xb, dtype=torch.float32).to(device))
    return torch.softmax(od, dim=1).cpu().numpy()[0]

def varcnn_prob(trace, sl=5000):
    x = np.asarray(trace)[:sl]
    x_dir = np.sign(x)
    x_time = np.abs(x)
    x_time = np.diff(x_time, axis=1)
    x_time[x_time < 0] = 0.0
    x_time = np.pad(x_time, ((0, 0), (0, 1)), mode='constant')[:, :sl]
    xb = torch.tensor(np.stack([x_dir, x_time], axis=1)[None], dtype=torch.float32).to(device)
    with torch.no_grad():
        od, _ = vc_model(xb)
    return torch.softmax(od, dim=1).cpu().numpy()[0]

def build_utility(n_calib):
    d = np.load('wflib_copy/datasets/CW/test.npz')
    X, y = d['X'][:n_calib].astype(np.float32), d['y'][:n_calib]
    utility = {}
    for i in range(n_calib):
        tr = X[i]; tl = int(y[i])
        nz = np.flatnonzero(tr > 0)
        if len(nz) == 0:
            continue
        times = tr[nz]
        slots = np.floor(times * (1799.0/80.0)).astype(int)
        bursts = []
        cs, ce, cn = slots[0], slots[0], 1
        for s in slots[1:]:
            if s - ce <= 4:
                ce = s; cn += 1
            else:
                bursts.append((cs, ce, cn)); cs = ce = s; cn = 1
        bursts.append((cs, ce, cn))
        n_b = len(bursts)
        tam = build_rf_tam_input(tr[None], max_len=5000, max_load_time=80.0, num_slots=1800)
        prob = predict_probabilities(rf, tam, device=device, batch_size=1)[0]
        m0_rf = margin_of(prob, tl)
        m0_df = margin_of(wflib_prob(df_model, tr), tl)
        m0_awf = margin_of(wflib_prob(awf_model, tr, 3000), tl)
        for bi, (s, e, bn) in enumerate(bursts):
            phase = 'early' if bi < n_b/3 else ('mid' if bi < 2*n_b/3 else 'late')
            for dose in [1, 2, 4, 8]:
                counts = np.zeros((2, 1800), dtype=np.int32)
                for bb in range(e, min(1800, e + dose)):
                    counts[0, bb] += 1
                tr_d, _tam_d, _ = _render_dummy(base_trace=tr, counts=counts, args=make_args())
                tr_pad = np.pad(tr_d, (0, max(0, 5000-len(tr_d))), mode='constant')[:5000]
                g_rf = margin_of(predict_probabilities(rf, build_rf_tam_input(tr_pad[None], max_len=5000, max_load_time=80.0, num_slots=1800), device=device, batch_size=1)[0], tl) - m0_rf
                g_df = margin_of(wflib_prob(df_model, tr_pad), tl) - m0_df
                g_awf = margin_of(wflib_prob(awf_model, tr_pad, 3000), tl) - m0_awf
                key = (phase, 'out')
                if key not in utility:
                    utility[key] = {dd: {'gain': []} for dd in [1, 2, 4, 8]}
                utility[key][dose]['gain'].append(0.8*g_rf + 0.1*g_df + 0.1*g_awf)
    table = {}
    for key in sorted(utility.keys()):
        row = {}
        for dose, v in utility[key].items():
            if v['gain']:
                row[dose] = float(np.mean(v['gain']))
        table[key] = row
    return table

# ---- 评估 (固定测试集) ----
def eval_tail0(X_test, y_test, utility):
    sm.utility = utility  # 替换全局表
    N = min(len(X_test), 512)
    def_stream = []; bws = []
    for i in range(N):
        tr_d, dbg = sm.defend_stream(X_test[i], i, debug=True)
        bws.append(dbg['raw_bw'])
        def_stream.append(np.pad(tr_d, (0, 5000), mode='constant')[:5000])
    def_stream = np.stack(def_stream)
    yb = y_test[:N]
    # 5 模型评估
    res = {}
    tam = build_rf_tam_input(def_stream, max_len=5000, max_load_time=80.0, num_slots=1800)
    p = predict_probabilities(rf, tam, device=device, batch_size=256)
    res['RF'] = float(np.mean(np.argmax(p, 1) == yb))
    from scripts.stage_b_run_ensemble_oracle_e2b_completion import _predict_wflib
    res['DF'] = float(np.mean(np.argmax(_predict_wflib(df_model, list(def_stream), feature='DIR', device=device, batch_size=256, seq_len=5000), 1) == yb))
    res['AWF'] = float(np.mean(np.argmax(_predict_wflib(awf_model, list(def_stream), feature='DIR', device=device, batch_size=256, seq_len=3000), 1) == yb))
    xb = torch.tensor(np.stack([np.sign(def_stream[:, :3000]),
                                np.pad(np.diff(np.abs(def_stream[:, :3000]), axis=1)[:, :2999], ((0,0),(0,1)), mode='constant')], axis=1), dtype=torch.float32).to(device)
    od, _ = vc_model(xb)
    res['VarCNN'] = float(np.mean(torch.argmax(od, 1).cpu().numpy() == yb))
    bw = float(np.mean(bws))
    return res, bw

# ---- 主流程 ----
print('[CAL] N_calib 敏感性实验启动...', flush=True)
d = np.load('wflib_copy/datasets/CW/test.npz')
X_test = d['X'][1024:10564].astype(np.float32)   # 固定测试集, 避开所有校准区间
y_test = d['y'][1024:10564]
print(f'[CAL] 固定测试集: test[1024:10564] = {len(X_test)} 条', flush=True)

for n_calib in [32, 64, 128, 256, 512, 1024]:
    t0 = time.time()
    print(f'[CAL] 生成 utility 表 N_calib={n_calib}...', flush=True)
    ut = build_utility(n_calib)
    t_build = time.time() - t0
    res, bw = eval_tail0(X_test, y_test, ut)
    print(f'[CAL] N={n_calib}: WC={max(res.values()):.4f} | ' + ' '.join(f'{k}={v:.4f}' for k, v in res.items()) +
          f' | BW={bw:.4f} | build={t_build:.0f}s', flush=True)
print('[CAL] DONE', flush=True)
