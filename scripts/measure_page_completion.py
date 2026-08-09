# -*- coding: utf-8 -*-
"""计算 time overhead 的真实口径.
1. clean trace 实际传输时长 (最后包时间戳)
2. defended trace 完成时间 (delay 后最后包时间)
3. delta = 时间开销秒数 + 相对 clean 时长百分比
"""
import sys, numpy as np, glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 1. clean trace 时长分布
print('[T] ===== clean trace 传输时长 =====', flush=True)
d = np.load('wflib_copy/datasets/CW/test.npz')
X = d['X'][:512].astype(np.float32)
durs = []
for i in range(512):
    nz = np.flatnonzero(X[i] != 0)
    if len(nz):
        durs.append(float(np.abs(X[i][nz]).max()))
durs = np.asarray(durs)
print(f'[T] clean 时长(秒): mean={durs.mean():.1f} p50={np.median(durs):.1f} p95={np.percentile(durs,95):.1f} max={durs.max():.1f}', flush=True)

# 2. defended 完成时间 (用已生成的全量 chunk)
print('[T] ===== defended 完成时间 =====', flush=True)
files = sorted(glob.glob('results/fullcw_rt_defended_chunk*.npz'))
dd = np.load(files[0])  # 取第一块 20000 条
def_X = dd['X'][:1000].astype(np.float32)
def_durs = []
for i in range(1000):
    nz = np.flatnonzero(def_X[i] != 0)
    if len(nz):
        def_durs.append(float(np.abs(def_X[i][nz]).max()))
def_durs = np.asarray(def_durs)
print(f'[T] defended 完成时间(秒): mean={def_durs.mean():.1f} p95={np.percentile(def_durs,95):.1f} max={def_durs.max():.1f}', flush=True)

# 3. 对应 clean 完成时间 (同索引)
clean_sub = []
for i in range(512):
    nz = np.flatnonzero(X[i] != 0)
    clean_sub.append(float(np.abs(X[i][nz]).max()) if len(nz) else 0)
clean_sub = np.asarray(clean_sub)
delta = def_durs - clean_sub
print(f'[T] 时间开销 delta(秒): mean={delta.mean():.1f} p95={np.percentile(delta,95):.1f} max={delta.max():.1f}', flush=True)
print(f'[T] delta/clean 百分比: mean={100*delta.mean()/clean_sub.mean():.1f}% p95={100*np.percentile(delta,95)/np.percentile(clean_sub,95):.1f}%', flush=True)

# 4. bins 换算
print(f'[T] 换算: 1 bin={80.0/1800*1000:.0f}ms; 61 bins={61*80.0/1800:.1f}s', flush=True)
print('[T] DONE', flush=True)
