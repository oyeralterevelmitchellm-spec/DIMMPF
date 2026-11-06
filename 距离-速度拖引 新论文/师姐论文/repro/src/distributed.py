# -*- coding: utf-8 -*-
"""
第 3 章架构（分布式 IMM + 近似 Chernoff 融合）与单平台对照的**端到端**实现。

论文架构（式3-25 ~ 3-49）
------------------------
* 每个无人机节点 M_l **独立**运行本地 IMM-EKF（不是把各平台量测堆叠进单个滤波器）；
* 本地 IMM 后验为高斯混合（每个运动模型一个分量）：
      p^{M_l}(x) = Σ_r μ_r^{M_l} N(x; X_r^{M_l}, P_r^{M_l})            (3-25)
* 用近似 Chernoff 融合把各节点 GMM 融合（顺序两两融合）：
      p_f(x) ∝ p_1(x)^w · p_2(x)^{1−w}                                 (3-27)
  一阶展开后仍保持 GMM 形式 —— 实现见 `src.chernoff.acf_fuse_log`
  （式3-34~3-40 的对数域等价式，9 维状态下避免 exp 溢出）；
* 融合权重 w 由重要性采样 + 黄金分割优化（式3-45~3-49）。

**为什么需要本模块**
现有 `src/pipeline.run_tracking` 的 `proposed` 模式是**集中式**的：
它把所有平台的量测**堆叠**成一个块对角量测送入**单个** IMM-EKF。
因此第 3 章的「分布式本地 IMM + GMM 层面 ACF 融合」架构
此前**从未端到端运行过**（见 `audit/out/method_coverage.md` §3：
ACF「未进入任何端到端实验」）。

模式
----
`dist_acf_is`   每平台本地 IMM(含一致性+PDA) → GMM → ACF 顺序融合，w 分段 IS 优化
`dist_acf_w05`  同上，但 w 固定 0.5（隔离权重优化的作用）
`dist_ci`       同上本地 IMM，融合用**协方差交叉**（传统单高斯融合）
`dist_product`  同上，融合用**矩匹配 + 乘积**（朴素贝叶斯）
`single_true`   单平台对照：仅 M1，用真值候选（候选 0）
`single_jam`    单平台对照：仅 M1，锁到第一条干扰（候选 1）
`single_pda`    单平台对照：仅 M1，全候选 + 传统 PDA（无一致性）
`single_cons`   单平台对照：仅 M1，全候选 + 一致性权重 + PDA

数值约定：候选 0 = 真实目标，候选 1.. = 干扰（见 `src/vgpo.build_candidates`）。
"""
import numpy as np

from . import config as C
from .measurement import (measurement_noise_cov, h_measure, H_jacobian)
from .imm import IMMFilter
from .pda import pda_fuse, pda_bs
from .consistency import consistency_scores, consistency_weight
from .chernoff import (acf_fuse_log, optimize_weight, gmm_mean_cov,
                       sequential_acf_fuse_log)

DIST_MODES = ('dist_acf_is', 'dist_acf_w05', 'dist_ci', 'dist_product')
SINGLE_MODES = ('single_true', 'single_jam', 'single_pda', 'single_cons')


# ---------------------------------------------------------------- 工具
def _scene_dt(scene):
    """从场景时间栅格推采样周期（各章均为 0.02 s）；兜底用第4章值。"""
    t = scene['t']
    return float(t[1] - t[0]) if len(t) > 1 else C.SceneCh4.dt


def _init_local(scene, target_ie_rng, pij=None):
    """单平台本地 IMM 滤波器（与集中式路径同一初始化口径）。

    pij=None 时用 config.IMM['Pij']（默认，零破坏）；F41 统一重跑可传
    论文值 0.9/0.05（不改全局常量）。
    """
    ie = C.INIT_ERR
    P0 = np.diag(np.r_[np.full(3, ie['pos'] ** 2), np.full(3, ie['vel'] ** 2),
                       np.full(3, ie['acc'] ** 2)])
    x0 = np.r_[scene['target_p'][0] + target_ie_rng.normal(0, ie['pos'], 3),
               scene['target_v'][0] + target_ie_rng.normal(0, ie['vel'], 3),
               scene['target_a'][0] + target_ie_rng.normal(0, ie['acc'], 3)]
    f = IMMFilter(dt=_scene_dt(scene))
    if pij is not None:
        f.Pij = np.asarray(pij, float)
    f.initialize(x0, P0)
    return f


def local_gmm(f):
    """本地 IMM 后验的 GMM 参数 (μ_r, X_r, P_r) —— 式(3-25)。"""
    mu, xs, Ps = f.gmm()
    return mu, np.asarray(xs, float), np.asarray(Ps, float)


# ---------------------------------------------------------------- 退化分量
def fix_singular(covs, acc_jitter=1e-2, max_try=6, method='jitter',
                 weights=None):
    """把奇异/未定协方差修正为正定，返回 (covs_pd, 被修正的分量数)。

    **为什么必须做**：本实现的 CV 模型 `F` 的加速度行全为 0、同块 `Q` 也为 0
    （见 `src/imm.py::axis_FQ`，CV 的 `F[2,:]=0`、`Q[2,:]=0`），
    因此 CV 分量的加速度块**恒为 0** ⇒ 该分量协方差**奇异**（det=0）。

    而论文式(3-34)~(3-40) 的近似 Chernoff 融合需要 `inv(P_r)`。
    于是「把 IMM 局部后验 GMM 直接喂给 ACF」这一步会**失败**。
    **论文未说明如何处理这类退化分量** —— 这是第 3 章 ACF 可落地性的一个
    未说明前提，也是本复现新增的 B 级记录项。

    本复现的处理：只在**加速度对角**（状态 6,7,8）加一个物理上有界的小抖动
    `acc_jitter`（单位 m/s²，相对目标 8g≈78.5 m/s² 的机动幅度可忽略），
    并对 `acc_jitter` 做敏感性检查（见 `F9_jitter_sensitivity.csv`）。
    """
    out = []
    kept_idx = []
    n_fixed = 0
    n_dropped = 0
    for ci, S in enumerate(covs):
        S = 0.5 * (S + S.T)
        if method == 'drop' and not (bool(np.all(np.isfinite(S)))
                                     and float(np.linalg.det(S)) > 0):
            # ★ A5b 的另一种处理：**丢弃退化分量**（而非正则化）。
            #   CV 分量加速度块恒为 0 ⇒ 其加速度信息本就为 0，
            #   丢弃后由 CA/Singer 分量承担加速度估计。
            n_dropped += 1
            continue
        j = float(acc_jitter)
        fixed = False
        for _ in range(max_try):
            ok = bool(np.all(np.isfinite(S))) and float(np.linalg.det(S)) > 0
            if ok:
                break
            S = S.copy()
            S[6, 6] += j * j
            S[7, 7] += j * j
            S[8, 8] += j * j
            j *= 10.0
            fixed = True
        if fixed:
            n_fixed += 1
        out.append(S)
        kept_idx.append(ci)
    return np.array(out), kept_idx, n_fixed, n_dropped


def _psd_inv(S, max_try=8):
    """数值稳健求逆：对非正定/奇异矩阵按逐级放大的对角抖动重试。

    用途：软回灌（`fb_lambda < 1`）需要在信息域取 **本地混合协方差** 的逆，
    而本地 IMM 混合协方差在 CV 分量占优时其加速度子空间方差可退化为 0
    （与论文 §4.2.4 所述 CV 块奇异同源），`np.linalg.inv` 会直接抛
    `LinAlgError`（2026-10-17 审查在 λ 消融的强污染场景复现）。
    本函数只在**退化情形**触发抖动重试：正常正定矩阵的返回值与
    `np.linalg.inv` 逐位一致，故不改变既有数值。
    """
    S = np.asarray(S, float)
    S = 0.5 * (S + S.T)
    j = 1e-8
    for _ in range(max_try):
        if bool(np.all(np.isfinite(S))) and float(np.linalg.det(S)) > 0:
            try:
                return np.linalg.inv(S)
            except np.linalg.LinAlgError:
                pass
        S = S.copy()
        S[np.diag_indices_from(S)] += j
        j *= 100.0
    return np.linalg.pinv(S)


def fuse_ci(gmms, omega=None):
    """矩匹配后协方差交叉（传统单高斯融合）。

    omega=None 时取等权 1/N（**proper CI**）。
    注：`audit/t/t3_acf_integration.fuse_ci` 用固定 omega=0.5/源，
    对 N 个源会使信息矩阵整体放大 N·0.5 倍 ⇒ 协方差被缩放；
    两者**均值相同**（等比例缩放在 m=(ΣωA)⁻¹(ΣωAm) 中抵消），
    但**协方差不同**。本函数采用 proper CI。
    """
    n = len(gmms)
    if omega is None:
        omega = 1.0 / n
    d = len(gmms[0][1][0])
    A = np.zeros((d, d)); b = np.zeros(d)
    for (w, mus, covs) in gmms:
        m, Cm = gmm_mean_cov(w, mus, covs)
        Ai = np.linalg.inv(Cm)
        A += omega * Ai
        b += omega * (Ai @ m)
    C = np.linalg.inv(A)
    return C @ b, C


def fuse_product(gmms):
    """矩匹配后乘积（朴素贝叶斯）。"""
    d = len(gmms[0][1][0])
    A = np.zeros((d, d)); b = np.zeros(d)
    for (w, mus, covs) in gmms:
        m, Cm = gmm_mean_cov(w, mus, covs)
        Ai = np.linalg.inv(Cm)
        A += Ai
        b += Ai @ m
    C = np.linalg.inv(A)
    return C @ b, C


# ---------------------------------------------------------------- MATLAB 主链机制
# 以下两个机制在师姐原始 MATLAB 主链 (VGPO.m) 中启用, 但论文未记载
# （见 审查报告_matlab与python复现_20261016.md A5 项）。此处移植以闭合缺口；
# 默认**关闭**（net_reweight=False），不影响既有全部数值。
def prune_weights(prob, tol=1e-6):
    """MATLAB VGPO.m::prune_weights 移植。

    非有限权重置零、权重 < tol 的分量剔除、重新归一化；
    全部消失时退化为等权分布。
    """
    p = np.asarray(prob, float).copy()
    p[~np.isfinite(p)] = 0.0
    p[p < tol] = 0.0
    s = p.sum()
    if s < 1e-12:
        p = np.full_like(p, 1.0 / len(p))
    else:
        p = p / s
    return p


def net_speed_consistency_weight(modeX, prob, missile_p, tau=20.0):
    """MATLAB VGPO.m::net_speed_consistency_weight 移植（网络级虚假量测抑制）。

    对融合 GMM 的每个分量: 由其速度分量计算相对各平台视线的径向速度，
    以样本方差 J 度量跨平台速度一致性（真实目标分量各视线径向投影协调，
    J 小；干扰虚假分量 J 大），权重乘 exp(-J/tau) 后归一化。
    若全部分量修正权重消失，则保持输入权重不变（与 MATLAB 兜底一致）。

    modeX: (L, d) 各分量均值（与 `acf_fuse_log` 返回的 musf 同构）;
    prob: (L,); missile_p: (N,3) 当前时刻平台位置。
    """
    p = np.asarray(prob, float).copy()
    p[~np.isfinite(p)] = 0.0
    if p.sum() <= 0:
        p = np.full_like(p, 1.0 / len(p))
    else:
        p = p / p.sum()
    X = np.asarray(modeX, float)                     # (L, d)
    L = X.shape[0]
    n = len(missile_p)
    w = np.zeros(L)
    for l in range(L):
        x = X[l]
        if not np.all(np.isfinite(x)):
            continue
        Vt = x[3:6]
        vr = np.empty(n)
        ok = True
        for i in range(n):
            Rt = x[0:3] - missile_p[i]
            nr = float(np.linalg.norm(Rt))
            if nr < 1e-6:
                ok = False
                break
            vr[i] = float(Vt @ Rt) / nr
        if not ok or not np.all(np.isfinite(vr)):
            continue
        J = float(np.var(vr))
        w[l] = np.exp(-J / tau) if np.isfinite(J) else 0.0
    w[~np.isfinite(w)] = 0.0
    if float((w * p).sum()) <= 0:
        return p / p.sum()
    p = p * w
    return p / p.sum()


def _cand_meas(cand, k, i, sel, want_range):
    rows = []
    for m in sel:
        r = [cand['cand_alpha'][k, i, m], cand['cand_beta'][k, i, m],
             cand['cand_doppler'][k, i, m]]
        if want_range:
            r = [cand['cand_range'][k, i, m]] + r
        rows.append(r)
    return np.array(rows)


def _mode_marginals(w, labels, N, node_map=None):
    """把融合 GMM 的分量权重按**模态来源标签**聚合成逐节点的模态边缘概率。

    F73（路线 C）用：ACF 保持 GMM ⇒ 融合中心可提取"网络级模态概率"；
    CI 把局部 GMM 矩匹配成单高斯 ⇒ 无此量（调用方以局部边缘代替）。
    """
    out = np.zeros((N, 3))
    for c, t in enumerate(labels):
        for p in range(len(t)):
            n = p if node_map is None else node_map[p]
            out[n, t[p]] += w[c]
    s = out.sum(axis=1, keepdims=True)
    return out / np.where(s > 0, s, 1.0)


# ---------------------------------------------------------------- 主流程
def run_track(scene, cand, noise, mode='dist_acf_is', params=None, rng=None,
              n_missile=3, want_range=False, w_is_stride=100,
              w_is_samples=500, use_consistency=True, acc_jitter=1e-2,
              diag=False, mode_hist=False):
    """
    统一入口：分布式 ACF / 传统融合 / 单平台对照。

    返回 dict: est (K,9), err (K,3) [pos/vel/acc 误差范数],
               w_hist (K,) 融合权重, n_comp (K,) 融合后分量数, ngate (K,N)

    diag=True（F57 一致性统计用，默认 False 不改变既有行为）时额外返回：
        nees9 (K,) 融合输出 9 维 NEES；nees3 (K,) 位置子空间 NEES；
        cover95 (K,) 位置 95% 椭球是否覆盖真值；
        nees3_local (K,N) 各平台局部后验位置 NEES；
        nis (K,N) 各平台局部混合加权新息一致性统计量（自由度 = 4，含距离）
    """
    if rng is None:
        rng = np.random.default_rng(0)
    params = params if params is not None else C.CONSISTENCY
    K = len(scene['t'])
    mp = scene['missile_p']; mv = scene['missile_v']
    dt = _scene_dt(scene)
    R1 = measurement_noise_cov(noise, want_range=want_range)

    is_single = mode in SINGLE_MODES
    N = 1 if is_single else int(n_missile)
    if not is_single and mode not in DIST_MODES:
        raise ValueError('未知 mode: %s' % mode)

    fs = [_init_local(scene, rng, pij=params.get('pij')) for _ in range(N)]
    # E3：dual/combo_acc 前端状态（仅 ref_mode='dual' 时使用）
    combo_acc = {'logw': None, 'locked': None, 'k_last': None,
                 'v_ref': None}

    est = np.zeros((K, 9))
    err = np.zeros((K, 3))
    w_hist = np.full(K, np.nan)
    n_comp = np.zeros(K, dtype=int)
    ngate = np.zeros((K, N))
    dg = dict(nees9=np.full(K, np.nan), nees3=np.full(K, np.nan),
              cover95=np.zeros(K, dtype=bool),
              nees3_local=np.full((K, N), np.nan),
              nis=np.full((K, N), np.nan),
              node_labels=[])          # 逐帧各平台保留分量的模型标签（P1 审计用）
    mh = None
    if mode_hist:
        # F73（路线 C）：记录逐节点局部 IMM 模态后验与 ACF 融合后的网络级模态
        mh = dict(mu_loc=np.full((K, N, 3), np.nan),
                  mu_fuse=np.full((K, N, 3), np.nan))

    w_cur = 0.5                      # ACF 当前权重（分段更新）
    n_fix = 0                        # 退化协方差修正计数
    pij_next = [None] * N            # F44：自适应 TPM（逐平台）
    pij_prev = [None] * N
    xo = np.zeros(9)                 # 上一帧融合估计（E3 的 R_pred/v_ref 用）
    Po = np.eye(9)
    for k in range(1, K):
        # 外部方法选择器需要"上一帧的（融合）一步预测"；集中式路径里它就是 xp/Pp，
        # 分布式路径没有单一滤波器，故用上一帧融合输出近似（首帧用各平台共同初值）。
        xo_prev = np.asarray(xo, float).copy()
        Po_prev = np.asarray(Po, float).copy()
        if params.get('pij_adaptive'):
            for i in range(N):
                if pij_next[i] is not None:
                    fs[i].set_Pij(pij_next[i])
        # ---------- 一致性权重（全局几何量，按平台取行） ----------
        # 注：沿用集中式路径的默认口径 ref_mode='geom'（论文式4-15 位置域）。
        # 速度域变体 consistency_scores_vel 需要单一滤波器预测速度作基准，
        # 在分布式架构下无明确定义，故此处不使用。
        q_im = None
        w_dual = None
        q_dual = None
        if (mode in ('single_cons', 'dist_acf_is', 'dist_acf_w05', 'dist_ci',
                     'dist_product') and use_consistency
                and cand['n_cand'] > 1):
            _rm = params.get('ref_mode', 'geom')
            if _rm == 'dual':
                # E3：把 pipeline 的双通道门控/盲锚定移植到分布式权重段。
                score = None
                from .consistency import (dual_channel_weight,
                                          diag_combo_costs,
                                          rd_integral_stat,
                                          rd_integral_sigma)
                _L = int(params.get('dual_L', 100))
                _gate = params.get('dual_gate', 18.48)
                _ref = params.get('dual_ref', 'index')
                _ref_idx = int(params.get('dual_ref_idx', 0))
                _soft = bool(params.get('dual_soft', False))
                if _ref == 'combo_acc':
                    M = cand['n_cand']
                    if combo_acc['logw'] is None:
                        combo_acc['logw'] = np.zeros(M)
                    k0 = int(params.get('combo_start', 500))
                    stride = int(params.get('combo_stride', 5))
                    tau = float(params.get('combo_tau', 0.8))
                    gap_min = float(params.get('combo_gap', 10.0))
                    gamma_v = float(params.get('combo_gamma_v', 0.0))
                    # Opt2：T1 预筛候选池
                    t1_mask = None
                    if params.get('combo_t1_prefilter', False):
                        Z = rd_integral_stat(cand['cand_range'],
                                             cand['cand_doppler'], k, _L, dt)
                        sZ = rd_integral_sigma(noise.sig_r, noise.sig_d,
                                               _L, dt)
                        t1_mask = np.abs(Z) <= (
                            float(params.get('combo_t1_k', 3.0)) * sZ)
                        for _i in range(N):
                            if not t1_mask[_i].any():
                                t1_mask[_i, int(np.argmin(np.abs(Z[_i])))] = True
                    if k < k0:
                        combo_acc['v_ref'] = np.asarray(xo[3:6], float).copy()
                    if (combo_acc['locked'] is None and k >= k0 and
                            (combo_acc['k_last'] is None or
                             k - combo_acc['k_last'] >= stride)):
                        c, vc = diag_combo_costs(
                            cand, k, mp, noise.sig_r, noise.sig_az,
                            sig_d=noise.sig_d,
                            V_ref=combo_acc.get('v_ref'),
                            gamma_v=gamma_v, mv=mv[k], return_v=True,
                            mask=t1_mask)
                        cmin = float(np.min(c))
                        if not np.isfinite(cmin):
                            c = np.zeros_like(c); cmin = 0.0
                        combo_acc['logw'] += -0.5 * (c - cmin)
                        combo_acc['k_last'] = k
                        pp = np.exp(combo_acc['logw'] -
                                    combo_acc['logw'].max())
                        pp = pp / pp.sum()
                        lwf = combo_acc['logw'][np.isfinite(combo_acc['logw'])]
                        lwf = np.sort(lwf)
                        lw_gap = (float(lwf[-1] - lwf[-2])
                                  if len(lwf) > 1 else np.inf)
                        order = np.argsort(np.where(
                            np.isfinite(combo_acc['logw']),
                            combo_acc['logw'], -np.inf))
                        vb = order[-1]
                        vs = order[-2] if len(order) > 1 else vb
                        v_ok = (gamma_v > 0 and
                                (vc[vs] - vc[vb]) >=
                                float(params.get('combo_gap_v', 5.0)) and
                                vc[vb] <= float(params.get('combo_v_max', 8.0)))
                        if pp.max() >= tau and (lw_gap >= gap_min or v_ok):
                            combo_acc['locked'] = int(pp.argmax())
                    if combo_acc['locked'] is None:
                        _ref_eff, _idx_eff = 'combo', 0
                        _soft = bool(params.get('combo_pre_soft', False))
                    else:
                        _ref_eff, _idx_eff = 'index', int(combo_acc['locked'])
                else:
                    _ref_eff, _idx_eff = _ref, _ref_idx
                if k > _L:
                    R_pred = np.linalg.norm(
                        np.asarray(xo[0:3])[None, :] - mp[k], axis=1)
                    w_all, _q = dual_channel_weight(
                        cand, k, mp, noise.sig_r, noise.sig_az, noise.sig_d,
                        _L, dt, alpha=params.get('dual_alpha', 1.0),
                        chi2_gate=_gate, soft=_soft,
                        fallback=params.get('dual_fallback', 'argmin'),
                        ref_mode=_ref_eff, ref_idx=_idx_eff, R_pred=R_pred)
                    w_dual = w_all
                    q_dual = _q
            elif _rm == 'ext':
                # 外部方法对比（NN/GNN/DVB/MSVC/MSVC-TopK/FC）：选择器由实验侧构造，
                # 逐平台输出 (N,M) 权重；预测取上一帧融合估计（与集中式口径对应）。
                sel = params.get('ext_selector')
                if sel is None:
                    raise ValueError("ref_mode='ext' 需要 params['ext_selector']")
                if k == 1:
                    _xp = np.asarray(fs[0].x[0], float)
                    _Pp = np.asarray(fs[0].P[0], float)
                else:
                    _xp, _Pp = xo_prev, Po_prev
                w_dual = np.asarray(
                    sel.update(k, cand, _xp, _Pp, mp[k], np.asarray(mv)[k]),
                    float)

            elif _rm == 'pos_chi2':
                # ★ F0a: 论文式(26)-(28) 位置域白化统计量（默认不用）
                from .consistency import pos_chi2_scores
                q_im = pos_chi2_scores(cand, k, mp, mv, dt,
                                       noise.sig_el, noise.sig_d)
                score = None
            else:
                score, _VT, _Pg = consistency_scores(
                    cand, k, mp, mv, dt, ref_mode=_rm)
        else:
            score = None

        gmms = []
        kept_lab = []            # 逐平台保留分量的**全局模型索引**（A3 模式对齐用）
        last = None
        locals_est = []          # F43：逐平台本地混合估计（软回灌用）
        for i in range(N):
            f = fs[i]
            xp, Pp = f.predict_mixed()
            mpv = [(mp[k, i], mv[k, i])]
            z_pred = h_measure(xp, mp[k, i], mv[k, i], want_range)
            H = H_jacobian(xp, mp[k, i], mv[k, i], want_range)
            S = H @ Pp @ H.T + R1

            M = cand['n_cand']
            if mode == 'single_true':
                sel = [0]
            elif mode == 'single_jam':
                sel = [1] if M > 1 else [0]
            else:
                sel = list(range(M))
            zc = _cand_meas(cand, k, i, sel, want_range)
            if q_im is not None:
                # ★ F0a: 论文口径 χ² 门限清洗（全零 → 全部放行兜底）
                wc = (q_im[i] <= params.get('pos_chi2_thresh',
                                            11.34)).astype(float)
                if not np.any(wc > 0):
                    wc = np.ones(cand['n_cand'])
                wc = wc[sel]
            elif w_dual is not None:
                wc = w_dual[i][sel]
                if not np.any(wc > 0):
                    wc = np.ones(len(sel))
            elif score is not None:
                wc = consistency_weight(score[i], params['sigma_p'])
                # ★ B6 补齐：式(4-15) 门限判据（同 pipeline 侧，见该处注释）
                if params.get('cons_gate', False):
                    wc = wc * (score[i] <= params['pos_thresh'])
                wc = wc[sel]
            else:
                wc = np.ones(len(sel))
            if params.get('pda_cov', 'paper') in ('bs', 'jpda'):
                zf, Rf, beta, ng, Ci = pda_bs(
                    zc, z_pred, S, wc, params['gate_chi2'], R1,
                    p_detect=params.get('p_detect', 1.0),
                    on_empty=params.get('pda_on_empty',
                                        'skip' if params.get('pda_cov', 'paper') in ('bs', 'jpda') else 'force'),
                    empty_fallback=params.get('empty_fallback', 'miss'))
            else:
                zf, Rf, beta, ng = pda_fuse(
                    zc, z_pred, S, wc, params['gate_chi2'],
                    cov_mode=params.get('pda_cov', 'paper'), R1=R1,
                    empty_fallback=params.get('empty_fallback', None))
                Ci = None
            ngate[k, i] = ng

            xo, Po, mu, xs, Ps, _nus, _Ss = f.step(zf, mpv, Rf, want_range,
                                                   cov_add=Ci)
            if mh is not None:
                mh['mu_loc'][k, i] = mu
            last = (xo, Po)
            locals_est.append((np.asarray(xo, float).copy(),
                               np.asarray(Po, float).copy()))
            if diag:
                dpl = xo[0:3] - np.asarray(scene['target_p'][k], float)
                Ppl = 0.5 * (np.asarray(Po, float)[0:3, 0:3]
                             + np.asarray(Po, float)[0:3, 0:3].T)
                dg['nees3_local'][k, i] = float(dpl @ np.linalg.solve(Ppl, dpl))
                nis_i = 0.0
                for r in range(len(mu)):
                    nu_r = np.asarray(_nus[r], float).ravel()
                    nis_i += float(mu[r]) * float(
                        nu_r @ np.linalg.solve(np.asarray(_Ss[r], float), nu_r))
                dg['nis'][k, i] = nis_i
            # F44 钩子：自适应 TPM（默认关闭）
            if params.get('pij_adaptive'):
                loglik = np.empty(len(mu))
                for r in range(len(mu)):
                    nu_r = np.asarray(_nus[r], float).ravel()
                    S_r = np.asarray(_Ss[r], float)
                    loglik[r] = -0.5 * (nu_r @ np.linalg.solve(S_r, nu_r)
                                        + np.log(np.linalg.det(S_r) + 1e-300))
                pij_prev[i] = np.asarray(f.Pij, float)   # 本步所用
                base = np.asarray(params['pij_base'], float)
                ell = loglik - loglik.mean()
                mod = base * np.exp(float(params.get('pij_beta', 1.0)) * ell[None, :])
                mod = np.clip(mod, 1e-3, 0.3)
                pij_next[i] = mod / mod.sum(axis=1, keepdims=True)
            Ppd, kept, nfix, ndrop = fix_singular(
                np.asarray(Ps, float), acc_jitter,
                method=params.get('singular_method', 'jitter'))
            n_fix += nfix
            mu_a = np.asarray(mu, float)[kept]
            xs_a = np.asarray(xs, float)[kept]
            if ndrop:
                mu_a = mu_a / mu_a.sum()
            gmms.append((mu_a, xs_a, Ppd))
            kept_lab.append(np.asarray(kept, int))

        if diag:
            dg['node_labels'].append(
                [tuple(int(v) for v in lab) for lab in kept_lab])

        # ---------- 融合 ----------
        _gw = _gm = _gS = _glab = None   # E1：逐模式融合分量缓存
        if is_single:
            xo, Po = last
            n_comp[k] = 1
            if mh is not None:
                mh['mu_fuse'][k, 0] = mh['mu_loc'][k, 0]
        elif mode in ('dist_acf_is', 'dist_acf_w05'):
            if mode == 'dist_acf_is' and (k % w_is_stride == 1
                                          or np.isnan(w_hist[k - 1])):
                # 分段 IS 优化：用当前第一对局部 GMM 估一次 w，段内复用
                w1, m1, c1 = gmms[0]
                w2, m2, c2 = gmms[1]
                w_try, _ = optimize_weight(w1, m1, c1, w2, m2, c2,
                                           beta=C.CHERNOFF['beta'],
                                           n_samples=w_is_samples,
                                           tol=C.CHERNOFF['golden_tol'],
                                           rng=rng)
                if np.isfinite(w_try):
                    w_cur = float(min(max(w_try, 0.05), 0.95))
            _order = params.get('fusion_order')
            _gmms_f = ([gmms[j] for j in _order] if _order is not None
                       else gmms)
            # ★ ACF 变体分派（默认关闭：acf_cfg=None 时走原路径，逐位不变）
            _cfg = params.get('acf_cfg')
            # Opt3：按各节点最佳双通道统计量 q_min 调整融合指数；
            #   q_min<=ν 时 alpha=1（与等权逐位一致），受扰节点自动降权。
            if (_cfg is not None and params.get('fusion_adaptive', False)
                    and q_dual is not None):
                _cfg = dict(_cfg)
                _qarr = np.asarray(q_dual, float)
                _qarr = np.where(np.isfinite(_qarr), _qarr, 1e6)
                _qm = np.min(_qarr, axis=1)
                _nu = float(params.get('fusion_nu', 7.0))
                _a = np.exp(-np.maximum(0.0, _qm - _nu) / (2.0 * _nu))
                _a = np.maximum(_a, 1e-12)
                _cfg['alphas'] = _a / _a.sum()
            if _cfg is not None:
                from .acf_variants import fuse_variant
                # ★ 模式标签按**模型索引**传递（而非位置下标）：走 drop 路径时
                #   各平台保留的分量可能不同，按位置取对角会把 CV 与 CA 错配。
                _lab = ([kept_lab[j] for j in _order] if _order is not None
                        else list(kept_lab))
                if len(_lab) != len(_gmms_f):
                    _lab = None
                _gw, _gm, _gS, _glab, _info = fuse_variant(
                    [g[:3] for g in _gmms_f], _cfg, labels=_lab)
                xo, Po = gmm_mean_cov(_gw, _gm, _gS)
                n_comp[k] = len(_gw)
                w_hist[k] = w_cur
                if mh is not None and _glab is not None:
                    mh['mu_fuse'][k] = _mode_marginals(
                        _gw, _glab, N, node_map=None)
            else:
                if mh is not None:
                    ((xo, Po), gf, nc), flab = sequential_acf_fuse_log(
                        _gmms_f, w_seq=w_cur,
                        eta_convention=params.get('eta_convention', 'correct'),
                        return_labels=True)
                else:
                    (xo, Po), gf, nc = sequential_acf_fuse_log(
                        _gmms_f, w_seq=w_cur,
                        eta_convention=params.get('eta_convention', 'correct'))
                # ★ A5 缺口闭合：MATLAB 主链在最终融合后启用
                #   prune_weights + net_speed_consistency_weight（论文未记载）。
                #   论文 §4.2.2「分量剪枝」：每完成一次顺序融合即对分量权重剪枝
                #   （非有限置零、权重 <1e-6 的剔除、随后重新归一化）——2026-10-17
                #   审查发现此前仅在 net_reweight=True 时才剪枝，与论文正文/算法
                #   流程表（步骤 7「+ 分量剪枝」）不符，故改为**默认启用**；
                #   网络级重加权仍由 net_reweight 单独控制。
                gf_w = prune_weights(gf[0])
                if params.get('net_reweight', False):
                    gf_w = net_speed_consistency_weight(
                        gf[1], gf_w, mp[k], tau=params.get('net_tau', 20.0))
                xo, Po = gmm_mean_cov(gf_w, gf[1], gf[2])
                nc = int((gf_w > 0).sum())
                w_hist[k] = w_cur
                n_comp[k] = nc
                if mh is not None:
                    nm = (None if _order is None
                          else {p: _order[p] for p in range(len(_order))})
                    mh['mu_fuse'][k] = _mode_marginals(gf_w, flab, N, node_map=nm)
        else:
            fn = fuse_ci if mode == 'dist_ci' else fuse_product
            xo, Po = fn(gmms)
            n_comp[k] = len(gmms)
            if mh is not None:
                # CI 把各节点 GMM 矩匹配为单高斯 ⇒ 无网络级模态量，
                # 以节点边缘概率代替（对 CI 最有利的可用形式）。
                mh['mu_fuse'][k] = mh['mu_loc'][k]

        # ★ G1 优化方案：分布式融合反馈（reset 架构）。
        #   本地滤波器只吃自己的量测（弱可观）是分布式劣于集中式 ~8 m 的根因。
        #   每 `fb_period` 步把全局融合估计回灌本地滤波器（协方差乘 κ 防信息
        #   过度自信——局部与全局信息重叠，直接置 Po 会双计数）。
        #   默认关闭（fb_period=None），不影响既有数值。
        fb_period = params.get('fb_period')
        if fb_period and (not is_single) and (k % int(fb_period) == 0):
            kappa = float(params.get('fb_kappa', 1.0))
            lam = float(params.get('fb_lambda', 1.0))
            xg = np.asarray(xo, float)
            Pg_ = np.asarray(Po, float) * kappa
            # E1：分模式一对一回灌（仅 ACF 变体路径有逐模式分量；
            #   CI 路径 _glab=None 时自动退化为坍缩回灌）。
            per_mode = (params.get('fb_mode', 'collapse') == 'per_mode'
                        and lam >= 1.0 and _glab is not None
                        and _gm is not None)
            for i in range(N):
                if per_mode:
                    # _glab 的元素在 N 源融合下是长度 N 的模型标签元组；
                    # 仅当各源标签一致时才是有效的“同模式”分量。
                    by = {}
                    for gm, gS, gw, lb in zip(_gm, _gS, _gw, _glab):
                        vals = ({int(v) for v in lb}
                                if isinstance(lb, (tuple, list)) else {int(lb)})
                        if len(vals) == 1:
                            by[vals.pop()] = (gm, gS, gw)
                    nm = len(fs[i].models)
                    if set(range(nm)).issubset(by.keys()):
                        xs = [by[j][0] for j in range(nm)]
                        Ps = [kappa * by[j][1] for j in range(nm)]
                        mus = [by[j][2] for j in range(nm)]
                        fs[i].initialize_mixture(xs, Ps, mus)
                    else:                     # 模式缺失：安全退化
                        fs[i].initialize(xg, Pg_)
                elif lam >= 1.0:
                    fs[i].initialize(xg, Pg_)
                else:
                    # F43 软回灌：信息域凸组合
                    #   P_new=[(1−λ)P_i⁻¹+λ(κP_f)⁻¹]⁻¹，均值同型
                    xi, Pi = locals_est[i]
                    Ii = _psd_inv(Pi)
                    If = _psd_inv(Pg_)
                    Pn = _psd_inv((1.0 - lam) * Ii + lam * If)
                    xn = Pn @ ((1.0 - lam) * Ii @ xi + lam * If @ xg)
                    fs[i].initialize(xn, Pn)

        est[k] = xo
        err[k] = [np.linalg.norm(xo[0:3] - scene['target_p'][k]),
                  np.linalg.norm(xo[3:6] - scene['target_v'][k]),
                  np.linalg.norm(xo[6:9] - scene['target_a'][k])]
        if diag:
            xt = np.concatenate([scene['target_p'][k], scene['target_v'][k],
                                 scene['target_a'][k]])
            dx = np.asarray(xo, float) - xt
            Po_ = 0.5 * (np.asarray(Po, float) + np.asarray(Po, float).T)
            try:
                dg['nees9'][k] = float(dx @ np.linalg.solve(Po_, dx))
            except np.linalg.LinAlgError:
                # 后验协方差退化时退化为伪逆（忽略零空间）
                dg['nees9'][k] = float(dx @ np.linalg.pinv(Po_) @ dx)
            dxp = dx[0:3]
            try:
                dg['nees3'][k] = float(dxp @ np.linalg.solve(Po_[0:3, 0:3], dxp))
            except np.linalg.LinAlgError:
                dg['nees3'][k] = float(dxp @ np.linalg.pinv(Po_[0:3, 0:3]) @ dxp)
            dg['cover95'][k] = bool(dg['nees3'][k] <= 7.815)

    out = dict(est=est, err=err, w_hist=w_hist, n_comp=n_comp, ngate=ngate,
               n_singular_fixed=n_fix)
    if diag:
        out.update(dg)
    if mh is not None:
        out.update(mh)
    return out
