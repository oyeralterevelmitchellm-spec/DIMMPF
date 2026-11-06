# -*- coding: utf-8 -*-
"""
三维平台-目标运动场景生成。

包含：
  * 目标 S 形规避机动 (分段恒法向过载, 航向变化达阈值后反向)  [CH4 4.4.1 / CH2 2.5.3]
  * 平台 PPN (a_M = N * (omega x V_M)) 生成协同接近航迹（航迹生成器，非制导/作战评价）   [CH2/CH3/CH4]
  * 前置角 (lead angle) 初始对准, 保证编队可成功接近终止
  * 接近终止判定 (任一平台-目标距离 < 5 m)
"""
import numpy as np

G0 = 9.80665


def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v * 0.0
    return v / n


def _wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def lead_point(pm, pt, vt, vm_speed):
    """求解平台以常速 vm_speed 接近匀速目标 pt+vt*t 的交会点 (前置角对准)。"""
    r = pt - pm
    a = float(vt @ vt - vm_speed ** 2)
    b = 2.0 * float(r @ vt)
    c = float(r @ r)
    disc = b * b - 4 * a * c
    if disc < 0 or abs(a) < 1e-12:
        return pt
    t = (-b - np.sqrt(disc)) / (2 * a)
    if t <= 0:
        t = (-b + np.sqrt(disc)) / (2 * a)
    if t <= 0:
        return pt
    return pt + vt * t


def simulate_scene(target_p0, target_v0, target_acc_g, missiles_p0, missiles_v0,
                   dt, t_max, pn_gain=3.0, intercept_r=5.0, turn_deg=60.0,
                   maneuver_start=0.0, ca_time=0.0, ca_acc=None,
                   acc_dir_fn=None, lead_init=False, rng=None):
    """
    通用场景仿真 (返回逐步真值序列)。

    ca_time / ca_acc : 前 ca_time 秒的匀加速段 (加速度矢量)
    maneuver_start   : 目标开始 S 机动的时刻 (s)
    lead_init        : 是否按前置角设置平台初始速度方向
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n_m = missiles_p0.shape[0]
    n_steps = int(t_max / dt) + 1

    tp = target_p0.astype(float).copy()
    tv = target_v0.astype(float).copy()
    mp = missiles_p0.astype(float).copy()
    mv = missiles_v0.astype(float).copy()

    if lead_init:
        for i in range(n_m):
            sp = np.linalg.norm(mv[i])
            aim = lead_point(mp[i], tp, tv, sp)
            mv[i] = unit(aim - mp[i]) * sp

    TP = np.zeros((n_steps, 3)); TV = np.zeros((n_steps, 3)); TA = np.zeros((n_steps, 3))
    MP = np.zeros((n_steps, n_m, 3)); MV = np.zeros((n_steps, n_m, 3))
    T = np.zeros(n_steps)

    psi_ref = np.arctan2(tv[1], tv[0])
    sign = 1.0
    t_hit = np.nan
    hit_idx = n_steps - 1
    kz = np.array([0.0, 0.0, 1.0])

    for k in range(n_steps):
        t = k * dt
        T[k] = t
        TP[k] = tp; TV[k] = tv
        MP[k] = mp; MV[k] = mv

        d = np.linalg.norm(mp - tp, axis=1)
        if np.min(d) < intercept_r:
            t_hit = t
            hit_idx = k
            TP[k + 1:] = tp; TV[k + 1:] = tv; TA[k + 1:] = 0.0
            MP[k + 1:] = mp; MV[k + 1:] = mv
            T[k + 1:] = t + np.arange(1, n_steps - k) * dt
            break

        # ---------- 目标加速度 ----------
        if t < ca_time:
            ta = np.zeros(3) if ca_acc is None else np.asarray(ca_acc, float)
        elif t >= maneuver_start and target_acc_g > 0:
            psi = np.arctan2(tv[1], tv[0])
            if abs(_wrap_pi(psi - psi_ref)) >= np.deg2rad(turn_deg):
                sign *= -1.0
                psi_ref = psi
            if acc_dir_fn is not None:
                adir = unit(np.asarray(acc_dir_fn(tp, tv), float))
            else:
                adir = unit(np.cross(tv, kz))
                if np.linalg.norm(adir) < 1e-9:
                    adir = np.array([0.0, 1.0, 0.0])
            ta = sign * target_acc_g * G0 * adir
        else:
            ta = np.zeros(3)
        TA[k] = ta

        # ---------- 平台 PPN 航迹生成（非制导/作战评价） ----------
        ma = np.zeros((n_m, 3))
        for i in range(n_m):
            r = tp - mp[i]
            R = np.linalg.norm(r)
            if R < 1e-6:
                continue
            u = r / R
            vrel = tv - mv[i]
            omega = np.cross(r, vrel) / (R * R)
            ma[i] = pn_gain * np.cross(omega, mv[i])
        an = np.linalg.norm(ma, axis=1)
        cap = 500.0 * G0
        for i in range(n_m):
            if an[i] > cap:
                ma[i] *= cap / an[i]

        # ---------- 积分 ----------
        tv = tv + ta * dt
        tp = tp + tv * dt
        mv = mv + ma * dt
        mp = mp + mv * dt

    return dict(target_p=TP[:hit_idx + 1], target_v=TV[:hit_idx + 1],
                target_a=TA[:hit_idx + 1], missile_p=MP[:hit_idx + 1],
                missile_v=MV[:hit_idx + 1], t=T[:hit_idx + 1],
                t_hit=t_hit, hit_idx=hit_idx)


# ----------------------------------------------------------------------
# 场景构造器
# ----------------------------------------------------------------------
def build_ch4_scene(rng=None, target_p0=None, target_v0=None, maneuver_g=None,
                    t_max=None, lead_init=True):
    """第4章 / 期刊初稿主场景：60 度构型三平台协同 + 8g S 机动目标。"""
    from . import config as C
    cfg = C.SceneCh4
    p0 = cfg.target_p0 if target_p0 is None else np.asarray(target_p0, float)
    if target_v0 is None:
        th = np.deg2rad(cfg.target_v_dir_deg)
        v0 = np.array([np.cos(th), np.sin(th), 0.0]) * cfg.target_speed
    else:
        v0 = np.asarray(target_v0, float)
    mg = cfg.target_maneuver_g if maneuver_g is None else maneuver_g

    mp0 = cfg.missile_p0
    mv0 = np.array([unit(p0 - mp0[i]) * cfg.missile_speed[i]
                    for i in range(mp0.shape[0])])

    return simulate_scene(p0, v0, mg, mp0, mv0, cfg.dt,
                          cfg.t_max if t_max is None else t_max,
                          pn_gain=cfg.pn_gain, intercept_r=cfg.intercept_r,
                          turn_deg=cfg.target_turn_deg,
                          maneuver_start=cfg.maneuver_start,
                          lead_init=lead_init, rng=rng)


def build_ch2_scene(scenario='S3_60deg', rng=None, t_max=None,
                    lead_init=False):
    """[CH2 2.5.3] 多平台协同探测可行性论证的四场景。

    S1_5deg / S2_30deg / S3_60deg：三平台非共面等角（5°/30°/60°）协同观测；
    S4_single：单平台独立测量（对照基准）。

    平台初始速度**直接使用原文给出值**（`lead_init=False`），
    因为原文已按构型逐平台给定 v0；若再按前置角重定向会破坏忠实性。
    """
    from . import config as C
    cfg = C.SceneCh2
    mp0 = cfg.scenarios[scenario]
    n = mp0.shape[0]
    mv0 = cfg.missile_v0[:n] if n == 3 else cfg.missile_v0_single
    return simulate_scene(cfg.target_p0, cfg.target_v0, cfg.target_maneuver_g,
                          mp0, mv0, cfg.dt,
                          cfg.t_max if t_max is None else t_max,
                          pn_gain=cfg.pn_gain, intercept_r=cfg.intercept_r,
                          turn_deg=cfg.target_turn_deg,
                          maneuver_start=cfg.maneuver_start,
                          lead_init=lead_init, rng=rng)


def build_ch3_scene(config_name='60deg', rng=None, t_max=None, lead_init=True):
    """第3章场景：目标先匀速 10 s 再 9g S 机动。"""
    from . import config as C
    cfg = C.SceneCh3
    mp0 = cfg.configs[config_name]
    mv0 = cfg.missile_v0.copy()
    return simulate_scene(cfg.target_p0, cfg.target_v0, cfg.target_maneuver_g,
                          mp0, mv0, cfg.dt,
                          cfg.t_max if t_max is None else t_max,
                          pn_gain=cfg.pn_gain, intercept_r=cfg.intercept_r,
                          turn_deg=cfg.target_turn_deg,
                          maneuver_start=cfg.target_ca_time,
                          lead_init=lead_init, rng=rng)


# ----------------------------------------------------------------------
# 随机几何协议（2026-11-03）：有界抖动 + 可容许集 + 拒绝重采样
# ----------------------------------------------------------------------
def sample_jitter(rng, spec, base):
    """在名义初始条件上做各向同性抖动，返回可直接喂给 simulate_scene 的参数。"""
    p0 = np.asarray(base['target_p0'], float) + \
        rng.normal(0.0, spec['target_pos_sigma'], 3)
    heading = np.deg2rad(base['target_v_dir_deg']) + \
        np.deg2rad(rng.normal(0.0, spec['target_heading_sigma']))
    speed = base['target_speed'] * (1.0 + rng.normal(0.0,
                                                     spec['target_speed_rel_sigma']))
    v0 = np.array([np.cos(heading), np.sin(heading), 0.0]) * speed
    mp0 = np.asarray(base['missile_p0'], float) + \
        rng.normal(0.0, spec['platform_pos_sigma'], (3, 3))
    t_man = base['maneuver_start'] + rng.normal(0.0, spec['maneuver_start_sigma'])
    return dict(target_p0=p0, target_v0=v0, missiles_p0=mp0,
                maneuver_start=float(t_man))


def geometry_metrics(sc):
    """整窗几何量：两两视线角、cond(U)、R0、接近速度、是否接近终止。"""
    tp = np.asarray(sc['target_p'], float)
    mp = np.asarray(sc['missile_p'], float)
    r = tp[:, None, :] - mp                       # (K,N,3)
    dist = np.linalg.norm(r, axis=2)              # (K,N)
    los = r / np.maximum(dist, 1e-9)[:, :, None]
    K, N = dist.shape

    ang = []
    for i in range(N):
        for j in range(i + 1, N):
            c = np.clip(np.sum(los[:, i] * los[:, j], axis=1), -1.0, 1.0)
            ang.append(np.degrees(np.arccos(c)))
    ang = np.asarray(ang)                          # (Npairs, K)

    cond = np.empty(K)
    for k in range(K):
        sv = np.linalg.svd(los[k], compute_uv=False)
        cond[k] = sv[0] / max(sv[-1], 1e-12)

    dt = float(sc['t'][1] - sc['t'][0])
    closure = -(dist[-1] - dist[0]) / max(dt * (K - 1), 1e-9)
    return dict(los_pair_min=float(ang.min()), los_pair_max=float(ang.max()),
                los_mean=float(ang.mean()), cond_max=float(cond.max()),
                cond_min=float(cond.min()), R0=dist[0], closure=closure,
                n_frames=K, t_end=float(sc['t'][-1]),
                hit=bool(dist[-1].min() < 6.0))


def check_admissible(geo, adm):
    """返回 (ok, reasons)。全部判据都基于整窗量。"""
    bad = []
    if geo['los_pair_min'] < adm['los_pair_min_deg']:
        bad.append('los_pair_min=%.2f' % geo['los_pair_min'])
    if geo['los_pair_max'] > adm['los_pair_max_deg']:
        bad.append('los_pair_max=%.2f' % geo['los_pair_max'])
    if not (adm['los_mean_min_deg'] <= geo['los_mean'] <= adm['los_mean_max_deg']):
        bad.append('los_mean=%.2f' % geo['los_mean'])
    if geo['cond_max'] > adm['cond_u_max'] or geo['cond_min'] < adm['cond_u_min']:
        bad.append('cond=[%.2f,%.2f]' % (geo['cond_min'], geo['cond_max']))
    if geo['R0'].min() < adm['R0_min'] or geo['R0'].max() > adm['R0_max']:
        bad.append('R0=[%.0f,%.0f]' % (geo['R0'].min(), geo['R0'].max()))
    if geo['closure'].min() < adm['closure_min']:
        bad.append('closure_min=%.1f' % geo['closure'].min())
    if geo['t_end'] < adm['t_hit_min']:
        bad.append('t_end=%.2f' % geo['t_end'])
    if geo['n_frames'] < adm['n_frames_min']:
        bad.append('n_frames=%d' % geo['n_frames'])
    return (len(bad) == 0), bad


def simulate_admissible(rng_of_attempt, base, spec, adm, *,
                        dt, t_max, kw_scene, k_win=1000,
                        max_tries=64, shrink=0.7):
    """拒绝重采样直到几何可容许；每次尝试用独立子流（attempt 索引）。

    rng_of_attempt: callable(attempt) -> np.random.Generator
    返回 (scene_truncated, meta)；失败抛 RuntimeError（不产出勉强合格的场景）。
    """
    tries = 0
    scale = 1.0
    reasons = []
    while tries < max_tries:
        jit = dict(spec)
        for k in list(jit):
            jit[k] = float(jit[k]) * scale
        rng = rng_of_attempt(tries)
        par = sample_jitter(rng, jit, base)
        mv0 = np.array([unit(par['target_p0'] - par['missiles_p0'][i]) *
                        base['missile_speed'][i] for i in range(3)])
        sc = simulate_scene(par['target_p0'], par['target_v0'],
                            base['target_maneuver_g'], par['missiles_p0'], mv0,
                            dt, t_max, maneuver_start=par['maneuver_start'],
                            **kw_scene)
        geo = geometry_metrics(sc)
        ok, why = check_admissible(geo, adm)
        tries += 1
        if ok:
            Ktot = len(sc['t'])
            out = {}
            for k, v in sc.items():
                if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == Ktot:
                    out[k] = v[:k_win]
                else:
                    out[k] = v
            meta = dict(tries=tries, scale=scale, accepted=True,
                        **{k: v for k, v in geo.items() if k != 'R0'},
                        R0_min=float(geo['R0'].min()),
                        R0_max=float(geo['R0'].max()))
            return out, meta
        reasons.append(why)
        if tries % 8 == 0:                    # 每 8 次失败收窄一次
            scale *= shrink
    raise RuntimeError('几何可容许集内 64 次采样全部失败，最后一次原因：%s'
                       % reasons[-1])
