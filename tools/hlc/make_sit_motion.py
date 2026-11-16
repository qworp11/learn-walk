"""Синтез референса «СЕСТЬ» для G1 — без mocap и лицензий.

Формат 1:1 с csv_to_npz.py (fps, joint_pos, joint_vel, body_pos_w (30,3),
body_quat_w (30,4), body_lin_vel_w, body_ang_vel_w), 50 Гц, честная FK
модели g1_robot_spec.

Таймлайн: 0-1с стойка → 1-2.5с плавный присед-сед → 2.5-4с удержание →
4-5.5с подъём → 5.5-6.5с стойка.

Высота таза на КАЖДОМ кадре решается из FK так, чтобы стопы стояли на
полу (как в стойке) — контакт всегда по стопам, никакой магии.
"""

import numpy as np
import mujoco
from mujoco_playground.robots.g1.sim import scene as g1_scene

FPS = 50.0
DT = 1.0 / FPS
OUT = "/mnt/g/LEARN WALK/data/motions/npz_g1/sit_synth.npz"


def smoothstep(x):
    return x * x * (3.0 - 2.0 * x)


def main():
    spec = g1_scene.g1_robot_spec()
    model = spec.compile()
    data = mujoco.MjData(model)
    nj = model.nq - 7
    print(f"модель: nq={model.nq} nj={nj} bodies={model.nbody}")
    if model.nbody - 1 != 30:
        raise SystemExit(f"ожидал 30 тел, а тут {model.nbody - 1} — СТОП")

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
             for j in range(model.njnt)]
    ranges = model.jnt_range

    def clamp(key, val):
        for j, nm in enumerate(names):
            if key in nm:
                lo, hi = ranges[j]
                return float(np.clip(val, lo, hi))
        return None

    def make_pose(hip, knee, ankle, shoulder):
        p = np.zeros(nj)
        for j, nm in enumerate(names):
            v = 0.0
            if "hip_pitch" in nm:
                v = hip
            elif "knee" in nm:
                v = knee
            elif "ankle_pitch" in nm:
                v = ankle
            elif "hip_roll" in nm:
                v = 0.10
            elif "hip_yaw" in nm:
                v = 0.15
            elif "waist_pitch" in nm:
                v = 0.25
            elif "shoulder_pitch" in nm:
                v = shoulder
            elif "elbow" in nm:
                v = 0.6
            if v != 0.0:
                lo, hi = ranges[j]
                p[j] = float(np.clip(v, lo, hi))
        return p

    def fk(z, joints):
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = [0.0, 0.0, z]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[7:] = joints
        mujoco.mj_kinematics(model, data)
        return data.xpos.copy(), data.xquat.copy()

    def ids_by(sub1, sub2):
        out = []
        for i in range(1, model.nbody):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if sub1 in nm and sub2 in nm:
                out.append(i)
        return out

    ank_ids = ids_by("ankle_roll", "")  # left/right ankle_roll bodies
    assert len(ank_ids) == 2, f"ожидал 2 ankle_roll тела, найдено {ank_ids}"

    z0 = float(model.qpos0[2])
    X0, _ = fk(z0, np.zeros(nj))
    h_ank = min(X0[i][2] for i in ank_ids)   # высота стопы в стойке
    print(f"стойка: pelvis z={z0:.3f}, ankle z={h_ank:.3f}")

    # --- подбор позы «сидя»: перебор углов, критерий — глубина и симметрия ---
    best = None
    for hip in np.arange(-1.4, -2.05, -0.2):
        for knee in np.arange(1.2, 2.25, 0.2):
            for ankle in np.arange(0.0, 0.55, 0.1):
                p = make_pose(hip, knee, ankle, 0.0)
                X, _ = fk(z0, p)
                la = X[ank_ids[0]][2] - z0   # вертикальный офсет стопы
                ra = X[ank_ids[1]][2] - z0
                z_sit = h_ank - min(la, ra)
                asym = abs(la - ra)
                if not (0.20 <= z_sit <= 0.45) or asym > 0.05:
                    continue
                # чем ниже таз (глубже сед), тем лучше; штраф за асимметрию
                score = z_sit + asym
                if best is None or score < best[0]:
                    best = (score, hip, knee, ankle, p, z_sit)
    if best is None:
        raise SystemExit("не нашёл позу сидя в диапазоне z 0.20-0.45 — СТОП")
    _, hip, knee, ankle, sit_pose, z_sit = best
    print(f"поза сидя: hip={hip:.2f} knee={knee:.2f} ankle={ankle:.2f} "
          f"-> pelvis z={z_sit:.3f}")

    # руки: выбираем знак плеча, при котором кисть ВПЕРЕДИ корпуса (+x)
    Xs, _ = fk(z_sit, sit_pose)
    sh_ids = [i for i in range(1, model.nbody)
              if "shoulder_pitch" in
              (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "")]
    pelvis_x = Xs[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                    "pelvis")][0]
    fwd = np.mean([Xs[i][0] - pelvis_x for i in sh_ids]) if sh_ids else 0.0
    shoulder = -1.2 if fwd >= 0 else 1.2
    sit_pose = make_pose(hip, knee, ankle, shoulder)
    print(f"плечи: знак {shoulder:+.1f} (кисти вперед)")

    # --- таймлайн ---
    T = 7.0
    n = int(T * FPS)
    ts = np.arange(n) / FPS
    j_t = np.zeros((n, nj))
    z_t = np.full(n, z0)

    def seg(t0, t1, a_j, b_j, a_z, b_z):
        for i, t in enumerate(ts):
            if t0 <= t <= t1:
                a = smoothstep((t - t0) / (t1 - t0))
                j_t[i] = a_j + (b_j - a_j) * a

    seg(1.0, 2.5, np.zeros(nj), sit_pose, z0, z_sit)
    seg(4.0, 6.0, sit_pose, np.zeros(nj), z_sit, z0)
    j_t[126:200] = sit_pose     # удержание 2.5-4.0 с: кадры 126-199 не
                                # затронуты seg — заполняем явно (иначе скачок)

    # высота таза: на каждом кадре FK решает z так, чтобы стопы были на полу
    body_pos, body_quat = [], []
    for i in range(n):
        X, _ = fk(z0, j_t[i])                       # C_i при тестовом z0
        ci = min(X[k][2] for k in ank_ids) - z0
        zi = h_ank - ci                             # стопы ровно на полу
        zi = float(np.clip(zi, 0.05, z0))
        Xq, Qq = fk(zi, j_t[i])                     # финальный кадр
        body_pos.append(Xq[1:].copy())
        body_quat.append(Qq[1:].copy())
    body_pos = np.stack(body_pos)
    body_quat = np.stack(body_quat)

    joint_vel = np.gradient(j_t, DT, axis=0)
    lin_vel = np.gradient(body_pos, DT, axis=0)

    import scipy.spatial.transform as st

    Rq = st.Rotation.from_quat(body_quat[:, :, [1, 2, 3, 0]])  # wxyz→xyzw
    rd = (Rq[1:] * Rq[:-1].inv()).as_rotvec() / DT               # (n-1, 30, 3)
    ang_vel = np.empty((n, body_pos.shape[1], 3))
    ang_vel[1:] = rd
    ang_vel[0] = rd[0]

    np.savez(
        OUT,
        fps=np.array([FPS]),
        joint_pos=j_t.astype(np.float64),
        joint_vel=joint_vel.astype(np.float64),
        body_pos_w=body_pos.astype(np.float64),
        body_quat_w=body_quat.astype(np.float64),
        body_lin_vel_w=lin_vel.astype(np.float64),
        body_ang_vel_w=ang_vel.astype(np.float64),
    )
    print(f"сохранено: {OUT} кадров={n}, z(сидя)={z_sit:.3f}")
    print("скорости max: joint", np.abs(joint_vel).max().round(3),
          "lin", np.abs(lin_vel).max().round(3),
          "ang", np.abs(ang_vel).max().round(3))


if __name__ == "__main__":
    main()
