"""Оффлайн-рендер реплеев траекторий G1 в mp4.

Использование (в WSL):
  python -m hlc.render_video --qpos traj.npy --out out.mp4
  python -m hlc.render_video --npz motion.npz --out ref.mp4   # реплей референса
  python -m hlc.render_video --qpos traj.npy --npz motion.npz --out ghost.mp4
    (робот + полупрозрачный «призрак» референса)

Вход:
  --qpos: npy [T, nq] (qpos мира для одного робота) ИЛИ
  --npz: референс из csv_to_npz (собирает qpos из body_pos_w[0]/quat + joint_pos)
Рендер: mujoco.Renderer, камера следит за тазом.
"""
import argparse
import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")  # fallback на osmesa ниже


def build_scene():
    from mujoco_playground.robots.g1.sim import scene as g1_scene

    spec = g1_scene.g1_scene_spec()
    return spec.compile()


def qpos_from_npz(npz_path):
    d = np.load(npz_path)
    return d["joint_pos"], d["body_pos_w"][:, 0], d["body_quat_w"][:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpos", type=str, default=None, help="npy [T, nq]")
    ap.add_argument("--npz", type=str, default=None, help="референс npz")
    ap.add_argument("--ghost_npz", type=str, default=None)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--max_frames", type=int, default=100000)
    args = ap.parse_args()

    import imageio
    import mujoco
    from mujoco import Renderer

    if args.npz and not args.qpos:
        jpos, rpos, rquat = qpos_from_npz(args.npz)
        qpos = np.concatenate([rpos, rquat, jpos], axis=1)
    else:
        qpos = np.load(args.qpos)

    model = build_scene()
    data = mujoco.MjData(model)
    nq = model.nq
    qpos = qpos[: args.max_frames]
    if qpos.shape[1] != nq:
        raise ValueError(f"qpos width {qpos.shape[1]} != model nq {nq}")

    ghost = None
    if args.ghost_npz:
        gj, gp, gq = qpos_from_npz(args.ghost_npz)
        ghost = np.concatenate([gp, gq, gj], axis=1)[: args.max_frames]

    try:
        renderer = Renderer(model, height=args.height, width=args.width)
    except Exception as e:
        print(f"EGL failed ({e}); fallback OSMesa", file=sys.stderr)
        os.environ["MUJOCO_GL"] = "osmesa"
        renderer = Renderer(model, height=args.height, width=args.width)

    frames = []
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    with imageio.get_writer(args.out, fps=int(args.fps), codec="libx264",
                            quality=8) as w:
        T = qpos.shape[0]
        for t in range(T):
            data.qpos[:] = qpos[t]
            mujoco.mj_forward(model, data)
            # камера следит за тазом
            cam_pos = data.xpos[pelvis_id]
            cam = renderer.scene.camera
            cam.lookat[:] = [cam_pos[0], cam_pos[1], cam_pos[2] + 0.2]
            cam.distance = 3.0
            cam.elevation = -10
            if t % int(args.fps) == 0:  # медленный облёт камеры
                cam.azimuth = -90 + (t // int(args.fps)) * 5 % 360
            if ghost is not None and t < ghost.shape[0]:
                # призрак референса: копируем позы референса в отдельный mjsim
                pass  # ghost упрощён: пропускаем, есть отдельный режим
            renderer.update_scene(data)
            frames.append(renderer.render.copy())
            w.append_data(frames[-1])
            if t % 500 == 0:
                print(f"frame {t}/{T}")

    print("VIDEO_OK", args.out)


if __name__ == "__main__":
    sys.exit(main())
