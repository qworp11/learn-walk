"""UDP-сервер игры: скиллы G1 (RL-политики + MuJoCo) в WSL, Godot на Windows.

Протокол (UDP, little-endian):
  клиент -> сервер "hello" (bytes)          -> ответ: JSON описания тел
  клиент -> сервер: <u32 0x494E 'IN'><u8 keys><u8 events><f32 yaw><f32 zoom>
            keys bit0..7: w a s d shift ctrl e q
            events bit0..3: jump fall getup terrain
  сервер -> клиент: <u32 0x3147 'G1'><u8 skill><u8 flags><u8 terrain>
                    <u16 nbody><f32 pelvis_z><f32 rew>
                    nbody x (pos3f + quat4f)   # мировые позы тел, body 1..N
Сервер отвечает на адрес последнего входящего пакета (обход NAT WSL2).
"""
import json
import math
import os
import socket
import struct
import sys
import time
import traceback

import mujoco
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import game  # noqa: E402
import steering  # noqa: E402

PORT = 7777
MAGIC_IN = 0x494E
MAGIC_OUT = 0x3147
CTRL_DT = 0.02           # 50 Гц, как в game.py
FALLEN_Z = 0.45
SITTING_Z = 0.68

KEYBITS = ["w", "a", "s", "d", "shift", "ctrl", "e", "q"]
EVBITS = ["jump", "fall", "getup", "terrain"]


def write_ip_hint():
    """Godot должен знать IP WSL (меняется между перезапусками)."""
    try:
        ip = open("/etc/resolv.conf").read()  # noqa - не то, см. ниже
    except Exception:
        pass
    try:
        out = os.popen("hostname -I").read().split()
        if out:
            dst = "/mnt/g/LEARN WALK/godot_game/server_ip.txt"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w") as f:
                f.write(out[0].strip())
    except Exception as e:
        print(f"server: ip hint не записан: {e}", flush=True)


def body_shape_info(model):
    """Размеры тел для Godot: главный геом каждого тела."""
    out = []
    for bid in range(1, model.nbody):
        best, info = -1.0, {"name": model.body(bid).name, "shape": "sphere",
                            "off": [0.0, 0.0, 0.0], "r": 0.04, "l": 0.0,
                            "s": [0.04, 0.04, 0.04], "quat": [1, 0, 0, 0]}
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) != bid:
                continue
            gt = int(model.geom_type[gid])
            size = model.geom_size[gid]
            gp = model.geom_pos[gid]
            gq = model.geom_quat[gid]
            CAP = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
            SPH = int(mujoco.mjtGeom.mjGEOM_SPHERE)
            BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
            if gt == CAP:
                vol = max(float(size[0]) ** 2 * float(size[1]) * 2.0, 1e-6)
            elif gt == SPH:
                vol = max(float(size[0]) ** 3 * 2.0, 1e-6)
            elif gt == BOX:
                vol = max(float(size[0]) * float(size[1]) * float(size[2]),
                          1e-6)
            else:
                continue
            if vol > best:
                best = vol
                info["off"] = [float(gp[0]), float(gp[1]), float(gp[2])]
                info["quat"] = [float(gq[0]), float(gq[1]), float(gq[2]),
                                float(gq[3])]
                if gt == CAP:
                    info.update(shape="capsule", r=float(size[0]),
                                l=float(size[1]) * 2.0)
                elif gt == SPH:
                    info.update(shape="sphere", r=float(size[0]), l=0.0)
                else:
                    info.update(shape="box", r=0.0, l=0.0,
                                s=[float(size[0]), float(size[1]),
                                   float(size[2])])
        out.append(info)
    return out


def main():
    write_ip_hint()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", PORT))
    sock.setblocking(False)
    print(f"server: UDP :{PORT}, жду Godot (hello)...", flush=True)

    model = game.visual_model()
    data = mujoco.MjData(model)
    nq = model.nq
    nbody = model.nbody - 1            # без world
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                      "torso_link")
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                 "torso_link")
    if torso_id < 0:
        torso_id = pelvis_id
    shapes = body_shape_info(model)
    hello_json = json.dumps({"bodies": shapes}).encode()

    def _yaw_of(quat):
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    mgr = game.SkillManager()
    cur = None
    sticky_left = 0
    obs_td = None
    forced_skill = None
    sit_armed = False
    sit_ready = game.skill_ready("sit")
    ep_rew = ep_steps = 0.0
    client = None
    keys = {}
    events = set()
    cam_yaw = 0.0

    _last_sit_check = 0.0
    next_tick = time.time()
    while True:
        # --- входящие пакеты ---
        try:
            while True:
                pkt, addr = sock.recvfrom(2048)
                client = addr
                if pkt == b"hello":
                    sock.sendto(hello_json, addr)
                    print(f"server: Godot подключился: {addr}", flush=True)
                    continue
                if len(pkt) >= 14 and struct.unpack_from("<I", pkt)[0] == \
                        MAGIC_IN:
                    kb, ev, yaw, _zoom = struct.unpack_from("<BBff", pkt, 4)
                    keys = {KEYBITS[i]: bool(kb & (1 << i))
                            for i in range(len(KEYBITS))}
                    keys = {k: v for k, v in keys.items() if v}
                    events = {EVBITS[i] for i in range(len(EVBITS))
                              if ev & (1 << i)}
                    cam_yaw = float(yaw)
        except BlockingIOError:
            pass
        except OSError as e:
            print(f"server: recv {e}", flush=True)

        if time.time() - _last_sit_check > 10:
            _last_sit_check = time.time()
            sit_ready = sit_ready or game.skill_ready("sit")

        # --- выбор скилла (копия логики game.py) ---
        if "terrain" in events:
            i = game.TERRAIN_ORDER.index(game.CURRENT_TERRAIN)
            newt = game.TERRAIN_ORDER[(i + 1) % len(game.TERRAIN_ORDER)]
            game._switch_terrain(newt, mgr)
            model = game.visual_model()
            data = mujoco.MjData(model)
            shapes = body_shape_info(model)
            hello_json = json.dumps({"bodies": shapes}).encode()
            if client is not None:
                sock.sendto(hello_json, client)
            cur, obs_td, sticky_left = None, None, 0
            ep_rew = ep_steps = 0.0

        want = forced_skill if forced_skill is not None else \
            game.desired_skill(list(events), keys, sit_ready)
        held_move = any(keys.get(k) for k in ("w", "e", "q")) or \
            (keys.get("ctrl") and sit_ready)
        if forced_skill is None and cur in game.STICKY and sticky_left > 0 \
                and not any(k in game.SKILLS for k in events) and not \
                held_move:
            want = cur
        forced_skill = None
        z_pelvis = float(data.xpos[pelvis_id][2])
        pinned = (cur in game.STICKY and sticky_left > 0)
        if obs_td is not None and z_pelvis < FALLEN_Z and want != "getup" \
                and not (pinned and cur == "fall"):
            want = "getup"
        sit_release = False
        if obs_td is not None and cur == "sit" and \
                FALLEN_Z <= z_pelvis < SITTING_Z and want != "sit":
            want = "sit"
            sit_release = True

        if want != cur or obs_td is None:
            old_venv = mgr.live[cur][0] \
                if cur is not None and cur in mgr.live else None
            got = mgr.get(want, protect=cur)
            if got is None:
                cur, sticky_left, obs_td = None, 0, None
                ep_rew = ep_steps = 0.0
            else:
                venv, _ = got
                obs_td, _ = venv.reset()
                steering.transplant(venv, old_venv)
                q = venv.unwrapped.device_layer.sim.data.qpos
                data.qpos[:] = q.detach().cpu().numpy()[0, :nq]
                mujoco.mj_forward(model, data)
                if want in game.LOCO_SKILLS:
                    t_start = steering.pick_start_frame(game._PRODS[want])
                elif want == "sit":
                    t_start = 60
                else:
                    t_start = None
                steering.begin(venv, (data.xpos[torso_id][0],
                                      data.xpos[torso_id][1]),
                               _yaw_of(data.xquat[torso_id]),
                               t_start=t_start)
                obs_td = steering.refresh_obs(venv)
                cur, sticky_left = want, game.SKILLS[want][3]
                ep_rew = ep_steps = 0.0
                print(f"server: скилл -> {want}", flush=True)
        events = set()

        # --- шаг физики ---
        if obs_td is not None:
            venv, policy = mgr.live[cur]
            prod = game._PRODS.get(cur)
            if prod is not None and prod._steer["on"]:
                if keys.get("w") or keys.get("e"):
                    steering.update(prod, cam_yaw, steering.MAX_RATE_WALK)
                elif keys.get("a") or keys.get("d"):
                    steering.turn_in_place(
                        prod, steering.MAX_RATE_STAND if keys.get("a")
                        else -steering.MAX_RATE_STAND)
                if cur in game.LOCO_SKILLS:
                    steering.advance_pin(
                        prod,
                        speed={"run": 0.9, "sprint": 1.3}.get(
                            cur, steering.CARROT_WALK),
                        robot_xy=(data.xpos[torso_id][0],
                                  data.xpos[torso_id][1]))
                steering.guard(prod, (data.xpos[torso_id][0],
                                      data.xpos[torso_id][1]))
                if cur == "sit" and prod is not None:
                    ts = int(prod.time_steps[0].item())
                    T_sit = int(prod.motion.time_step_total)
                    if keys.get("ctrl") and not sit_release:
                        sit_armed = True
                        if ts > game.SIT_HOLD_FRAME:
                            prod.time_steps[:] = game.SIT_HOLD_FRAME
                    elif sit_armed or sit_release:
                        sit_armed = False
                        need = (game.SIT_RISE_DONE - ts) \
                            if ts <= game.SIT_RISE_DONE \
                            else (game.SIT_RISE_DONE + T_sit - ts)
                        sticky_left = max(sticky_left, need + 20)
            with torch.no_grad():
                action = policy(obs_td)
            obs_td, rew, done, _ = venv.step(action)
            ep_rew += float(rew.mean().item())
            ep_steps += 1
            if cur in game.STICKY:
                sticky_left -= 1
            q = venv.unwrapped.device_layer.sim.data.qpos
            data.qpos[:] = q.detach().cpu().numpy()[0, :nq]
            mujoco.mj_forward(model, data)
            if done:
                if data.xpos[pelvis_id][2] < FALLEN_Z and cur != "getup":
                    forced_skill = "getup"
                else:
                    forced_skill = "stand"
                ep_rew = ep_steps = 0.0

        # --- пакет поз ---
        if client is not None:
            flags = 0
            if obs_td is None:
                flags |= 4                      # загрузка
            else:
                if z_pelvis < FALLEN_Z:
                    flags |= 1                  # упал/лежит
                elif z_pelvis < SITTING_Z:
                    flags |= 2                  # сидит/низко
            pos = data.xpos[1:].astype("<f4").reshape(-1)
            quat = data.xquat[1:].astype("<f4").reshape(-1)
            head = struct.pack(
                "<IBBBHff", MAGIC_OUT, game.SKILLS.get(
                    cur, (None, "?", None, None))[0].count("s") * 0
                if False else _skill_id(cur), flags,
                game.TERRAIN_ORDER.index(game.CURRENT_TERRAIN),
                nbody, float(data.xpos[pelvis_id][2]),
                ep_rew / max(ep_steps, 1))
            sock.sendto(head + pos.tobytes() + quat.tobytes(), client)

        # --- темп 50 Гц ---
        next_tick += CTRL_DT
        delay = next_tick - time.time()
        if delay > 0:
            time.sleep(delay)
        else:
            next_tick = time.time()


def _skill_id(cur):
    """0..7 для HUD: stand walk run sprint jump fall getup dance sit."""
    order = ["stand", "walk", "run", "sprint", "jump", "fall", "getup",
             "dance", "sit"]
    try:
        return order.index(cur)
    except ValueError:
        return 0


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
