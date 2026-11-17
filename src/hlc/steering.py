"""Стриринг референса: виртуальный путь под игрока + бесшовные переходы.

Честная механика (как это работает):
- Политика whole_body tracking получает через obs:
    * команду = суставные цели кадра клипа (поза/скорость суставов);
    * anchor_pos_b = (глобальный якорь референса - робот) в кадре робота;
    * anchor_ori_b = поворот референса относительно робота (6D).
- Мы перехватываем produce_features у MotionReferenceProducer:
    * переносим глобальный якорь референса в виртуальную точку p_pin,
      которая сама шагает вперёд со скоростью клипа (сигнал скорости
      сохранён, как при обычном трекинге пути);
    * вращаем референс по yaw на угол psi (курс виртуального пути),
      psi ведёт за вводом игрока (камера).
- В итоге политика гонится за виртуальным путём ровно так же, как в
  тренировке гналась за глобальным путём клипа - она в своём
  распределении, ничего переобучать не нужно.
- Бесшовность: при смене скилла qpos/qvel/act копируются из старого env
  в новый (без телепорта в кадр 0), а виртуальный путь начинается
  там, где стоит робот, с курсом, куда он смотрит.
"""

import math

import torch

MAX_RATE_WALK = 0.5      # рад/с, доворот на ходу (политика не любит резких)
MAX_RATE_STAND = 0.4     # рад/с, поворот на месте (медленнее)
PIN_SNAP_DIST = 0.7      # м, если путь убежал от робота - прибить к роботу
PIN_CARROT_MAX = 0.5     # м, зазор морковки (дальше политика OOD)
CARROT_WALK = 0.4        # м/с, скорость морковки по умолчанию (ходьба)
STEP_DT = 0.02           # шаг управления, с
STEP_VMIN = 0.30         # м/с, порог "клип реально шагает" (0.15-0.3 - это
                         # переступание, которое политика игнорирует - E2)
LOCO_START_VMIN = 0.25   # м/с, запасной порог (медленные клипы)
PRE_ROLL_FRAMES = 150    # кадров (3 с) до первого шага: бленд 1 с в стойке
                         # + 2 с разгона


def _yaw_of_quat(q):
    """Yaw (рад) из кватерниона (w,x,y,z), форма (..., 4)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(a):
    """Нормализация угла в (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def install():
    """Monkey-patch MotionReferenceProducer: стирринг + зацикливание клипа.

    Идемпотентно: повторный вызов ничего не меняет.
    """
    import mujoco_playground.tasks.whole_body.tracking.motion_command as mc

    if getattr(mc.MotionReferenceProducer, "_steer_patched", False):
        return

    orig_produce = mc.MotionReferenceProducer.produce_features
    orig_advance = mc.MotionReferenceProducer.advance_time_steps
    pg = mc.pg_math

    def produce_steered(self, required_features):
        feats = orig_produce(self, required_features)
        st = getattr(self, "_steer", None)
        if st is None or not st.get("on", False):
            return feats
        # кроссфейд на стыке зацикливания: ref прыгает из конца клипа в
        # начало - сглаживаем от ПОСЛЕДНЕЙ отданной команды к новой
        t_now = int(self.time_steps[0].item())
        prev_t = st.get("prev_t")
        if prev_t is not None and t_now < prev_t:      # обёртка клипа
            st["blend_from"] = st.get("prev_jp")
            st["blend_t0"] = t_now
            st["blend_len"] = 30
        st["prev_t"] = t_now
        # плавный вход: команду суставов съезжаем от позы робота к референсу
        bf = st.get("blend_from")
        if bf is not None:
            k = (t_now - st.get("blend_t0", t_now)) / \
                max(st.get("blend_len", 1), 1)
            if 0.0 <= k < 1.0:
                a = max(0.0, min(1.0, k))
                jp = feats["motion/joint_pos_ref"]
                feats["motion/joint_pos_ref"] = \
                    jp * a + bf.to(jp) * (1.0 - a)
                feats["motion/joint_vel_ref"] = \
                    feats["motion/joint_vel_ref"] * a
        st["prev_jp"] = feats["motion/joint_pos_ref"][0].detach().clone()
        bt = self._body_tracking_name
        t = torch.clamp(self.time_steps, 0, self.motion.time_step_total - 1)
        ref_pos = self.motion.body_pos_w[t]           # (B,N,3)
        ref_quat = self.motion.body_quat_w[t]         # (B,N,4)
        ref_lin = self.motion.body_lin_vel_w[t]       # (B,N,3)
        ref_ang = self.motion.body_ang_vel_w[t]       # (B,N,3)
        B, N, _ = ref_pos.shape
        ai = self._anchor_index

        a_pos = ref_pos[:, ai]                        # (B,3) якорь кадра
        a_quat = ref_quat[:, ai]
        psi = torch.full((B,), float(st["psi"]), device=ref_pos.device)
        dpsi = (psi - _yaw_of_quat(a_quat)).unsqueeze(-1)      # (B,1)
        cq = pg.yaw_quat(torch.cat([
            torch.cos(dpsi / 2), torch.zeros_like(dpsi),
            torch.zeros_like(dpsi), torch.sin(dpsi / 2)], dim=-1)).unsqueeze(1)
        cq_n = cq.expand(-1, N, -1)                   # (B,N,4)

        # поворот всех тел вокруг якоря кадра + перенос якоря в p_pin (xy)
        rel = ref_pos - a_pos.unsqueeze(1)
        rot_rel = pg.quat_rotate(cq_n, rel)
        pin = torch.tensor(st["p_pin"], dtype=ref_pos.dtype,
                           device=ref_pos.device)     # (2,)
        new_xy = rot_rel[..., :2] + pin
        new_pos = torch.cat([new_xy, ref_pos[..., 2:3]], dim=-1)
        new_quat = pg.quat_mul(cq_n, ref_quat)
        new_lin = pg.quat_rotate(cq_n, ref_lin)
        new_ang = pg.quat_rotate(cq_n, ref_ang)
        new_a_pos = torch.cat([pin.unsqueeze(0).expand(B, 2),
                               a_pos[..., 2:3]], dim=-1)
        new_a_quat = pg.quat_mul(cq[:, 0], a_quat)    # (B,4)

        feats["motion/body_pos_ref"] = new_pos
        feats["motion/body_quat_ref"] = new_quat
        feats["motion/body_lin_vel_ref"] = new_lin
        feats["motion/body_ang_vel_ref"] = new_ang
        feats["motion/anchor_pos_ref"] = new_a_pos
        feats["motion/anchor_quat_ref"] = new_a_quat
        feats["motion/anchor_lin_vel_ref"] = new_lin[:, ai]
        feats["motion/anchor_ang_vel_ref"] = new_ang[:, ai]

        # относительные кадры (для наград/метрик) пересчитать согласованно
        bt_pos = required_features[f"{bt}/body_pos_w"].float()
        bt_quat = required_features[f"{bt}/body_quat_w"].float()
        rel_pos, rel_quat = mc._compute_relative_body_poses(
            new_a_pos, new_a_quat,
            bt_pos[:, self._anchor_index], bt_quat[:, self._anchor_index],
            new_pos, new_quat)
        feats["motion/body_pos_relative"] = rel_pos
        feats["motion/body_quat_relative"] = rel_quat
        return feats

    def advance_looped(self):
        """+1 кадр; в play-режиме клип зацикливается (иначе замерзает)."""
        self.time_steps += 1
        if self._sampling_mode == "start":
            self.time_steps %= self.motion.time_step_total
            return torch.zeros_like(self.time_steps, dtype=torch.bool)
        return orig_advance(self)

    mc.MotionReferenceProducer.produce_features = produce_steered
    mc.MotionReferenceProducer.advance_time_steps = advance_looped
    mc.MotionReferenceProducer._steer_patched = True


def attach(venv):
    """Создать steer-состояние у producer скилла (выкл. по умолчанию)."""
    prod = venv.unwrapped.task_layer.get_component_by_name("motion")
    prod._steer = {"on": False, "psi": 0.0, "p_pin": [0.0, 0.0]}
    return prod


def update(prod, psi_target, max_rate, dt=STEP_DT):
    """Плавный доворот курса виртуального пути к цели (рад)."""
    st = prod._steer
    d = _wrap(float(psi_target) - st["psi"])
    step = max(-max_rate * dt, min(max_rate * dt, d))
    st["psi"] = _wrap(st["psi"] + step)


def turn_in_place(prod, rate, dt=STEP_DT):
    """Непрерывный поворот на месте: psi += rate*dt (знак = направление)."""
    st = prod._steer
    st["psi"] = _wrap(st["psi"] + rate * dt)


def advance_pin(prod, dt=STEP_DT, speed=None, robot_xy=None):
    """Вести виртуальный путь морковкой.

    Политика в тренировке догоняла якорь с рассогласованием - кормим
    её якорем на ФИКСИРОВАННОЙ скорости: pin уходит вперёд по курсу
    psi, зазор ограничен PIN_CARROT_MAX (дальше политика вне своего
    распределения). На бленде и при повороте на месте pin стоит.
    robot_xy нужен для ограничения зазора.
    """
    st = prod._steer
    if not st.get("on", False):
        return
    if speed is None:
        speed = CARROT_WALK
    bf = st.get("blend_from")
    if bf is not None:
        t = int(prod.time_steps[0].item())
        k = (t - st.get("blend_t0", t)) / max(st.get("blend_len", 1), 1)
        if 0.0 <= k < 1.0:
            return  # бленд идёт - путь ждёт робота
    if robot_xy is not None:
        dx = st["p_pin"][0] - float(robot_xy[0])
        dy = st["p_pin"][1] - float(robot_xy[1])
        if math.hypot(dx, dy) >= PIN_CARROT_MAX:
            return  # морковка достаточно далеко - ждём робота
    st["p_pin"][0] += math.cos(st["psi"]) * speed * dt
    st["p_pin"][1] += math.sin(st["psi"]) * speed * dt


def begin(venv, robot_xy, robot_yaw, t_start=None, blend_steps=50):
    """Запустить стирринг: путь из точки, где стоит робот, курс = robot_yaw.

    t_start: кадр клипа, с которого начать (фазовый подбор для
    локомоции - пропускаем стоячую раскачку в начале клипов LAFAN).
    blend_steps: плавный вход - команду суставов начинаем с ТЕКУЩЕЙ
    позы робота и за blend_steps шагов съезжаем к референсу (резкий
    скачок позы посреди шага политика не отрабатывает - замирает).
    """
    prod = venv.unwrapped.task_layer.get_component_by_name("motion")
    st = prod._steer
    st["on"] = True
    st["psi"] = float(robot_yaw)
    st["p_pin"] = [float(robot_xy[0]), float(robot_xy[1])]
    # якорь команды: поза робота в момент входа (для плавного бленда)
    nj = int(prod.motion.joint_pos.shape[1])
    q = venv.unwrapped.device_layer.sim.data.qpos
    st["blend_from"] = q[0, 7:7 + nj].detach().clone()
    st["blend_t0"] = int(prod.time_steps[0].item()) if t_start is None \
        else int(t_start)
    st["blend_len"] = int(blend_steps)
    if t_start is not None:
        prod.time_steps[:] = int(t_start)


def pick_start_frame(prod, v_min=STEP_VMIN, hold=50):
    """Кадр старта: за PRE_ROLL_FRAMES до первого устойчивого шага клипа.

    Эксперименты (E1-E6): робот трогается, только если успел "прожить"
    стоячую раскачку клипа перед первым шагом. Вход прямо в шаг (t=S)
    замораживает его (E1: 0.11 м), длинный бленд не спасает (E6: падение).
    Ставим плейхед за 3 с до шага: бленд (1 с) проходит в стойке, дальше
    робот качается вместе с клипом и трогается вместе с ним (E4: 0.99 м,
    E5: 0.82 м). Шаговых участков нет вовсе - играем с нуля (E2: стоячая
    фаза сама выводит в шаг).
    """
    pos = prod.motion.body_pos_w                                 # (T,N,3)
    ai = prod._anchor_index
    v = (pos[1:, ai, :2] - pos[:-1, ai, :2]).norm(dim=-1) / STEP_DT  # (T-1,)
    for thr in (v_min, LOCO_START_VMIN):
        moving = (v > thr).to(torch.float32)
        cs = torch.cumsum(moving, dim=0)
        pad = torch.zeros(1, dtype=cs.dtype)
        win = cs[hold - 1:] - torch.cat([pad, cs[:cs.numel() - hold]])
        hit = (win >= hold).nonzero()
        if hit.numel():
            return max(int(hit[0]) - PRE_ROLL_FRAMES, 0)
    return 0


def guard(prod, robot_xy):
    """Прибить виртуальный путь к роботу, если он отъехал (страховка OOD)."""
    st = prod._steer
    if not st.get("on", False):
        return
    dx = st["p_pin"][0] - float(robot_xy[0])
    dy = st["p_pin"][1] - float(robot_xy[1])
    if math.hypot(dx, dy) > PIN_SNAP_DIST:
        st["p_pin"] = [float(robot_xy[0]), float(robot_xy[1])]


def transplant(new_venv, old_venv):
    """Бесшовный переход: скопировать состояние робота из старого env.

    ВАЖНО: вызвать ПОСЛЕ new_venv.reset() (он взводит буферы и
    playhead=0) и ДО первого шага нового скилла.
    """
    if old_venv is None:
        return
    s_new = new_venv.unwrapped.device_layer.sim.data
    s_old = old_venv.unwrapped.device_layer.sim.data
    with torch.no_grad():
        s_new.qpos[:] = s_old.qpos.clone()
        s_new.qvel[:] = s_old.qvel.clone()
        for f in ("act", "ctrl"):
            a_new, a_old = getattr(s_new, f, None), getattr(s_old, f, None)
            if a_new is not None and a_old is not None and \
                    a_new.shape == a_old.shape:
                a_new[:] = a_old.clone()


def refresh_obs(venv):
    """Пересчитать obs из ТЕКУЩЕГО состояния сима (без шага физики).

    Нужен после transplant: obs от reset() описывают позу ДО пересадки,
    а политика должна увидеть реальную позу робота, иначе её первое
    действие будет рассчитано на старое состояние.
    """
    from tensordict import TensorDict
    env = venv.unwrapped
    measurements = env.device_layer.get_measurements()
    features = env.task_layer.compute_all_features(measurements)
    observations = env._compute_observations_from_features(features)
    return TensorDict(dict(observations), batch_size=[env.nworld])
