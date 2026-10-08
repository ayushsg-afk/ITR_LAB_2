#!/usr/bin/env python3
"""
HEAL 6-DOF  -  you control every joint yourself (standalone file).

Two windows open:
  * MuJoCo viewer          : shows the HEAL arm
  * "HEAL joint control"   : 6 sliders (degrees) + live forward kinematics

Put this file next to the robot_descriptions/ folder and run:
    python heal_control.py
Needs Tk once:  sudo apt install -y python3-tk
"""
import argparse, os, time
import numpy as np
import mujoco

# find robot_descriptions/ whether this file sits next to it or inside scripts/
_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (_here, os.path.join(_here, ".."), os.getcwd()):
    if os.path.isdir(os.path.join(_cand, "robot_descriptions")):
        ROOT = os.path.abspath(_cand)
        break
else:
    raise SystemExit("robot_descriptions/ folder not found - put this file next to it")
RD = os.path.join(ROOT, "robot_descriptions")


NAME, DOF = "HEAL", 6
XML = os.path.join(RD, "single_arm_heal_effort_actuation_rs_mj_2.xml")
EE_BODY = "end_effector"
JOINTS = [f"joint_{i}" for i in range(1, 7)]
HOME = [0.0, 0.5, 0.5, 0.0, 0.5, 0.0]


def fix_model(model):
    # the XML gives the "arm" material alpha=0 -> arm is invisible; make it visible
    m = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "arm")
    if m >= 0:
        model.mat_rgba[m, 3] = 1.0

# ---- Product-of-exponentials FK (textbook / Modern Robotics form) ----
# T(q) = exp([S1]q1) exp([S2]q2) ... exp([S6]q6) M      (all in the world frame)
# S_i = (w_i, v_i) screw axes and M = end-effector pose at q = 0, taken from the XML tree
S_LIST = np.array([
    [ 0.000000000000000,  0.000000000000000,  1.000000000000000, -0.000000000000000, -0.000000000000000, -0.000000000000000],
    [ 0.000000000000000,  0.999999999991000,  0.000004242642374, -0.320799628765905, -0.000000000000000, -0.000000000000000],
    [ 0.000000000000000, -0.999999999991000, -0.000004242642374,  0.620799628765905,  0.000000000000000, -0.000000000000000],
    [ 0.999999365863770, -0.000796330089254,  0.000796323079707,  0.000494461428132,  0.620799235632756, -0.000125582013679],
    [ 0.872594648257166, -0.488444571945883,  0.000692798350739,  0.303372864210161,  0.541707215050757, -0.184164642763605],
    [-0.000403663017528,  0.000688555709723,  0.999999681473551, -0.000651723680921, -0.377693949559318,  0.000259800331502],
])
M_HOME = np.array([
    [ 0.488444893398024,  0.872594649977618,  0.000403663017528,  0.377492884057899],
    [ 0.872594649977618, -0.488444574871575, -0.000688555709723, -0.000308547451512],
    [-0.000403663017528,  0.000688555709723, -0.999999681473551,  0.498400235223334],
    [ 0.000000000000000,  0.000000000000000,  0.000000000000000,  1.000000000000000],
])


def _skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def _exp_screw(s, th):
    w, v = s[:3], s[3:]
    W = _skew(w)
    R = np.eye(3) + np.sin(th)*W + (1 - np.cos(th))*W @ W
    G = np.eye(3)*th + (1 - np.cos(th))*W + (th - np.sin(th))*W @ W
    return make_T(R, G @ v)


def dh_fk(q):
    T = np.eye(4)
    for s, th in zip(S_LIST, q):
        T = T @ _exp_screw(s, th)
    return T @ M_HOME


# ---------- SE(3) helpers ----------
def quat_to_R(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]])


def axis_angle_R(axis, th):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th)*K + (1 - np.cos(th))*K @ K


def make_T(R=None, p=None):
    T = np.eye(4)
    if R is not None: T[:3, :3] = R
    if p is not None: T[:3, 3] = p
    return T


def rot_err(R1, R2):
    """Geodesic angle between rotations (stable near 0, unlike arccos of the trace)."""
    Rr = R1.T @ R2
    v = np.array([Rr[2, 1] - Rr[1, 2], Rr[0, 2] - Rr[2, 0], Rr[1, 0] - Rr[0, 1]]) / 2
    return float(np.arctan2(np.linalg.norm(v), (np.trace(Rr) - 1) / 2))


def fmt(T):
    return np.array2string(T, precision=4, suppress_small=True)


# ---------- MuJoCo helpers ----------
def jid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def bid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def load():
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    fix_model(model)
    return model, data


def limits(model):
    lo = np.array([model.jnt_range[jid(model, n), 0] for n in JOINTS])
    hi = np.array([model.jnt_range[jid(model, n), 1] for n in JOINTS])
    return lo, hi


def qaddr(model):
    return [model.jnt_qposadr[jid(model, n)] for n in JOINTS]


# ---------- Own FK: product of link transforms ----------
def own_fk(model, q):
    """T_world_ee = prod Trans(body_pos) Rot(body_quat) Rot(joint_axis, q_i)"""
    chain, b = [], bid(model, EE_BODY)
    while b != 0:
        chain.append(b); b = model.body_parentid[b]
    qd = dict(zip(JOINTS, q))
    T = np.eye(4)
    for b in chain[::-1]:
        T = T @ make_T(quat_to_R(model.body_quat[b]), model.body_pos[b])
        for j in range(model.njnt):
            if model.jnt_bodyid[j] == b:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                if name in qd:
                    T = T @ make_T(axis_angle_R(model.jnt_axis[j], qd[name]))
    return T


def mujoco_fk(model, data, q):
    data.qpos[qaddr(model)] = q
    mujoco.mj_kinematics(model, data)
    b = bid(model, EE_BODY)
    return make_T(data.xmat[b].reshape(3, 3), data.xpos[b])



import mujoco.viewer

try:
    import tkinter as tk
except ImportError:
    raise SystemExit("Tkinter missing. Install it with:  sudo apt install -y python3-tk")

model, data = load()
lo, hi = limits(model)
addr = qaddr(model)
ee = bid(model, EE_BODY)
home = np.clip(np.array(HOME, float), lo, hi)

data.qpos[addr] = home
mujoco.mj_kinematics(model, data)

# ------------------------------------------------------------------ GUI
root = tk.Tk()
root.title("HEAL joint control")
vars_ = []
for i, name in enumerate(JOINTS):
    v = tk.DoubleVar(value=float(np.degrees(home[i])))
    vars_.append(v)
    tk.Scale(root, variable=v, from_=float(np.degrees(lo[i])), to=float(np.degrees(hi[i])),
             resolution=0.5, orient=tk.HORIZONTAL, length=420,
             label=f"{name}  (deg)").pack(padx=10, pady=2)


def set_pose(q_rad):
    for v, qi in zip(vars_, q_rad):
        v.set(float(np.degrees(qi)))


bar = tk.Frame(root)
bar.pack(pady=4)
tk.Button(bar, text="Home pose", command=lambda: set_pose(home)).pack(side=tk.LEFT, padx=6)
tk.Button(bar, text="All zero", command=lambda: set_pose(np.zeros(DOF))).pack(side=tk.LEFT, padx=6)

info = tk.Label(root, text="", font=("DejaVu Sans Mono", 10), justify=tk.LEFT)
info.pack(padx=10, pady=8)

# ------------------------------------------------------------------ viewer + loop
viewer = mujoco.viewer.launch_passive(model, data)


def tick():
    if not viewer.is_running():          # MuJoCo window closed -> quit everything
        root.destroy()
        return
    q = np.radians([v.get() for v in vars_])
    with viewer.lock():
        data.qpos[addr] = q
        mujoco.mj_kinematics(model, data)    # kinematics only: arm holds the pose
        p_mj = np.array(data.xpos[ee])
    viewer.sync()

    T = own_fk(model, q)
    err = np.linalg.norm(T[:3, 3] - p_mj)
    mat = "\n".join("  [" + "  ".join(f"{x:+.4f}" for x in row) + "]" for row in T)
    info.config(text=f"T_world_ee (own FK):\n{mat}\n\n"
                     f"ee xyz = {np.round(T[:3, 3], 4)} m\n"
                     f"|own FK - MuJoCo| = {err:.1e} m")
    root.after(20, tick)


root.after(20, tick)
root.protocol("WM_DELETE_WINDOW", lambda: (viewer.close(), root.destroy()))
root.mainloop()
