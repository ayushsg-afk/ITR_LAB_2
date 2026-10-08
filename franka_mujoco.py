import mujoco
import mujoco.viewer
import numpy as np

# 1. Path to the Franka MJCF model file
MODEL_PATH = "robot_descriptions/franka/panda.xml"

# 2. Load model and simulation data
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# 3. Define target 7-DOF joint positions (in radians)
# Franka Panda has 7 arm joints: panda_joint1 through panda_joint7
q_target = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# Set joint angles in qpos (first 7 elements correspond to arm joints)
data.qpos[:7] = q_target

# 4. Perform Forward Kinematics
# mujoco.mj_kinematics computes positions and orientations of all bodies/sites
mujoco.mj_kinematics(model, data)

# Alternatively, mj_forward updates full physics state including collision/sensors:
# mujoco.mj_forward(model, data)

# 5. Retrieve End-Effector Position and Orientation
# You can query by body name (e.g., 'hand' or 'link7') or site name (e.g., 'attachment_site')
ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

ee_position = data.xpos[ee_body_id]        # [x, y, z] position in world frame
ee_rotation_mat = data.xmat[ee_body_id]    # 3x3 rotation matrix (flattened as 9x1 array)

print("--- Forward Kinematics Results ---")
print(f"Joint Angles (q): {q_target}")
print(f"End-Effector Position (x, y, z): {ee_position}")
print(f"End-Effector Rotation Matrix:\n{ee_rotation_mat.reshape(3, 3)}")

# 6. Launch interactive viewer (optional)
mujoco.viewer.launch(model, data)
