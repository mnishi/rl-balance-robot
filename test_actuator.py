import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("balance.xml")
data = mujoco.MjData(model)

pitch_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pitch_motor"
)

roll_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, "roll_motor"
)

with mujoco.viewer.launch_passive(model, data) as viewer:

    start = time.time()

    while viewer.is_running():
        elapsed = time.time() - start

        data.ctrl[pitch_id] = 0.0 if elapsed < 3.0 else 1.0 
        data.ctrl[roll_id] = 0.0 if elapsed < 3.0 else 1.0 

        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(model.opt.timestep)

