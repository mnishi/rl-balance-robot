import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("balance.xml")
data = mujoco.MjData(model)

roll_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, "roll_motor"
)

with mujoco.viewer.launch_passive(model, data) as viewer:

    start = time.time()

    while viewer.is_running():
        
        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(model.opt.timestep)

