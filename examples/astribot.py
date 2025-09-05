from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "astribot" / "flattened_model.xml"


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    # mujoco.mj_saveLastXML("flattened_model.xml", model)
    
    configuration = mink.Configuration(model)
    hands = ["astribot_arm_left_tool_site"]#, "astribot_arm_right_tool_site"]

    tasks = [
        # torso_orientation_task := mink.FrameTask(
        #     frame_name="torso_base",
        #     frame_type="body",
        #     position_cost=0.2,
        #     orientation_cost=2.0,
        #     lm_damping=1.0,
        # ),
        posture_task:= mink.PostureTask(model, cost=1),
        # com_task:= mink.ComTask(cost=10.0),
    ]

    hand_tasks = []
    for hand in hands:
        task = mink.FrameTask(
            frame_name=hand,
            frame_type="site",
            position_cost=5.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        hand_tasks.append(task)
    tasks.extend(hand_tasks)

    com_mid = model.body("com_target").mocapid[0]
    hands_mid = [model.body(f"{hand}_target").mocapid[0] for hand in hands]

    model = configuration.model
    data = configuration.data
    solver = "daqp"

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        configuration.update_from_keyframe("home")
        # Initialize to the home keyframe.
        for hand in hands:
            mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")

        rate = RateLimiter(frequency=200.0, warn=False)
        posture_task.set_target_from_configuration(configuration)
        # torso_orientation_task.set_target_from_configuration(configuration)
        # data.mocap_pos[com_mid] = data.subtree_com[1]
        while viewer.is_running():
            # com_task.set_target(data.mocap_pos[com_mid])
            # Update task targets.
            
            for i, hand_task in enumerate(hand_tasks):
                hand_task.set_target(mink.SE3.from_mocap_id(data, hands_mid[i]))
                
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 1)


            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, 1e-1,
            )
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Note the below are optional: they are used to visualize the output of the
            # fromto sensor which is used by the collision avoidance constraint.
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_sensorPos(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
