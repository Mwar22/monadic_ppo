import os
# 1. Environment & Hardware Setup
# Forces JAX to use the system allocator (No more 'sticky' VRAM)
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# Forces the EGL rendering backend
os.environ["MUJOCO_GL"] = "egl" 

# Since we are using 'platform', we don't strictly need these, 
# but keeping them doesn't hurt.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"

import time
import mujoco
import mujoco.viewer
import jax
import jax.numpy as jnp
from mujoco import mjx
from etils import epath

# Custom imports from your project files
from robot import RobotSharedData, create_step, get_goal
from networks import create_networks, load_params
from dataclassutils import RunningParameters
from config import MujocoSimConfig, RewardConfig, RangeConfig

def main():
    range_cfg = RangeConfig.init(
    300,
    pos_min = jnp.array([-0.468, -0.468, 0]),
    pos_max = jnp.array([0.468, 0.468, 0.664]),
    posvel_min = jnp.array([1e-2, 1e-2, 1e-2]),
    posvel_max = jnp.array([0.1, 0.1, 0.1]),
    ori_min = jnp.array([-3.14, -3.14, -3.14]),
    ori_max = jnp.array([3.14, 3.14, 3.14]),
)

    robot_shared_data = RobotSharedData.init(
        epath.Path("model/joystick_env.xml"),
        epath.Path("model"),
        epath.Path("model/meshes"),
        MujocoSimConfig(),
        RewardConfig(),
        range_cfg,
        robot_name="thor"
    )

    if robot_shared_data.value is None:
        raise RuntimeError("RSD is None")
    
    rsd = robot_shared_data.value
    
    m_cpu = rsd.mj_model
    d_cpu = mujoco.MjData(m_cpu)
    
    # 3. Load Trained Model
    rng = jax.random.PRNGKey(0)
    rng, net_settings, net_params = create_networks(rng, obs_size=34, action_size=6)
    trained_params = load_params(net_params, "trained_params.msgpack")
    runpar = RunningParameters.init((34,))
    step_fn = jax.jit(create_step(net_settings, trained_params, rsd))

    # 4. Keyboard Control State
    # Starting target positions and Euler rotations
    target_pos = [0.2, 0.0, 0.4]  # X, Y, Z
    target_rot = [0.0, 0.0, 0.0]  # Roll, Pitch, Yaw
    ctrl_pressed = False
    step_size = 0.02 # Meters/Radians per key press
    
    def key_callback(keycode):
        nonlocal target_pos, target_rot, ctrl_pressed
        
        # 341/345 are Left/Right Control keys in GLFW
        if keycode in [341, 345]: ctrl_pressed = True
        
        # XY PLANE: Arrow Keys
        if keycode == 265: # Up Arrow
            if ctrl_pressed: target_rot[1] += step_size # Pitch
            else: target_pos[0] += step_size            # X+
        if keycode == 264: # Down Arrow
            if ctrl_pressed: target_rot[1] -= step_size 
            else: target_pos[0] -= step_size            # X-
        if keycode == 263: # Left Arrow
            if ctrl_pressed: target_rot[2] += step_size # Yaw
            else: target_pos[1] += step_size            # Y+
        if keycode == 262: # Right Arrow
            if ctrl_pressed: target_rot[2] -= step_size 
            else: target_pos[1] -= step_size            # Y-
            
        # Z-AXIS: Page Up / Page Down
        if keycode == 266: # Page Up
            if ctrl_pressed: target_rot[0] += step_size # Roll
            else: target_pos[2] += step_size            # Z+
        if keycode == 267: # Page Down
            if ctrl_pressed: target_rot[0] -= step_size 
            else: target_pos[2] -= step_size            # Z-

    # Initial Goal setup
    progress = 1.0
    rng, goal = get_goal(rsd.range_config, progress, rng)
    mx_data = mjx.put_data(m_cpu, d_cpu)
    
    current_state = {
        "mjx_data": mx_data,
        "goal": goal,
        "rng": rng,
        "obs": jnp.zeros((34,)),
        "success_count": 0,
        "step": 0,
        "last_action": jnp.zeros((6,)),
        "err": 1.0
    }



    # 5. Loop with manual goal updates
    slow_motion_factor = 1.0 # Real-time for better control feel
    frame_dt = (m_cpu.opt.timestep * rsd.enviroment_config.n_substeps) * slow_motion_factor
    
    with mujoco.viewer.launch_passive(m_cpu, d_cpu, key_callback=key_callback) as viewer:
        print("\n" + "="*30)
        print("KEYBOARD CONTROL ACTIVE")
        print("WASD: Move X/Y | QE: Move Z")
        print("HOLD CTRL + WASDQE: Rotations")
        print("="*30 + "\n")

        while viewer.is_running():
            start_time = time.time()

            # A. Update the Goal structure to match what robot.py expects
            # We update the dictionary keys instead of replacing the whole thing with an array
            current_state["goal"] = {
                "goal_position_coordinates": jnp.array(target_pos),
                "goal_orientation_coordinates": jnp.array(target_rot),
                # Including velocities as zeros for manual target tracking
                "goal_position_velocities": jnp.array([0.0, 0.0, 0.0])
            }
            

            # B. Run the "Brain" (The PPO policy follows the goal)
            current_state, pdata = step_fn(progress, current_state, runpar)
            
            # C. Sync MJX -> Viewer
            d_gpu_to_cpu = mjx.get_data(m_cpu, current_state["mjx_data"])
            jax.block_until_ready(current_state)

            with viewer.lock():
                d_cpu.qpos[:] = d_gpu_to_cpu.qpos
                d_cpu.qvel[:] = d_gpu_to_cpu.qvel
                # Move the 'goal' site in the viewer so you see where you're pointing
                # Note: 'goal' is the site name in your joystick_env.xml
                try:
                    d_cpu.site('goal').xpos[:] = jnp.array(target_pos)
                except: pass
                
                mujoco.mj_forward(m_cpu, d_cpu)


            tip = d_gpu_to_cpu.sensordata[rsd.mj_model.sensor('tool_position').adr[0] : rsd.mj_model.sensor('tool_position').adr[0]+3]
            
            print(f"\rTARGET: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}] | "
                  f"ACTUAL: [{tip[0]:.2f}, {tip[1]:.2f}, {tip[2]:.2f}] | "
                  f"ERR: {current_state['err']:.4f}", end="")

            viewer.sync()
            
            # Reset modifiers at end of loop to detect release
            ctrl_pressed = False 

            # D. Speed Maintenance
            elapsed = time.time() - start_time
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)

if __name__ == "__main__":
    main()