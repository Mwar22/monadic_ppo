import os
import time
import mujoco
import mujoco.viewer
import jax
import jax.numpy as jnp
from mujoco import mjx
from etils import epath

# Custom imports from your project files
from robot import create_rsd, create_step, get_goal
from networks import create_networks, load_params
from dataclassutils import RunningParameters
from config import MujocoSimConfig, RewardConfig, RangeConfig

# 1. Environment & Hardware Setup
# Forces the EGL rendering backend to avoid NV-GLX errors on Linux
os.environ["MUJOCO_GL"] = "egl" 
# Configure JAX memory allocation for your 8GB card
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"

jax.config.update("jax_enable_x64", False)

def main():
    # 2. Setup Models and Config
    range_cfg = RangeConfig.init(
        numberof_goals=100,
        position_min_values = jnp.array([-0.6, -0.6, 1.0]),
        position_max_values = jnp.array([0.6, 0.6, 1.3]),
        position_velocities_min_values = jnp.array([0, 0, 0]),
        position_velocities_max_values = jnp.array([0.1, 0.1, 0.1]),
        orientation_min_values = jnp.array([-2, -2, -2]),
        orientation_max_values = jnp.array([2, 2, 2])
    )

    rsd = create_rsd(
        epath.Path("model/joystick_env.xml"),
        epath.Path("model"),
        epath.Path("model/meshes"),
        MujocoSimConfig(),
        RewardConfig(),
        range_cfg,
        ["junta1", "junta2", "junta3", "junta4","junta5", "junta6"]
    )
    
    # Shadows for the viewer (CPU) and physics (GPU/MJX)
    m_cpu = rsd.mj_model
    d_cpu = mujoco.MjData(m_cpu)
    
    # 3. Load the Trained Brain
    rng = jax.random.PRNGKey(0)
    rng, net_settings, net_params = create_networks(rng, obs_size=34, action_size=6)
    trained_params = load_params(net_params, "trained_params.msgpack")

    # Load Normalization Stats (Critical for correct sensor scaling)
    runpar = RunningParameters.init((34,))
    
    # 4. Initialize the Step Function
    # Reuses your modular training logic directly on the GPU
    step_fn = jax.jit(create_step(net_settings, trained_params, rsd))

    # Initialize State for the Step Pipeline
    progress = 1.0 # Evaluate at maximum difficulty
    rng, goal = get_goal(rsd, progress, rng)
    
    # Move initial state to GPU
    mx_data = mjx.put_data(m_cpu, d_cpu)
    

    def starting_state():
        return {
            "mjx_data": mx_data,
            "goal": goal,
            "rng": rng,
            "obs": jnp.zeros((34,)),
            "success_count": 0,
            "step": 0,
            "action": jnp.zeros((6,)),
            "err": 1.0
        }

    # 5. The Correct Viewer Loop
    frame_dt = m_cpu.opt.timestep * rsd.enviroment_config.n_substeps
    current_state = starting_state()
    with mujoco.viewer.launch_passive(m_cpu, d_cpu) as viewer:
        while viewer.is_running():
            start_time = time.time()

            # A. Advance GPU Physics
            current_state, pdata = step_fn(progress, current_state, runpar)
            
            # B. Sync GPU -> CPU
            # Must use the CPU model (m_cpu) to fetch from MJX
            d_gpu_to_cpu = mjx.get_data(m_cpu, current_state["mjx_data"])
            jax.block_until_ready(current_state)

            # C. Update Viewer Memory
            with viewer.lock():
                # Manual array copy since your version lacks mj_copyData
                d_cpu.qpos[:] = d_gpu_to_cpu.qpos
                d_cpu.qvel[:] = d_gpu_to_cpu.qvel
                # Update 3D visual positions (xpos) for the renderer
                mujoco.mj_forward(m_cpu, d_cpu)

            # D. Sync Viewer GUI
            viewer.sync()
            
            # E. Handle Goal Resets
            if pdata["done"]:
                print(f"Goal Reached! Error: {current_state['err']:.4f}")
                rng, new_goal = get_goal(rsd, progress, current_state["rng"])
                current_state["goal"] = new_goal
                current_state["step"] = 0
                current_state["rng"] = rng
                current_state = starting_state()

            # F. Maintain Real-Time Speed
            elapsed = time.time() - start_time
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)

if __name__ == "__main__":
    main()