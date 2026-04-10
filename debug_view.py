import os
import time
import mujoco
import mujoco.viewer

# Force EGL to avoid the Xlib/NV-GLX errors on your RTX 2060
os.environ["MUJOCO_GL"] = "egl"

def main():
    # Load your MJCF
    xml_path = "model/joystick_env.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Pause flag to allow manual fiddling
    paused = False

    def key_callback(keycode):
        nonlocal paused
        if keycode == 32:  # Spacebar to Pause/Resume
            paused = not paused
            print(f"\n--- PHYSICS {'PAUSED' if paused else 'RESUMED'} ---")
        if keycode == 259: # Backspace to Reset robot to home position
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            print("\n--- ROBOT RESET ---")

    # Launch the passive viewer
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        print("\n" + "="*50)
        print("MANUAL FIDDLE MODE")
        print(" - Mouse: Ctrl + Left Click & Drag to move by hand.")
        print(" - Space: Pause physics to stop motors from fighting you.")
        print(" - Backspace: Reset robot to starting pose.")
        print(" - UI: Use 'Control' sliders instead of 'Joint' sliders.")
        print("="*50 + "\n")

        while viewer.is_running():
            step_start = time.time()

            # Advance physics only if not paused
            if not paused:
                mujoco.mj_step(model, data)

            # Print the tool tip position to CMD
            try:
                # Named access is easier than the 'Watch' index box
                pos = data.sensor('tool_position').data
                print(f"\rTIP -> X: {pos[0]:.4f} | Y: {pos[1]:.4f} | Z: {pos[2]:.4f} | Status: {'PAUSED' if paused else 'LIVE'}", end="")
            except KeyError:
                print("\rError: 'tool_position' sensor name not found in XML!", end="")

            # Sync the viewer to redraw your manual changes
            viewer.sync()

            # Match real-time speed
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()