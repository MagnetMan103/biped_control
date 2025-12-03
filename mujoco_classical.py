import mujoco
import mujoco.viewer
import numpy as np
import threading
import time

# Load your model
model = mujoco.MjModel.from_xml_path("biped/scene.xml")
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

# Get IDs for body and actuators
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body")

# Get actuator IDs
femur_1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "femur_1")
tibia_1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tibia_1")
wheel_1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_1")
femur_2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "femur_2")
tibia_2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tibia_2")
wheel_2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_2")

# Inverted Pendulum Control Gains (tune these!)
balance_gain = 13          # Proportional gain on roll angle
velocity_gain = 0         # Derivative gain on roll rate (INCREASED!)

# Simulation control
is_paused = True
timestep_counter = 0
print_every_n_steps = 100

def get_body_roll(data, body_id):
    """
    Get the roll angle (side-to-side tilt) of the body.
    Returns angle in radians.
    """
    quat = data.xquat[body_id]
    
    # Convert quaternion to rotation matrix
    rotation_matrix = np.zeros(9)
    mujoco.mju_quat2Mat(rotation_matrix, quat)
    
    # Extract roll (side-to-side tilt around x-axis)
    roll = np.arctan2(rotation_matrix[7], rotation_matrix[8])
    
    return roll

def inverted_pendulum_roll_controller(model, data):
    """
    Inverted pendulum controller for roll stabilization.
    
    Physics:
    - The robot body is the pendulum (mass on top)
    - The wheels are the moving base
    - If tilting right (roll > 0), accelerate base RIGHT to catch the fall
    - If tilting left (roll < 0), accelerate base LEFT to catch the fall
    
    This is identical to a Segway balancing in the roll axis.
    """
    # Get current roll angle and angular velocity
    roll = get_body_roll(data, body_id)
    
    # Get body angular velocity (roll rate)
    body_angular_velocity = data.cvel[body_id][3:6]
    roll_rate = body_angular_velocity[0]
    
    # === INVERTED PENDULUM CONTROL ===
    # Classic control law: wheel_velocity = Kp * angle + Kd * angle_rate
    # If tilting right (roll > 0), wheels must spin to move robot right
    # If tilting left (roll < 0), wheels must spin to move robot left
    desired_wheel_velocity = balance_gain * roll + velocity_gain * roll_rate
    # Apply same velocity to both wheels (coordinated motion to move laterally)
    data.ctrl[wheel_1_id] = -desired_wheel_velocity
    data.ctrl[wheel_2_id] = desired_wheel_velocity
    
    # === LEG CONTROL ===
    # Keep legs mostly neutral/straight to maintain consistent pendulum dynamics
    # Small adjustments can help, but the primary control is through wheels
    data.ctrl[femur_1_id] = 0.0
    data.ctrl[tibia_1_id] = 0.0
    data.ctrl[femur_2_id] = 0.0
    data.ctrl[tibia_2_id] = 0.0

def input_listener():
    """
    Listen for keyboard input in the terminal to toggle pause.
    """
    global is_paused
    while True:
        input()  # Wait for Enter key
        is_paused = not is_paused
        status = "PAUSED" if is_paused else "RUNNING"
        print(f"\n>>> Simulation {status} <<<\n")

# Start the input listener thread
listener_thread = threading.Thread(target=input_listener, daemon=True)
listener_thread.start()

# Print initial status
print("=" * 70)
print("INVERTED PENDULUM ROLL CONTROL")
print("Physics: Roll is an inverted pendulum - wheels move to catch the fall")
print("SIMULATION STARTING PAUSED")
print("Press ENTER in this terminal to pause/unpause")
print(f"Model timestep: {model.opt.timestep} seconds")
print(f"Control Gains: Kp={balance_gain}, Kd={velocity_gain}")
print("=" * 70)
print()

# Run simulation with viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Get the current time for real-time simulation
        step_start = time.time()
        
        # Only step the simulation if not paused
        if not is_paused:
            # Apply inverted pendulum control
            inverted_pendulum_roll_controller(model, data)
            
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Print status periodically
            if timestep_counter % print_every_n_steps == 0:
                roll = get_body_roll(data, body_id)
                roll_rate = data.cvel[body_id][0]
                pos = data.xpos[body_id]
                wheel_vel = data.ctrl[wheel_1_id]
                print(f"Step {timestep_counter}: "
                      f"Roll = {np.degrees(roll):6.2f}°, "
                      f"Roll Rate = {np.degrees(roll_rate):6.2f}°/s, "
                      f"Height = {pos[2]:.3f}m, "
                      f"Wheel Vel = {wheel_vel:6.2f}")
            
            timestep_counter += 1
        
        # Sync viewer (this still updates the display even when paused)
        viewer.sync()
        
        # Sleep to maintain real-time speed
        elapsed = time.time() - step_start
        sleep_time = model.opt.timestep - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
