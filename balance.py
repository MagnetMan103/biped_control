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

# Torque Control Gains (you'll need to re-tune these!)
balance_gain = 10.0        # Proportional gain on roll angle
velocity_gain = 5.0        # Derivative gain on roll rate
wheel_damping = 0.1        # Damping on wheel velocity to prevent runaway

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


def get_wheel_velocities(model, data):
    """
    Get the current angular velocities of the wheels.
    Returns (wheel_1_vel, wheel_2_vel) in rad/s.
    """
    # Find the joint IDs for the wheels
    wheel_1_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wheel_1")
    wheel_2_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wheel_2")
    
    # Get velocities from qvel using joint addresses
    wheel_1_vel = data.qvel[model.jnt_dofadr[wheel_1_joint]]
    wheel_2_vel = data.qvel[model.jnt_dofadr[wheel_2_joint]]
    
    return wheel_1_vel, wheel_2_vel


def inverted_pendulum_torque_controller(model, data):
    """
    Inverted pendulum controller using TORQUE control for roll stabilization.
    
    Physics:
    - The robot body is the pendulum (mass on top)
    - The wheels apply torque to accelerate the base
    - If tilting right (roll > 0), apply torque to accelerate base RIGHT
    - If tilting left (roll < 0), apply torque to accelerate base LEFT
    
    Control Law:
        torque = Kp * roll_angle + Kd * roll_rate - Kw * wheel_velocity
    
    The wheel_velocity term provides damping to prevent wheels from spinning forever.
    """
    # Get current roll angle
    roll = get_body_roll(data, body_id)
    
    # Get body angular velocity (roll rate)
    body_angular_velocity = data.cvel[body_id][3:6]
    roll_rate = body_angular_velocity[0]
    
    # Get current wheel velocities for damping
    wheel_1_vel, wheel_2_vel = get_wheel_velocities(model, data)
    
    # === TORQUE CONTROL LAW ===
    # Base torque from pendulum state
    base_torque = balance_gain * roll + velocity_gain * roll_rate
    
    # Apply torque to wheels (with individual damping)
    # Note: signs may need adjustment based on your wheel/joint orientation
    torque_1 = -base_torque 
    torque_2 = base_torque
    
    data.ctrl[wheel_1_id] = torque_1
    data.ctrl[wheel_2_id] = torque_2
    
    # === LEG CONTROL ===
    # Keep legs neutral
    data.ctrl[femur_1_id] = 0.0
    data.ctrl[tibia_1_id] = 0.0
    data.ctrl[femur_2_id] = 0.0
    data.ctrl[tibia_2_id] = 0.0


def input_listener():
    """Listen for keyboard input in the terminal to toggle pause."""
    global is_paused
    while True:
        input()
        is_paused = not is_paused
        status = "PAUSED" if is_paused else "RUNNING"
        print(f"\n>>> Simulation {status} <<<\n")


# Start the input listener thread
listener_thread = threading.Thread(target=input_listener, daemon=True)
listener_thread.start()

# Print initial status
print("=" * 70)
print("INVERTED PENDULUM ROLL CONTROL (TORQUE MODE)")
print("Physics: Roll is an inverted pendulum - wheel torques accelerate base")
print("SIMULATION STARTING PAUSED")
print("Press ENTER in this terminal to pause/unpause")
print(f"Model timestep: {model.opt.timestep} seconds")
print(f"Control Gains: Kp={balance_gain}, Kd={velocity_gain}, Kw={wheel_damping}")
print("=" * 70)
print()

# Run simulation with viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        
        if not is_paused:
            # Apply torque control
            inverted_pendulum_torque_controller(model, data)
            
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Print status periodically
            if timestep_counter % print_every_n_steps == 0:
                roll = get_body_roll(data, body_id)
                roll_rate = data.cvel[body_id][0]
                pos = data.xpos[body_id]
                torque = data.ctrl[wheel_1_id]
                w1_vel, w2_vel = get_wheel_velocities(model, data)
                print(f"Step {timestep_counter}: "
                      f"Roll = {np.degrees(roll):6.2f}°, "
                      f"Roll Rate = {np.degrees(roll_rate):6.2f}°/s, "
                      f"Height = {pos[2]:.3f}m, "
                      f"Torque = {torque:6.2f}, "
                      f"Wheel Vel = {w1_vel:5.1f}")
            
            timestep_counter += 1
        
        viewer.sync()
        
        elapsed = time.time() - step_start
        sleep_time = model.opt.timestep - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
