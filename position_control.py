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

# Control Gains
balance_gain = 10.0           # Proportional gain on roll error

velocity_gain = 5.0           # Derivative gain on roll rate
position_gain = 2.0           # Gain for converting position error to desired roll angle
position_velocity_gain = 1.5  # Damping gain on x-axis velocity
max_roll_setpoint = 0.5       # Maximum roll angle setpoint (radians, ~29 degrees)

# Position Control
target_y_position = 0.0       # Target y-coordinate to move to
enable_position_control = False  # Toggle position control on/off

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


def inverted_pendulum_position_controller(model, data):
    """
    Inverted pendulum controller with optional position control.
    
    Mode 1 - Balance Only (enable_position_control = False):
        - Tries to maintain roll = 0 (upright)
        - Robot stays roughly in place
    
    Mode 2 - Position Control (enable_position_control = True):
        - Calculates desired roll angle based on position error
        - Leans forward/backward to move toward target y-position
        - Like a Segway: lean forward to go forward
    
    Control Law:
        desired_roll = -position_gain * (current_y - target_y) - position_velocity_gain * y_velocity
        roll_error = roll - desired_roll
        torque = balance_gain * roll_error + velocity_gain * roll_rate
    """
    # Get current state
    roll = get_body_roll(data, body_id)
    current_y = data.xpos[body_id][1]  # Y-axis position
    roll_rate = data.cvel[body_id][0]
    
    # Get y-axis velocity (forward/backward velocity)
    y_velocity = data.cvel[body_id][4]  # Linear velocity in y direction
    
    # Calculate desired roll angle based on position control
    if enable_position_control:
        # Position error: how far are we from target?
        position_error = current_y - target_y_position
        
        # Convert position error to desired roll angle with velocity damping
        # Positive sign: if we're behind target (error < 0), lean forward (positive roll)
        desired_roll = position_gain * position_error + position_velocity_gain * y_velocity
        
        # Limit desired roll to prevent excessive leaning
        desired_roll = np.clip(desired_roll, -max_roll_setpoint, max_roll_setpoint)
    else:
        # Just try to stay upright
        desired_roll = 0.0
    
    # Calculate roll error (how far from desired angle)
    roll_error = roll - desired_roll
    
    # Control law: torque proportional to angle error and angular velocity
    base_torque = balance_gain * roll_error + velocity_gain * roll_rate
    
    # Apply torque to wheels
    torque_1 = -base_torque 
    torque_2 = base_torque
    
    data.ctrl[wheel_1_id] = torque_1
    data.ctrl[wheel_2_id] = torque_2
    
    # Keep legs neutral
    data.ctrl[femur_1_id] = 0.0
    data.ctrl[tibia_1_id] = 0.0
    data.ctrl[femur_2_id] = 0.0
    data.ctrl[tibia_2_id] = 0.0


def input_listener():
    """
    Listen for keyboard input to control simulation.
    Commands:
        ENTER - Toggle pause/unpause
        'p' - Toggle position control on/off
        't <value>' - Set target y position (e.g., 't 2.0')
    """
    global is_paused, enable_position_control, target_y_position
    
    while True:
        user_input = input().strip().lower()
        
        # Empty input (just ENTER) - toggle pause
        if user_input == '':
            is_paused = not is_paused
            status = "PAUSED" if is_paused else "RUNNING"
            print(f"\n>>> Simulation {status} <<<\n")
        
        # 'p' - toggle position control
        elif user_input == 'p':
            enable_position_control = not enable_position_control
            status = "ENABLED" if enable_position_control else "DISABLED"
            print(f"\n>>> Position Control {status} <<<")
            if enable_position_control:
                print(f"    Target Y: {target_y_position:.2f}m\n")
            else:
                print(f"    (Back to balance-only mode)\n")
        
        # 't <value>' - set target position
        elif user_input.startswith('t '):
            try:
                new_target = float(user_input.split()[1])
                target_y_position = new_target
                print(f"\n>>> Target Y Position set to: {target_y_position:.2f}m <<<")
                if not enable_position_control:
                    print("    (Position control is currently OFF - press 'p' to enable)\n")
                else:
                    print()
            except (ValueError, IndexError):
                print("\n>>> Invalid format. Use: t <number> (e.g., 't 2.0') <<<\n")
        
        else:
            print("\n>>> Commands: ENTER (pause), 'p' (toggle position control), 't <value>' (set target) <<<\n")


# Start the input listener thread
listener_thread = threading.Thread(target=input_listener, daemon=True)
listener_thread.start()

# Print initial status
print("=" * 80)
print("INVERTED PENDULUM WITH POSITION CONTROL")
print("=" * 80)
print("CONTROLS:")
print("  ENTER          - Pause/unpause simulation")
print("  p              - Toggle position control on/off")
print("  t <value>      - Set target y position (e.g., 't 2.0' or 't -1.5')")
print()
print("CURRENT SETTINGS:")
print(f"  Position Control: {'ENABLED' if enable_position_control else 'DISABLED'}")
print(f"  Target Y: {target_y_position:.2f}m")
print(f"  Balance Gain (Kp): {balance_gain}")
print(f"  Velocity Gain (Kd): {velocity_gain}")
print(f"  Position Gain: {position_gain}")
print(f"  Position Velocity Gain: {position_velocity_gain}")
print(f"  Max Roll Setpoint: {np.degrees(max_roll_setpoint):.1f}°")
print(f"  Timestep: {model.opt.timestep}s")
print()
print("SIMULATION STARTING PAUSED - Press ENTER to begin")
print("=" * 80)
print()

# Run simulation with viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        
        if not is_paused:
            # Apply controller
            inverted_pendulum_position_controller(model, data)
            
            # Step the simulation
            mujoco.mj_step(model, data)
            
            timestep_counter += 1
            
            # Print status periodically (only when running)
            if timestep_counter % print_every_n_steps == 0:
                roll = get_body_roll(data, body_id)
                roll_rate = data.cvel[body_id][0]
                current_y = data.xpos[body_id][1]
                pos_error = current_y - target_y_position
                torque = data.ctrl[wheel_1_id]
                w1_vel, w2_vel = get_wheel_velocities(model, data)
                
                mode = "POS CTRL" if enable_position_control else "BALANCE"
                print(f"[{mode}] Step {timestep_counter}: "
                      f"Y = {current_y:6.2f}m (err: {pos_error:+6.2f}m), "
                      f"Roll = {np.degrees(roll):6.2f}°, "
                      f"Torque = {torque:6.2f}")
        
        viewer.sync()
        
        # Real-time pacing
        elapsed = time.time() - step_start
        sleep_time = model.opt.timestep - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
