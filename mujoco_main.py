import mujoco
import mujoco.viewer
import numpy as np
import threading
import time
from scipy import linalg

# Load your model
model = mujoco.MjModel.from_xml_path("biped/scene.xml")
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

# === INCREASE FRICTION TO PREVENT WHEEL SLIP ===
for i in range(model.ngeom):
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    if geom_name and 'wheel' in geom_name.lower():
        model.geom_friction[i, 0] = 2.0
        model.geom_friction[i, 1] = 0.1
        model.geom_friction[i, 2] = 0.01
        print(f"Set high friction for {geom_name}")

for i in range(model.ngeom):
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    if geom_name and ('floor' in geom_name.lower() or 'ground' in geom_name.lower()):
        model.geom_friction[i, 0] = 2.0

model.opt.iterations = 100
model.opt.ls_iterations = 20

# Get IDs for body and actuators
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body")

# Get actuator IDs
femur_1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "femur_1")
tibia_1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tibia_1")
wheel_1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_1")
femur_2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "femur_2")
tibia_2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tibia_2")
wheel_2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_2")

# === EXTRACT PHYSICAL PARAMETERS FROM MODEL ===
print("=" * 80)
print("EXTRACTING PHYSICAL PARAMETERS FROM MODEL")
print("=" * 80)

# Get body mass
M_body = model.body_mass[body_id]
print(f"Body mass (M): {M_body:.4f} kg")

# Get body inertia (roll axis is x-axis, index 0 in diagonal)
body_inertia = model.body_inertia[body_id]
I_body = body_inertia[0]  # Roll inertia (about x-axis)
print(f"Body roll inertia (I): {I_body:.6f} kg*m^2")

# Estimate wheel radius and total wheel mass
wheel_radius = 0.05  # You may need to adjust this
M_wheels = 0.0
for i in range(model.nbody):
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if body_name and 'wheel' in body_name.lower():
        M_wheels += model.body_mass[i]
        print(f"Wheel body '{body_name}': {model.body_mass[i]:.4f} kg")

M_cart = M_wheels  # Effective cart mass (wheels)
print(f"Total wheel/cart mass (m): {M_cart:.4f} kg")
print(f"Estimated wheel radius (r): {wheel_radius:.4f} m")

# Estimate center of mass height (distance from wheel axis to body COM)
body_com_pos = data.xpos[body_id]
l = body_com_pos[2]  # Height is the Z coordinate
print(f"Pendulum length (l) - COM height: {l:.4f} m")

# Gravity
g = -model.opt.gravity[2]  # Gravity magnitude (typically 9.81)
print(f"Gravity (g): {g:.4f} m/s^2")

# Damping coefficient
b = 0.1
print(f"Damping coefficient (b): {b:.4f} N*s/m (estimated)")
print()

# === BUILD STATE-SPACE MODEL ===
M_total = M_body + M_cart
p = I_body * M_total + M_body * M_cart * l**2

A = np.array([
    [0, 1, 0, 0],
    [0, -(I_body + M_body*l**2)*b/p, (M_body**2 * g * l**2)/p, 0],
    [0, 0, 0, 1],
    [0, -(M_body*l*b)/p, M_body*g*l*M_total/p, 0]
])

B = np.array([
    [0],
    [(I_body + M_body*l**2)/p],
    [0],
    [M_body*l/p]
])

print("STATE-SPACE MATRICES:")
print("A matrix:")
print(A)
print("\nB matrix:")
print(B.flatten())
print()

# === LQR CONTROLLER DESIGN ===
Q = np.diag([1.0, 1.0, 100.0, 10.0])  # Heavily penalize roll angle
R = np.array([[1.0]])  # Control effort penalty (increased for smoother control)

try:
    P = linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    K = K.flatten()
    
    print("LQR OPTIMAL GAINS:")
    print(f"K = {K}")
    print(f"  K_position    = {K[0]:.4f}")
    print(f"  K_velocity    = {K[1]:.4f}")
    print(f"  K_angle       = {K[2]:.4f}")
    print(f"  K_angular_vel = {K[3]:.4f}")
    print()
except Exception as e:
    print(f"LQR computation failed: {e}")
    print("Using manually tuned gains instead")
    K = np.array([0.0, 5.0, 30.0, 15.0])

# State tracking
lateral_position = 0.0
lateral_velocity = 0.0
prev_body_pos = None

# Wheel velocity tracking (integrated from acceleration commands)
wheel_velocity = 0.0
max_wheel_velocity = 500000.0  # Maximum wheel velocity (rad/s)

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

def state_space_lqr_controller(model, data):
    """
    LQR-based state-space controller for inverted pendulum.
    Control output is wheel ACCELERATION, which is integrated to get velocity.
    """
    global lateral_position, lateral_velocity, prev_body_pos, wheel_velocity
    
    # === STATE 1 & 2: Lateral position and velocity ===
    body_pos = data.xpos[body_id].copy()
    
    if prev_body_pos is not None:
        dt = model.opt.timestep
        lateral_velocity = (body_pos[1] - prev_body_pos[1]) / dt
    
    lateral_position = body_pos[1]
    prev_body_pos = body_pos
    
    # === STATE 3 & 4: Roll angle and angular velocity ===
    roll_angle = get_body_roll(data, body_id)
    body_angular_velocity = data.cvel[body_id][3:6]
    roll_rate = body_angular_velocity[0]
    
    # === CONSTRUCT STATE VECTOR ===
    state = np.array([
        lateral_position,
        lateral_velocity,
        roll_angle,
        roll_rate
    ])
    
    # === LQR FEEDBACK CONTROL LAW ===
    # Control output is desired wheel ACCELERATION
    wheel_acceleration = -np.dot(K, state)
    
    # Integrate to get wheel velocity
    dt = model.opt.timestep
    wheel_velocity += wheel_acceleration * dt
    
    # Clamp wheel velocity to reasonable limits
    wheel_velocity = np.clip(wheel_velocity, -max_wheel_velocity, max_wheel_velocity)
    
    # Apply smooth velocity command to both wheels
    data.ctrl[wheel_1_id] = wheel_velocity
    data.ctrl[wheel_2_id] = -wheel_velocity
    
    # Keep legs neutral
    data.ctrl[femur_1_id] = 0.0
    data.ctrl[tibia_1_id] = 0.0
    data.ctrl[femur_2_id] = 0.0
    data.ctrl[tibia_2_id] = 0.0
    
    return state, wheel_acceleration, wheel_velocity

def input_listener():
    """
    Listen for keyboard input in the terminal to toggle pause.
    """
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
print("=" * 80)
print("SMOOTH LQR INVERTED PENDULUM CONTROLLER")
print("Control output = wheel ACCELERATION (integrated to velocity)")
print()
print("SIMULATION STARTING PAUSED")
print("Press ENTER in this terminal to pause/unpause")
print("=" * 80)
print()

# Run simulation with viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        
        if not is_paused:
            state, accel, velocity = state_space_lqr_controller(model, data)
            mujoco.mj_step(model, data)
            
            if timestep_counter % print_every_n_steps == 0:
                print(f"Step {timestep_counter}:")
                print(f"  State: y={state[0]:6.3f}m, y_dot={state[1]:6.3f}m/s, "
                      f"phi={np.degrees(state[2]):6.2f}°, phi_dot={np.degrees(state[3]):6.2f}°/s")
                print(f"  Wheel Accel: {accel:6.2f} rad/s^2, Wheel Vel: {velocity:6.2f} rad/s")
                print(f"  Height: {data.xpos[body_id][2]:.3f}m")
            
            timestep_counter += 1
        
        viewer.sync()
        
        elapsed = time.time() - step_start
        sleep_time = model.opt.timestep - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
