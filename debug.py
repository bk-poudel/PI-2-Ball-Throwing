import numpy as np
import matplotlib.pyplot as plt
from MujocoSim import FR3Sim
from scipy.spatial.transform import Rotation as R
import time


def gaussian_basis_functions(n_basis, T, dt, h=0.5):
    t = np.linspace(0, T, int(T / dt))
    centers = np.linspace(0, T, n_basis)
    widths = np.ones(n_basis) * h
    phi = np.exp(-0.5 * ((t[:, None] - centers[None, :]) ** 2) / (widths[None, :] ** 2))
    phi /= phi.sum(axis=1, keepdims=True)
    return phi


def generate_velocity_profiles(w_x, w_y, phi):
    v_x = phi @ w_x
    v_y = phi @ w_y
    return v_x, v_y


def generate_linear_trajectory(x_start, x_end, duration, n_points):
    x_traj = np.linspace(x_start[0], x_end[0], n_points)
    y_traj = np.linspace(x_start[1], x_end[1], n_points)
    z_traj = np.linspace(x_start[2], x_end[2], n_points)
    return np.vstack([x_traj, y_traj, z_traj]).T


def rollout(
    robot, trajectory, R_init, v_x, v_y, dt, gravity_force, K, D, post_duration=2.0
):
    actual_trajectory = []
    robot.reset(True)
    for t, x_des in enumerate(trajectory):
        dx_des = np.zeros(6)
        if t == len(trajectory) - 1:
            dx_des = np.zeros(6)
        else:
            dx_des[0] = v_x[t]
            # dx_des[2] = v_y[t]
        q, dq = robot.get_state()
        T_current = robot.get_pose(q)
        x_current = T_current[:3, 3]
        R_ee = T_current[:3, :3]
        actual_trajectory.append(x_current.copy())
        pos_err = x_des - x_current
        R_err = R_init @ R_ee.T
        rotvec_err = R.from_matrix(R_err).as_rotvec()
        err = np.concatenate([pos_err, rotvec_err])
        J = robot.get_jacobian(q)
        J_spatial = np.vstack([R_ee @ J[3:], R_ee @ J[:3]])
        dx = J_spatial @ dq
        vel_error = dx_des - dx
        desired_wrench = K @ err + D @ vel_error - gravity_force
        tau = J_spatial.T @ desired_wrench
        tau += robot.get_gravity(q)
        robot.send_joint_torque(tau, 10)
        time.sleep(dt)
    # Release the ball
    robot.set_weld_active("ball_weld", False)
    # Passive simulation to see ball landing
    steps = int(post_duration / dt)
    ball_positions = []
    x_des_final = trajectory[-1]
    dx_des_final = np.zeros(6)
    # Gravity force without the ball
    gravity_force_noball = np.zeros(6)
    for _ in range(steps):
        q, dq = robot.get_state()
        # Controller to hold position
        T_current = robot.get_pose(q)
        x_current = T_current[:3, 3]
        R_ee = T_current[:3, :3]
        pos_err = x_des_final - x_current
        R_err = R_init @ R_ee.T
        rotvec_err = R.from_matrix(R_err).as_rotvec()
        err = np.concatenate([pos_err, rotvec_err])
        J = robot.get_jacobian(q)
        J_spatial = np.vstack([R_ee @ J[3:], R_ee @ J[:3]])
        dx = J_spatial @ dq
        vel_error = dx_des_final - dx
        desired_wrench = K @ err + D @ vel_error - gravity_force_noball
        tau = J_spatial.T @ desired_wrench
        tau += robot.get_gravity(q)
        robot.send_joint_torque(tau, 10)
        ball_positions.append(robot.get_body_pose("ball")[1])  # Add this line
        time.sleep(dt)
    return np.array(actual_trajectory), np.array(ball_positions)


def compute_cost(trajectory, goal):
    final_pos = trajectory[-1]
    return np.linalg.norm(final_pos - goal) ** 2


def pi2_update(w, costs, noise, lambda_=0.5):
    costs = np.array(costs)
    min_cost = np.min(costs)
    exp_costs = np.exp(-1.0 / lambda_ * (costs - min_cost))
    probs = exp_costs / np.sum(exp_costs)
    delta_w = np.sum(noise * probs[:, None], axis=0)
    return w + delta_w


def plot_trajectory(traj, goal=None, title="Trajectory"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label="Actual Trajectory", color="b")
    if goal is not None:
        ax.scatter(*goal, color="r", label="Goal", s=50)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()
    plt.show()


def plot_ball_landing(ball_flight):
    plt.plot(ball_flight[:, 1], ball_flight[:, 2], label="Ball trajectory (X-Z)")
    # Plot basket location if available in global scope
    try:
        basket_x = basket_location[1]
        basket_z = basket_location[2]
        plt.scatter(basket_x, basket_z, color="r", label="Basket", s=80, marker="*")
    except Exception:
        pass
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("Ball Landing Trajectory")
    plt.grid()
    plt.legend()
    plt.show()


# Main PI^2 learning
dt = 0.001
T = 2
steps = int(T / dt)
robot = FR3Sim()
q, dq = robot.get_state()
T_current = robot.get_pose(q)
x_current = T_current[:3, 3]
R_current = T_current[:3, :3]
x_goal = x_current + np.array([0.5, 0.0, 0.0])
basket_location = robot.get_site_pose("box_center")
print(f"Basket location: {basket_location}")
trajectory = generate_linear_trajectory(
    x_start=x_current, x_end=x_goal, duration=T, n_points=steps
)
gravity_vector = robot.get_gravity_vector()
ball_mass = robot.get_body_mass("ball")
gravity_force = np.concatenate((ball_mass * gravity_vector, np.zeros(3)))
K_constant = np.diag([1800, 18000, 1800, 300, 300, 300])
D_constant = np.diag([200, 200, 200, 100, 100, 100])
n_basis = 20
n_rollouts = 20
n_iters = 30
phi = gaussian_basis_functions(n_basis, T, dt)
w_x = np.zeros(n_basis)
w_y = np.zeros(n_basis)
for it in range(n_iters):
    costs = []
    noise_x_list = []
    noise_y_list = []
    for _ in range(n_rollouts):
        noise_x = np.random.normal(0, 2, n_basis)
        noise_y = np.random.normal(0, 2, n_basis)
        w_trial_x = w_x + noise_x
        w_trial_y = w_y + noise_y
        v_x, v_y = generate_velocity_profiles(w_trial_x, w_trial_y, phi)
        _, actual = rollout(
            robot,
            trajectory,
            R_current,
            v_x,
            v_y,
            dt,
            gravity_force,
            K_constant,
            D_constant,
        )
        cost = compute_cost(actual, basket_location)
        costs.append(cost)
        # plot_ball_landing(actual)
        noise_x_list.append(noise_x)
        noise_y_list.append(noise_y)
    w_x = pi2_update(w_x, costs, np.array(noise_x_list))
    w_y = pi2_update(w_y, costs, np.array(noise_y_list))
    print(f"[Iter {it+1}] Best cost: {min(costs):.4f}")
# Final rollout
v_x, v_y = generate_velocity_profiles(w_x, w_y, phi)
final_traj, ball_flight = rollout(
    robot, trajectory, R_current, v_x, v_y, dt, gravity_force, K_constant, D_constant
)
plot_trajectory(final_traj, goal=x_goal, title="Learned EE Trajectory")
# Plot ball flight
print("Simulating ball flight after throw...")
plot_ball_landing(ball_flight)
# Report landing position
landing_idx = np.argmax(ball_flight[:, 2] < 0.01)
landing_position = ball_flight[landing_idx] if landing_idx > 0 else ball_flight[-1]
print(f"Ball landed at: {landing_position}")
