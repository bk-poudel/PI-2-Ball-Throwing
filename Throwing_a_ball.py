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


def generate_velocity_profiles(w_x, w_z, phi):
    v_x = phi @ w_x
    v_z = phi @ w_z
    return v_x, v_z


def generate_throwing_trajectory(
    x_start, throw_direction, amplitude, duration, n_points
):
    t = np.linspace(0, 1, n_points)
    trajectory = np.tile(x_start, (n_points, 1))
    trajectory[:, 0] = x_start[0] + amplitude * t  # Linear along X
    return trajectory


def rollout(
    robot, trajectory, R_init, v_x, v_z, dt, gravity_force, K, D, post_duration=2.0
):
    actual_trajectory = []
    velocities = []
    robot.reset(True)
    n_points = len(trajectory)
    for t in range(n_points):
        dx_des = np.zeros(6)
        if t < n_points - 1:
            dx_des[0] = v_x[t]
            dx_des[2] = v_z[t]
        q, dq = robot.get_state()
        T_current = robot.get_pose(q)
        x_current = T_current[:3, 3]
        R_ee = T_current[:3, :3]
        actual_trajectory.append(x_current.copy())
        velocities.append(dx_des[:3].copy())
        pos_err = trajectory[t] - x_current
        R_err = R_init @ R_ee.T
        rotvec_err = R.from_matrix(R_err).as_rotvec()
        err = np.concatenate([pos_err, rotvec_err])
        J = robot.get_jacobian(q)  # Should be 6 x n_joints
        J_spatial = np.vstack([R_ee @ J[3:], R_ee @ J[:3]])
        dx = J_spatial @ dq
        vel_error = dx_des - dx
        desired_wrench = K @ err + D @ vel_error - gravity_force
        tau = J_spatial.T @ desired_wrench + robot.get_gravity(q)
        robot.send_joint_torque(tau, 10)
        time.sleep(dt)
    # Stopping before ball release
    for _ in range(int(0.5 / dt)):
        q, dq = robot.get_state()
        T_current = robot.get_pose(q)
        x_current = T_current[:3, 3]
        R_ee = T_current[:3, :3]
        pos_err = trajectory[-1] - x_current
        R_err = R_init @ R_ee.T
        rotvec_err = R.from_matrix(R_err).as_rotvec()
        err = np.concatenate([pos_err, rotvec_err])
        J = robot.get_jacobian(q)
        J_spatial = np.vstack([R_ee @ J[3:], R_ee @ J[:3]])
        dx = J_spatial @ dq
        vel_error = -dx
        K_stop = K * 2
        D_stop = D * 3
        desired_wrench = K_stop @ err + D_stop @ vel_error - gravity_force
        tau = J_spatial.T @ desired_wrench + robot.get_gravity(q)
        robot.send_joint_torque(tau, 10)
        time.sleep(dt)
    # Release ball
    robot.set_weld_active("ball_weld", False)
    steps = int(post_duration / dt)
    ball_positions = []
    for _ in range(steps):
        q, dq = robot.get_state()
        T_current = robot.get_pose(q)
        x_current = T_current[:3, 3]
        R_ee = T_current[:3, :3]
        pos_err = trajectory[-1] - x_current
        R_err = R_init @ R_ee.T
        rotvec_err = R.from_matrix(R_err).as_rotvec()
        err = np.concatenate([pos_err, rotvec_err])
        J = robot.get_jacobian(q)
        J_spatial = np.vstack([R_ee @ J[3:], R_ee @ J[:3]])
        dx = J_spatial @ dq
        vel_error = -dx
        desired_wrench = K @ err + D @ vel_error - gravity_force
        tau = J_spatial.T @ desired_wrench + robot.get_gravity(q)
        robot.send_joint_torque(tau, 10)
        ball_pos = robot.get_body_pose("ball")[1]
        ball_positions.append(ball_pos)
        time.sleep(dt)
    return np.array(actual_trajectory), np.array(ball_positions), np.array(velocities)


def compute_cost(ball_trajectory, basket_location, velocities):
    ground_threshold = 0.05
    landing_indices = np.where(ball_trajectory[:, 2] < ground_threshold)[0]
    if len(landing_indices) > 0:
        landing_pos = ball_trajectory[landing_indices[0]]
    else:
        landing_pos = ball_trajectory[-1]
    distance_cost = np.linalg.norm(landing_pos - basket_location) ** 2
    velocity_cost = 0.01 * np.sum(velocities**2)
    max_velocity = np.max(np.linalg.norm(velocities, axis=1))
    velocity_penalty = 100 if max_velocity < 0.1 else 0
    return distance_cost + velocity_cost + velocity_penalty


def pi2_update(w, costs, noise, lambda_=0.1):
    costs = np.array(costs)
    min_cost = np.min(costs)
    max_cost = np.max(costs)
    if max_cost - min_cost < 1e-10:
        probs = np.ones(len(costs)) / len(costs)
    else:
        normalized_costs = (costs - min_cost) / (max_cost - min_cost)
        exp_costs = np.exp(-1.0 / lambda_ * normalized_costs)
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


def plot_ball_landing(ball_flight, basket_location):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(ball_flight[:, 0], ball_flight[:, 2], "b-", label="Ball trajectory")
    ax1.scatter(
        basket_location[0],
        basket_location[2],
        color="r",
        label="Basket",
        s=80,
        marker="*",
    )
    ax1.set_xlabel("X")
    ax1.set_ylabel("Z")
    ax1.set_title("Ball Trajectory (X-Z view)")
    ax1.grid()
    ax1.legend()
    ax2.plot(ball_flight[:, 1], ball_flight[:, 2], "b-", label="Ball trajectory")
    ax2.scatter(
        basket_location[1],
        basket_location[2],
        color="r",
        label="Basket",
        s=80,
        marker="*",
    )
    ax2.set_xlabel("Y")
    ax2.set_ylabel("Z")
    ax2.set_title("Ball Trajectory (Y-Z view)")
    ax2.grid()
    ax2.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dt = 0.001
    T = 2
    steps = int(T / dt)
    robot = FR3Sim()
    q, dq = robot.get_state()
    T_current = robot.get_pose(q)
    x_current = T_current[:3, 3]
    R_current = T_current[:3, :3]
    basket_location = robot.get_site_pose("box_center")
    print(f"Basket location: {basket_location}")
    throw_direction = basket_location - x_current
    throw_direction[1] = 0  # Force to X-Z plane only
    throw_direction /= np.linalg.norm(throw_direction)
    throw_amplitude = 0.3
    trajectory = generate_throwing_trajectory(
        x_current, throw_direction, throw_amplitude, T, steps
    )
    gravity_vector = robot.get_gravity_vector()
    ball_mass = robot.get_body_mass("ball")
    gravity_force = np.concatenate((ball_mass * gravity_vector, np.zeros(3)))
    K_constant = np.diag([800, 800, 800, 200, 200, 200])
    D_constant = np.diag([100, 100, 100, 50, 50, 50])
    n_basis = 8
    n_rollouts = 15
    n_iters = 25
    noise_std = 0.5
    phi = gaussian_basis_functions(n_basis, T, dt)
    w_x = np.zeros(n_basis)
    w_z = np.zeros(n_basis)
    print("Starting PI² learning...")
    best_cost = float("inf")
    best_weights = None
    for it in range(n_iters):
        costs = []
        noise_x_list = []
        noise_z_list = []
        for rollout_idx in range(n_rollouts):
            noise_x = np.random.normal(0, noise_std, n_basis)
            noise_z = np.random.normal(0, noise_std, n_basis)
            w_trial_x = w_x + noise_x
            w_trial_z = w_z + noise_z
            v_x, v_z = generate_velocity_profiles(w_trial_x, w_trial_z, phi)
            try:
                _, ball_flight, velocities = rollout(
                    robot,
                    trajectory,
                    R_current,
                    v_x,
                    v_z,
                    dt,
                    gravity_force,
                    K_constant,
                    D_constant,
                )
                cost = compute_cost(ball_flight, basket_location, velocities)
                costs.append(cost)
                if cost < best_cost:
                    best_cost = cost
                    best_weights = (w_trial_x.copy(), w_trial_z.copy())
            except Exception as e:
                print(f"Rollout {rollout_idx} failed: {e}")
                costs.append(1000)
            noise_x_list.append(noise_x)
            noise_z_list.append(noise_z)
        w_x = pi2_update(w_x, costs, np.array(noise_x_list))
        w_z = pi2_update(w_z, costs, np.array(noise_z_list))
        noise_std *= 0.98
        print(
            f"[Iter {it+1}] Best cost: {min(costs):.4f}, Avg cost: {np.mean(costs):.4f}"
        )
    print(f"\nLearning completed! Best cost achieved: {best_cost:.4f}")
    if best_weights is not None:
        w_x, w_z = best_weights
    v_x, v_z = generate_velocity_profiles(w_x, w_z, phi)
    final_traj, ball_flight, final_velocities = rollout(
        robot,
        trajectory,
        R_current,
        v_x,
        v_z,
        dt,
        gravity_force,
        K_constant,
        D_constant,
    )
    plot_trajectory(
        final_traj, goal=trajectory[-1], title="Learned Throwing Trajectory"
    )
    plot_ball_landing(ball_flight, basket_location)
    landing_idx = np.where(ball_flight[:, 2] < 0.05)[0]
    landing_position = (
        ball_flight[landing_idx[0]] if len(landing_idx) > 0 else ball_flight[-1]
    )
    distance_to_basket = np.linalg.norm(landing_position - basket_location)
    print(f"Ball landed at: {landing_position}")
    print(f"Distance to basket: {distance_to_basket:.4f}")
    print(f"Basket location: {basket_location}")
