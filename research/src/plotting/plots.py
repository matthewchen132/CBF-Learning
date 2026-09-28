import matplotlib.pyplot as plt
import numpy as np
'''
WIP functions to Plot data from Main
'''
def plot_obstacle(axes, fig_num, obs_x, obs_y, color_str, label):
    axes[fig_num].plot(obs_x, obs_y, 'k', label="Obstacle")

def plot_trajectories(axes, circle_obstacle, waypoint, robot_pos, noisy_robot_pos, gp_robot_pos, n_std_deviations, query_points):
    # TODO will remove in favor of modular plotting
    plot_obstacle(axes=axes, fig_num=0, obs_x=circle_obstacle[:,0], obs_y=circle_obstacle[:,1], color_str='k', label="Obstacle")
    axes[0].plot(waypoint[0], waypoint[1], 'g*', label="waypoint")
    axes[0].plot(robot_pos[:,0], robot_pos[:,1], 'b--', label="Ideal (Zero Noise)")
    axes[0].plot(noisy_robot_pos[:,0], noisy_robot_pos[:,1], 'r-', label="Noisy)")
    axes[0].plot(gp_robot_pos[:,0], gp_robot_pos[:,1], 'g-', linewidth=2, label=f"GP-GIBO-CBF with {n_std_deviations} Deviations")

    axes[0].scatter(query_points[:,0], query_points[:,1], 
                    c=np.linspace(0, 1, len(query_points)), 
                    cmap='Blues', s=10, alpha=0.3, label="GIBO Pts (Darker blue = Latest)")
    axes[0].set_title("Trajectories (Ideal CBF vs. Noisy CBF vs. GP-GIBO-CBF)")
    axes[0].axis("equal")
    axes[0].legend()
    axes[0].grid(True)


def plot_error_wrt_time(axes, times, dist_data, dist_noisy_data, dist_gp_data, label):
    # TODO will remove in favor of modular plotting
    '''
    Plots Error (Distance from waypoint) w.r.t Time
    
    '''
    d_ideal = np.array(dist_data[:len(times)])
    d_noisy = np.array(dist_noisy_data[:len(times)])
    d_gp    = np.array(dist_gp_data[:len(times)])

    axes[1].plot(times, d_ideal, 'b-', label="Ideal")
    axes[1].plot(times, d_noisy, 'r-', label="Noisy")
    axes[1].plot(times, d_gp, 'g-', label="GP-GIBO")
    axes[1].set_title("Error (Dist. to Waypoint)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Distance (m)")
    axes[1].grid(True)
    axes[1].legend()

def plot_control_wrt_time(axes, u_data, u_noisy_data, u_gp_data, times):
    u_ideal_plot = np.array(u_data[:len(times)])
    u_noisy_plot = np.array(u_noisy_data[:len(times)])
    u_gp_plot    = np.array(u_gp_data[:len(times)])

    axes[2].plot(times, u_ideal_plot, 'b', label="Nominal u")
    axes[2].plot(times, u_noisy_plot, 'r', label="Noisy u")
    axes[2].plot(times, u_gp_plot, 'g', label="GP-CBF u")
    axes[2].set_title("Control Input (u) Comparison")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Angular Velocity (rad/s)")
    axes[2].legend()
# -----------------------------------------------------------------------------------------
# new modular plotting functions
def plot_trajectory(axes, obstacle, waypoint, robot_pos, query_points, fig_num, label):
    # -- Trajectory -- 
    plot_obstacle(axes=axes, fig_num=fig_num, obs_x=obstacle[:,0], obs_y=obstacle[:,1], color_str='k', label="Obstacle")
    axes[fig_num].plot(waypoint[0], waypoint[1], 'g*', label="waypoint")
    axes[fig_num].plot(robot_pos[:,0], robot_pos[:,1], 'g-', linewidth=2, label=label)
    axes[fig_num].scatter(query_points[:,0], query_points[:,1], 
                    c=np.linspace(0, 1, len(query_points)), 
                    cmap='Blues', s=10, alpha=0.3, label="GIBO Pts (Darker blue = Latest)")
    axes[fig_num].set_title("Trajectories (Ideal CBF vs. Noisy CBF vs. GP-GIBO-CBF)")
    axes[fig_num].axis("equal")
    axes[fig_num].legend()
    axes[fig_num].grid(True)

def plot_distance_error(axes, times, dist_data, fig_num, label):
    dist    = np.array(dist_data[:len(times)])

    axes[fig_num].plot(times, dist, 'g-', label=label)
    axes[fig_num].set_title("Error (Dist. to Waypoint)")
    axes[fig_num].set_xlabel("Time (s)")
    axes[fig_num].set_ylabel("Distance (m)")
    axes[fig_num].grid(True)
    axes[fig_num].legend()

def plot_control(axes, u_data, times, fig_num, label):
    u_plot = np.array(u_data[:len(times)])
    axes[fig_num].plot(times, u_plot, 'b', label=label)
    axes[fig_num].set_title("Control Input (u) Comparison")
    axes[fig_num].set_xlabel("Time (s)")
    axes[fig_num].set_ylabel("Angular Velocity (rad/s)")
    axes[fig_num].legend()