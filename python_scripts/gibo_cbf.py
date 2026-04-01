import torch
import gpytorch as gp
import numpy as np
import matplotlib.pyplot as plt
import pybullet_data
import math

# from GIBO

'''
03/24 Deliverable
Safe navigation of robot with obstacles, (get familiar with signed distance functions)
 - implement learning cbf
 - compare performance with Azra's weighted acq funciton, gibo

What am I trying to do?
 - See if learning a NOISY control bar using a GP and GIBO provides a betteer

 1) spawn robot <- Done
 2) spawn obstacle <- Done
 3) make signed distance function <- Done
 4) traditional navigation to waypoint
 5) GP Learning based navigation to waypoint
'''

class State:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def __add__(self, other):
        return State(
            self.x + other.x,
            self.y + other.y,
            self.theta + other.theta
        )

    def __mul__(self, scalar):
        return State(
            self.x * scalar,
            self.y * scalar,
            self.theta * scalar
        )

def optimal_control(X, Kp, goal_xy):
    '''
    Solves for the optimal control input u
    by finding the instantaneous angular error and doing Kp * angular_error
    . returns in rads2deg
    '''
    dx = goal_xy[0] - X.x
    dy = goal_xy[1] - X.y
    theta_target = np.atan2(dy, dx)
    theta_error = theta_target - X.theta
    theta_error = (theta_error + np.pi) % (2 * np.pi) - np.pi
    return Kp * theta_error # 
    


def dXdt(t, X, V, u):
    '''
    Modeling as a Dubin's Car
    
    -- inputs --
    . velocity (Pre-defined Constat)
    . u: angular velocity (control input)

    -- State Equations -- (X is state, x is position variable)
    . X_dot = AX + Bu 
    . x_dot = V * cos(theta)
    . y_dot = V * sin(theta)
    . theta_dot = u
    '''

    x_dot = V * np.cos(X.theta)
    y_dot = V * np.sin(X.theta)
    theta_dot = u
    return State(x_dot, y_dot, theta_dot)

def rk4_step(X, V, t, dt, u):
    '''
    Runge Kutta 4 Numerical Integration
    
    '''
    k1 = dXdt(t, X, V, u)
    k2 = dXdt(t + 0.5*dt, X + k1*0.5*dt, V, u)
    k3 = dXdt(t + 0.5*dt, X + k2*0.5*dt, V, u)
    k4 = dXdt(t + dt, X + k3*dt, V, u)
    X = X + (k1 + k2*2 + k3*2 + k4) * (dt/6.0)
    print(f"X : {round(X.x,2)}, Y : {round(X.y,2)} theta : {round(X.theta * 180/np.pi, 2)}")
    return X;



def gen_waypoint(gridsize):
    '''
    Generates a random waypoint in gridsize
    '''
    waypoint_x = np.random.uniform(0.5*gridsize, -0.5*gridsize)
    waypoint_y = np.random.uniform(0.5*gridsize, -0.5*gridsize)
    waypoint = np.array([waypoint_x, waypoint_y])
    return waypoint

def sdf(robot_x, robot_y, circle_pts, Noisy, noise):
    noise_copy = noise
    circle_pts_copy = circle_pts.copy()
    '''
    -- Signed Distance Function --
    . returns distance from the closest point defining the obstacle, along with the index to access the closest point
    . robot_x, robot_y: current x,y position
    . circle_pts: Nx2 numpy array
    '''
    rng = np.random.default_rng(0)
    if Noisy:
        noise_copy = noise[0:circle_pts.shape[0]]
        circle_pts_copy[:,0] += noise_copy
        circle_pts_copy[:,1] += noise_copy

    dx = robot_x - circle_pts_copy[:, 0]
    dy = robot_y - circle_pts_copy[:, 1]
    distances = np.array([dx,dy]).T
    norms = np.linalg.norm(distances, axis=1)
    min_dist_idx = np.argmin(norms)
    return [norms[min_dist_idx], min_dist_idx]

def spawn_circle(x, y, r):
    '''
    Spawns a 2d circle based on x-y position, radius, and 
    360 pts used to define the circle
    '''
    theta = 0
    pts = []
    while(theta <= 360):
        circle_x = r * np.cos(np.deg2rad(theta)) + x
        circle_y = r * np.sin(np.deg2rad(theta)) + y
        pts.append([circle_x, circle_y])
        theta += 10
    
    pts = np.array(pts)
    return pts
def cbf(alpha1, alpha2, X, closest_idx, h, circle_obstacle, Kp, V, waypoint):
    '''
    -- Control Barrier Function for Dubin's Car
    - Returns the control "u" at that point
    '''
    u_data = []
# -- dhdx >= -alpha * h(x) --
    alpha1 = 1.0
    grad_x = (X.x - circle_obstacle[closest_idx, 0]) / h
    grad_y = (X.y - circle_obstacle[closest_idx, 1]) / h
    # -- dhdt --
    dhdt = grad_x*V*np.cos(X.theta) + grad_y*V*np.sin(X.theta)

    # -- Let "b(x) = h_dot(x) + alpha1 * h(x)" --
    b = dhdt + alpha1 * h
    # -- Forcing b_dot(x) >= - alpha2 * b(x) brings "u" term back --
    coeff_u = V * (-grad_x * np.sin(X.theta) + grad_y * np.cos(X.theta))
    # -- The Remainder of b_dot (The 'drift' term LfLf_h) --
    dbdt = alpha1 * dhdt 
    # -- The CBF Condition: b_dot >= -alpha2 * b --
    # . (coeff_u * u) + dbdt(x) >= -alpha2 * b
    u_min_allowed = (-alpha2 * b - dbdt) / (coeff_u)
    u_nom = optimal_control(X, Kp, waypoint)
    # -- Minimally Invasive Filter -- 
    # . (coeff_u * u) + dbdt(x) >= -alpha2 * b
    # . u >= (-alpha2 * b(x) - dbdt(x) ) / coeff_u
    if coeff_u > 0:
        # Base case with positive signed coeff_u
        u_final = max(u_nom, (-alpha2 * b - dbdt) / (coeff_u + 1e-6))
    elif coeff_u < 0:
        # u must be LESS than some value
        u_final = min(u_nom, (-alpha2 * b - dbdt) / (coeff_u + 1e-6))
    else:
        u_final = u_nom
    # -- find u_nom -- 
    return u_final

def main():
    # -- Simulation Setup --
    t = 0.0
    dt = 0.1
    u = 0.0
    Kp = 4
    end_time = 20.0
    # -- Create a Waypoint --
    waypoint = np.array([5.0, 5.0])
    # -- Spawn Robot --
    X = State(0.0, 0.0, 0.0)
    X_noisy = State(0.0, 0.0, 0.0)
    V = 1.0
    # -- Spawn Obstacle --
    circle_obstacle = spawn_circle(3.5, 3.5, r=1.0)
    noisy_circle_obstacle = spawn_circle(3.5, 3.5, r=1.0)

    # -- Simulation Loop --
    # -- Position --
    robot_pos = []
    noisy_robot_pos = []
    # -- Control --
    u_data = []
    noisy_u_data = []
    times = []

    rng = np.random.default_rng(seed=1)
    noise_std_dev = 0.01
    goal1_reached = False
    goal2_reached = False

    while(t < end_time):
        noise = rng.normal(0, noise_std_dev, size=circle_obstacle.size)
        times.append(t)
        # -- Conventional CBF --
        # . h(x) = sqrt ( (x-xc)^2 + (y-yc)^2 ) - r_c
        h, closest_idx = sdf(X.x, X.y, circle_obstacle, Noisy=False, noise=noise) # Note that conventionally, h = SDF
        # -- NOISY -- 
        h_noisy, closest_noisy_idx = sdf(X.x, X.y, noisy_circle_obstacle, Noisy=True, noise=noise) # Note that conventionally, h = SDF
        print(f"h:{h} , h_noisy: {h_noisy}")
        
        alpha1 = 1.0
        alpha2 = 1.5
        u_final = cbf(alpha1, alpha2, X, closest_idx, h, circle_obstacle, Kp, V, waypoint)
        noisy_u_final = cbf(alpha1, alpha2, X_noisy, closest_noisy_idx, h_noisy, circle_obstacle, Kp, V, waypoint)


        # -- find u_nom -- 
        u_data.append(u_final)
        noisy_u_data.append(noisy_u_final)
        # -- Update Position w/ RK4 --
        X = rk4_step(X, V, t, dt, u_final)
        X_noisy = rk4_step(X_noisy, V, t, dt, noisy_u_final)
        print(f"Current u (no noise): {u_final}")
        print(f"Current u (noisy): {noisy_u_final}")
        robot_pos.append([X.x, X.y])
        noisy_robot_pos.append([X_noisy.x, X_noisy.y])
        # -- Check if Waypoint Reached --
        dx = abs(waypoint[0] - X.x)
        dy = abs(waypoint[1] - X.y)
        dx_noisy = abs(waypoint[0] - X_noisy.x)
        dy_noisy = abs(waypoint[1] - X_noisy.y)
        if math.sqrt(dx**2 + dy**2) < 0.05 or goal1_reached:
            # -- Stay at Waypoint once reached --
            X.x = waypoint[0]
            X.y = waypoint[1]
            goal1_reached = True
        if math.sqrt(dx_noisy**2 + dy_noisy**2) < 0.1 or goal2_reached:
            # -- Stay at Waypoint once reached --
            X_noisy.x = waypoint[0]
            X_noisy.y = waypoint[1]
            goal2_reached = True
        if goal1_reached and goal2_reached:
            break
        # -- Increase Sim. Time --
        t += dt

    robot_pos = np.array(robot_pos)
    noisy_robot_pos = np.array(noisy_robot_pos)

    # -- Plot Config
    plt.plot(circle_obstacle[:,0], circle_obstacle[:,1], label="Obstacle", color='k', linewidth=1)
    plt.plot(noisy_circle_obstacle[:,0], noisy_circle_obstacle[:,1], label="Noisy Obstacle", color='r', linewidth=1)
    plt.plot(waypoint[0], waypoint[1], 'g*', markersize=10, label="Waypoint")
    plt.plot(robot_pos[:,0], robot_pos[:,1], 'b-', markersize=2, label="Zero Noise Position")
    plt.plot(noisy_robot_pos[:,0], noisy_robot_pos[:,1], 'g-', markersize=2, label="Noisy Position")
    plt.xlabel("X")
    plt.xlim(-2,6)
    plt.ylabel("Y")
    plt.ylim(-2,6)
    plt.title(f"Ideal CBF vs CBF with Noise Std. Dev. of {noise_std_dev}")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()
    print("End Function")

if __name__ == "__main__":
    main()