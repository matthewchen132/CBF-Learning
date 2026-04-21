import matplotlib.pyplot as plt
import numpy as np

class Simulation():
    '''
    Simulation Class
     - times
     - obstacles
     - logging
    
    TODO: add config.py as an input to initialize
    '''
    def __init__(self, dt, end_time):
        self.curr_time = 0.0
        self.dt = dt
        self.end_time = end_time

        self.logs = {
            "state": [],
            "u": [],
            "dist": [],
            "times": [],
            "query_points": []
        }
    def is_running(self):
        return self.curr_time < self.end_time

    def break_if_arrived(self, goal_radius, GIBO):
        if GIBO.dist_to_goal <= goal_radius:
            GIBO.goal_reached = True

    def step(self, print_t):
        self.curr_time += self.dt
        if print_t:
            print(f"Current Time: {self.curr_time}")

    def log_data(self, X, u, dist, time, query_points=None):
        self.logs["state"].append(X)
        self.logs["u"].append(u)
        self.logs["dist"].append(dist)
        self.logs["times"].append(time)
        if query_points is not None:
                self.logs["query_points"].append(query_points.flatten().tolist())

    def plot_trajectory(self, GIBO):
        X = []
        X_p = []
        Y = []
        Y_p = []
        for State in self.logs["state"]:
            X.append(State.x)
            X_p.append(State.xp)
            Y.append(State.y)
            Y_p.append(State.yp)
        fig, ax = plt.subplots(figsize=(8, 8))

        # Robot center and lookahead point trajectories
        ax.plot(X, Y, color='blue', linewidth=1.5, label='Robot center')
        ax.plot(X_p, Y_p, color='green', label='Lookahead point')

        # Start marker
        ax.scatter(X[0], Y[0], color='blue', marker='o', s=80, zorder=5)

        # Obstacle (close the loop by appending first point)
        obs = GIBO.Obstacle.circle_points
        ax.plot(np.append(obs[:, 0], obs[0, 0]),
                np.append(obs[:, 1], obs[0, 1]),
                color='red', linewidth=2, label='Obstacle')
        ax.fill(obs[:, 0], obs[:, 1], color='red', alpha=0.15)

        # Waypoint
        ax.scatter(GIBO.waypoint[0], GIBO.waypoint[1],
                   color='green', marker='*', s=250, zorder=5, label='Waypoint')

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(f"GIBO Trajectory | M samples: {GIBO.M_number_of_queries}")
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()



    def list_to_np(self):
        processed_logs = {}
        for key, value in self.logs.items():
            if key == "state":
                # Extract x and y attributes from each State object
                # This turns a list of Objects into a [N, 2] float matrix
                processed_logs["state"] = np.array([[s.x, s.y] for s in value])
            elif key == "query_points":
                # Ensure query points are also a 2D array [N, 2]
                processed_logs["query_points"] = np.array(value)
            else:
                processed_logs[key] = np.array(value)
        return processed_logs