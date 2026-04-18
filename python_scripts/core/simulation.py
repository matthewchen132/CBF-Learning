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

    def step(self):
        self.curr_time += self.dt

    def log_data(self, X, u, dist, time, query_point=None):
        self.logs["state"].append(X)
        self.logs["u"].append(u)
        self.logs["dist"].append(dist)
        self.logs["times"].append(time)
        if query_point is not None:
                self.logs["query_points"].append(query_point.flatten().tolist())

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