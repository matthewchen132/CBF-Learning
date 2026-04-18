# Example container to create Simulation objects in the future
config = {
    "sim": {"dt": 0.1, "end_time": 20.0, "seed": 2},
    "robot": {"v": 1.0, "kp": 4.0, "max_w": 4.0},
    "cbf": {"alpha1": 1.0, "alpha2": 1.5},
    "gp": {"noise_std": 0.05, "step_size": 0.1}
}