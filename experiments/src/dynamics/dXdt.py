import numpy as np
from src.dynamics.state import State
from src.dynamics.augmented_state import LookaheadState

def dXdt(X, V, u):
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

