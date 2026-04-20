from dynamics.state import State
from dynamics.dXdt import dXdt

def GIBO_rk4_step(X_state, exp_grad, step_size, dt):
    '''
    Runge Kutta 4 Numerical Integration for GIBO Gradient Ascent
    X_state: Current State object
    exp_grad: The [dx, dy] gradient vector from Step 13
    step_size: The GIBO η (learning rate)
    dt: The simulation timestep
    '''
    
    # We define the derivative as being constant over the small dt
    def dynamics(state_val, gradient):
        # This represents X_dot = eta * grad(J)
        return step_size * gradient

    k1 = dynamics(X_state, exp_grad)
    k2 = dynamics(X_state + k1 * 0.5 * dt, exp_grad)
    k3 = dynamics(X_state + k2 * 0.5 * dt, exp_grad)
    k4 = dynamics(X_state + k3 * dt, exp_grad)

    # Update state using weighted average of slopes
    new_state_val = X_state + (k1 + 2*k2 + 2*k3 + k4) * (dt / 6.0)
    
    return new_state_val

def rk4_step(X, V, t, dt, u, step_size):
    '''
    Runge Kutta 4 Numerical Integration
    
    '''
    k1 = dXdt(X, V, u)
    k2 = dXdt(X + k1*0.5*dt, V, u)
    k3 = dXdt(X + k2*0.5*dt, V, u)
    k4 = dXdt(X + k3*dt, V, u)
    X = X + (k1 + k2*2 + k3*2 + k4) * (dt/6.0)
    return X;
