from src.dynamics.state import State
from src.dynamics.dXdt import dXdt

def rk4_step(X, V, t, dt, u):
    '''
    Runge Kutta 4 Numerical Integration
    
    '''
    k1 = dXdt(X, V, u)
    k2 = dXdt(X + k1*0.5*dt, V, u)
    k3 = dXdt(X + k2*0.5*dt, V, u)
    k4 = dXdt(X + k3*dt, V, u)
    X = X + (k1 + k2*2 + k3*2 + k4) * (dt/6.0)
    return X;
