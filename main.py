import numpy as np
import gpytorch as gp
from gpytorch.kernels import MaternKernel, RBFKernel
import matplotlib.pyplot as plt

def main():
    # define underlying function
    x = np.linspace(1,10,500)
    rng = np.random.default_rng(0)  # reproducible
    noise = rng.normal(loc=0.0, scale= 0.1, size=x.shape)
    y = np.sin(np.pi*x) + noise
    



    # plot
    plt.plot(x,y, 'b.')
    plt.show()

if __name__ == "__main__":
    main()