## THis document is to plan the setup of metrics to track and monte carlo over for the 3 different acquisition functions.

We specifically want to prove that acquisition function 3 is able to see some significant performance improvement.
We want to create a logger class to handle all of these things as well.

# Overview:
1. Monte carlo over some sensitive variables.
2. Evaluate Performance Metrics for each algorithm 


## Track these settings in configuration of each run and dump out to outputs with a timestamped date time.

## (1) Monte Carlo decisions to change:
- Noise initialization of the GP (also is set to outputscale)
        sigma2 = {0.05, 0.2, 0.5}
- Algorithm Ran (GIBO, hybrid, etc.)
- Sweep over # of M samples 
        M = {1,2,8}
- Sweep over starting theta ranges (5 theta presets for each M and Sigma2)

-  3 x 5 x 5 = 125 monte carlos. 

        

## (2) Performance Metrics (averages):
- Posterior Covariance Reduction
- Gradient Posterior Covariance Reduction [!!!] 
- [?] Some conservatism measurement, example (RMS distance)
- Runtime of algorithm



## Configuration variables (config/config.py)
- Max theta1 angle in deg
- Hyperparameter lengthscales and outputscales (should only need to log the starting values)
- Number of retrain points M in a single local search window.
- L0, S0, Noise0
- manipulator class members
- GP_JITTER