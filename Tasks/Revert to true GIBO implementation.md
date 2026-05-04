Tasks

4/17 - 4/20
[X] Test correct GIBO as well
    [X] BUILD DATA WITH GIBO, dont train first. <- Done
    [X] m52 to SE  <- Done
    [X] Extend State to have {x, y, theta} <- resolved with Lookahead State Logic
    [X] Verify Lie Derivative math <- resolved with Lookahead State Logic

[X] Clean Code
    [X] Simulator Class
    [X] GIBO control Class


Next Steps (high priority):
[] Robotic manipulator, implement dynamics and GIBO
    [] Follow a trajectory under learned safety condition
    [*] Plots by 1 week from now
    [] CBF based on angle limit:  r = pisin(t), pi/2 sin(4t)





Lower Priority:
[] Implement
[] Monte Carlo/gather useful info about performance
    [] GP improvement of covariance over time. We see this visually but can we verify? 
    [] Baseline vs conventional cbf
    [] Hybrid acq function



