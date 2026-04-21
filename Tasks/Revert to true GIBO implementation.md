04/17 Revert to true GIBO implementation

[] Test correct GIBO as well
    [X] BUILD DATA WITH GIBO, dont train first. <- Done
    [X] m52 to SE  <- Done
    [X] Extend State to have {x, y, theta} <- resolved with Lookahead State Logic
    [X] Verify Lie Derivative math <- resolved with Lookahead State Logic

[ ] Clean Code
    [X] Simulator Class
    [X] GIBO control Class

[ ] Robotic manipulator, implement dynamics and GIBO
     [ ] CBF based on angle limit:  r = pisin(t), pi/2 sin(4t)



