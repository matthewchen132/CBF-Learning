import math

class LookaheadState:
    '''
    Augmented State which gets around the non-holonomic properties of a CBF
    
    '''
    def __init__(self, x, y, theta, l):
        self.x = x
        self.y = y
        self.l = l
        self.xp = self.x + l*math.cos(theta)
        self.yp = self.y + l*math.sin(theta)
        self.theta = theta

    def __add__(self, other):
        return LookaheadState(
            self.x + other.x,
            self.y + other.y,
            self.theta + other.theta,
            self.l
        )

    def __mul__(self, scalar):
        return LookaheadState(
            self.x * scalar,
            self.y * scalar,
            self.theta * scalar,
            self.l
        )
