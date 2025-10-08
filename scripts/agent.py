class Agent:
    def __init__(self, eps, eps_dec, eps_min, gamma):
        self.eps = eps
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.gamma = gamma
