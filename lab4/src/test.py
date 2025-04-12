class TabularQLearning(RLAlgorithm):
    def __init__(self, actions: List[ActionT], discount: float, explorationProb: float = 0.2, initialQ: float = 0):
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.initialQ = initialQ
        self.Q = defaultdict(lambda: initialQ)
        self.numIters = 0

    def getAction(self, state: StateT, explore: bool = True) -> ActionT:
        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4:
            explorationProb = 1.0
        elif self.numIters > 1e5:
            explorationProb = explorationProb / math.log(self.numIters - 100000 + 1)

        if explore and random.random() < explorationProb:
            return random.choice(self.actions)
        
        return max(self.actions, key=lambda action: self.Q[(state, action)])

    def getStepSize(self) -> float:
        return 0.1

    def incorporateFeedback(self, state: StateT, action: ActionT, reward: float, nextState: StateT, terminal: bool) -> None:
        step_size = self.getStepSize()
        future_val = 0 if terminal else max(self.Q[(nextState, a)] for a in self.actions)
        self.Q[(state, action)] += step_size * (reward + self.discount * future_val - self.Q[(state, action)])
