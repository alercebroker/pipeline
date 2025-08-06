
class NoImprovementStopper:
    def __init__(self, num_steps: int, mode: str = 'max'):
        self.num_steps = num_steps
        self.mode = mode
        self.historic_best = -float('inf') if mode == 'max' else float('inf')
        self.steps_without_improvement = 0

    def should_break(self, current_value):
        if (self.mode == 'max' and current_value > self.historic_best) or \
           (self.mode == 'min' and current_value < self.historic_best):
            self.historic_best = current_value
            self.steps_without_improvement = 0
            return False
        else:
            self.steps_without_improvement += 1

        return self.steps_without_improvement >= self.num_steps
        