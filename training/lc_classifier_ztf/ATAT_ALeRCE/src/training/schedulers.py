import torch
import numpy as np


class CosineDecayWithWarmup:
    def __init__(
        self, warmup_steps, total_steps, warmup_start_lr, warmup_target_lr, alpha=0.05
    ):
        self.warmup_steps = torch.tensor(warmup_steps)
        self.total_steps = torch.tensor(total_steps)
        self.warmup_start_lr = warmup_start_lr
        self.warmup_target_lr = warmup_target_lr
        self.alpha = alpha

    def __call__(self, current_step):
        current_step = torch.tensor(current_step)
        if current_step < self.warmup_steps:
            warmup_progress = current_step / self.warmup_steps
            lr = (
                self.warmup_target_lr - self.warmup_start_lr
            ) * warmup_progress + self.warmup_start_lr
        else:
            decay_steps = torch.tensor(self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (
                1
                + torch.cos(torch.pi * (current_step - self.warmup_steps) / decay_steps)
            )
            decayed = (1 - self.alpha) * cosine_decay + self.alpha
            lr = (
                self.warmup_target_lr - self.warmup_start_lr
            ) * decayed + self.warmup_start_lr
        return lr


def cosine_decay_ireyes(
    epoch: int, warm_up_epochs: int, decay_steps: int, alpha: float
):
    if epoch < warm_up_epochs:
        return (epoch + 1) / warm_up_epochs

    factor = np.cos(epoch / decay_steps * np.pi / 2)
    factor = max(factor, alpha)
    return factor


if __name__ == "__main__":
    import numpy as np

    for i in range(100):
        print(cosine_decay_ireyes(i, 10, 90, 0.05))
