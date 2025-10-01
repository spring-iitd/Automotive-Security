from ..attack import StatisticalAttack
import numpy as np

class IntervalStatisticalAttack(StatisticalAttack):
    attack_params = ["delta"]

    def __init__(self, delta: float = 0.05):
        self.delta = delta
        super().__init__()

    def apply(self, frames: list[dict], **kwargs) -> list[dict]:
        """
        Scale inter-arrival times by a random factor.
        """
        adv_frames = []
        if not frames:
            return adv_frames

        adv_frames.append(frames[0].copy())  # first frame unchanged

        for i in range(1, len(frames)):
            prev = adv_frames[-1]
            f = frames[i].copy()

            # compute inter-arrival time
            interval = f["timestamp"] - prev["timestamp"]

            # perturb interval
            factor = 1 + np.random.uniform(-self.delta, self.delta)
            new_interval = max(0, interval * factor)

            f["timestamp"] = prev["timestamp"] + new_interval
            adv_frames.append(f)

        return adv_frames
