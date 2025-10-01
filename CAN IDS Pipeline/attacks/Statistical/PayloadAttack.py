from ..attack import StatisticalAttack
import numpy as np

class PayloadStatisticalAttack(StatisticalAttack):
    attack_params = ["prob"]

    def __init__(self, prob: float = 0.01):
        self.prob = prob
        super().__init__()

    def apply(self, frames: list[dict], **kwargs) -> list[dict]:
        """
        Flip bits in CAN payload with given probability.
        """
        adv_frames = []
        for f in frames:
            new_f = f.copy()
            data = bytearray(f["data"])  # ensure mutable

            for i in range(len(data)):
                for bit in range(8):
                    if np.random.rand() < self.prob:
                        data[i] ^= (1 << bit)

            new_f["data"] = bytes(data)
            adv_frames.append(new_f)
        return adv_frames
