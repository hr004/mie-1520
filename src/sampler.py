import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class GreedySampling:
    def sample(self, decoder_outputs):
        _, topi = decoder_outputs.topk(1)
        return topi


class TemperatureSampling:
    def __init__(self, temperature=0.5):
        self.temperature = temperature

    def sample(self, decoder_outputs):
        return Categorical(logits=decoder_outputs / self.temperature).sample()


class TopKSampling:
    def sample(self):
        pass


class NucleusSampling:
    def __init__(self, p):
        self.p = p

    def sample(self, decoder_outputs):
        sorted, indices = torch.sort(decoder_outputs, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted, dim=-1), dim=-1)
        mask = cumulative_probs < self.p
        mask = torch.cat(
            [mask, torch.ones_like(mask[:, -1:], dtype=torch.bool)], dim=-1
        )
        mask = mask[:, :-1]
        indices = indices[mask]
        return indices
