from .base import ReportingMethod
from ..anneal import AnnealOutputs

import numpy as np


def _snapshot_value(target, outs) -> np.float64:
    matching = np.all(outs == target, axis=1)
    percentage = np.sum(matching) / (len(outs) - 4)
    return percentage


class Target(ReportingMethod):
    """
    Reporting method:

    Given some array of tile names, check what percentage of scaffolds have 
    exactly that configuration.

    For example, if we have an sdc system of length two, and we use the repoorter
    method Target(["0A0", "1B1"]), then we will see what percentage of the
    scaffolds had the strand named "0A0" on the input domain, and the strand named
    "1B1" in the domain to its right.
    """

    desc = "Target"

    def __init__(self, target_names: list[str]):
        self.target_names = target_names

    def reporter_method(self, anneal_outp: AnnealOutputs):
        system = anneal_outp.system
        target_ids = np.array([0, 0] + [system.tile_number_from_name(
            name) for name in self.target_names] + [0, 0])

        target_percentage = np.zeros(len(anneal_outp.canvas_arr))
        for i, snapshot in enumerate(anneal_outp.canvas_arr):
            target_percentage[i] = _snapshot_value(target_ids, snapshot)

        return target_percentage
