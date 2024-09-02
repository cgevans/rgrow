from .base import ReportingMethod
from ..anneal import AnnealOutputs

import numpy as np


class Target(ReportingMethod):
    """
    Reporting method:

    Check if the scaffold matches some array exactly
    """

    desc = "Target"

    def __init__(self, target_names: list[str]):
        self.target_names = target_names

    def reporter_method(self, anneal_outp: AnnealOutputs):
        target_ids = np.array(
            (
                [0, 0]
                + [
                    anneal_outp.system.rgrow_system.tile_number_from_name(name)
                    for name in self.target_names
                ]
                + [0, 0]
            )
        )

        target_percentage = np.zeros(len(anneal_outp.canvas_arr))
        i = 0
        for snapshot in anneal_outp.canvas_arr:
            correct_target = 0
            for scaffold in snapshot:
                if (scaffold == target_ids).all():
                    correct_target += 1

            target_percentage[i] = correct_target / len(snapshot)
            i += 1

        return target_percentage
