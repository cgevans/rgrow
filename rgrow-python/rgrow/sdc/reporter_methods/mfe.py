from .base import ReportingMethod
from ..anneal import AnnealOutputs

import numpy as np


def _snapshot_value(target, outs) -> np.float64:
    matching = np.all(outs == target, axis=1)
    percentage = np.sum(matching) / (len(outs) - 4)
    return percentage


class MinFreeEnergy(ReportingMethod):
    """
    What percentage of the scaffolds were at mfe at each given temperature
    """

    desc = "Minimum Free Energy"

    def reporter_method(self, anneal_outp: AnnealOutputs):
        times, temperatures = anneal_outp.anneal.gen_arrays()
        sdc = anneal_outp.system
        ans = []

        for temperature, outp in zip(temperatures, anneal_outp.canvas_arr):
            sdc.set_param("temperature", float(temperature))
            mfe_config, energy = sdc.mfe_config()
            ans.append(_snapshot_value(mfe_config, outp))

        return ans
