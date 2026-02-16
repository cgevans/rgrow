from .base import ReportingMethod
from ..anneal import AnnealOutputs


class LastComputeDomain(ReportingMethod):
    """
    Reporting method:

    Check percentage of scaffolds that contained the strand with a given name
    in the last computational domain.
    """

    desc = "Correct Computation"

    def __init__(self, last_strand_name):
        self.last_strand_name = last_strand_name

    def reporter_method(self, anneal_outp: AnnealOutputs):
        # This assumes that the scaffold looks like this:
        #
        # None, None, input, C1, C2, ..., Cn, Reporter, None, None
        quencher_position_index = len(anneal_outp.canvas_arr[0][0]) - 4

        # The length of the scaffold -- Minus four since the scaffold (under
        # the hood) must start with two None positions, and end in two None
        # positions
        scaffold_len = len(anneal_outp.canvas_arr[0]) - 4

        rgrows = anneal_outp.system.rgrow_system
        quencher_strand_index = rgrows.tile_number_from_name(
            self.last_strand_name)

        percentage_quencher = (
            (
                anneal_outp.canvas_arr[:, :, quencher_position_index]
                == quencher_strand_index
            )
        ).sum(axis=-1) / scaffold_len

        return percentage_quencher
