from .base import ReportingMethod
from ..anneal import AnnealOutputs


class ReporterAndComputational(ReportingMethod):
    """
    Reporting method:

    Check that the reporter strand has attached and that the
    last computational strand is correct
    """

    desc = "Reporter + Computation"

    def __init__(self,  quencher_name, reporter_name):
        self.quencher_n = quencher_name
        self.reporter_n = reporter_name

    def reporter_method(self, anneal_outp: AnnealOutputs):
        # This assumes that the scaffold looks like this:
        #
        # None, None, input, C1, C2, ..., Cn, Reporter, None, None
        quencher_position_index = len(anneal_outp.canvas_arr[0][0]) - 4
        reporter_position_index = quencher_position_index + 1

        # The length of the scaffold -- Minus four since the scaffold (under
        # the hood) must start with two None positions, and end in two None
        # positions
        scaffold_len = len(anneal_outp.canvas_arr[0]) - 4
        rgrows = anneal_outp.system.rgrow_system

        # Check the percentage quencher_strand attached
        quencher_strand_index = rgrows.tile_number_from_name(
            self.quencher_n
        )
        percentage_quencher = (
            (
                anneal_outp.canvas_arr[:, :, quencher_position_index]
                == quencher_strand_index
            )
        ).sum(axis=-1) / scaffold_len

        # Check the percentage reporter attached
        reporter_strand_index = rgrows.tile_number_from_name(
            self.reporter_n
        )
        percentage_reporter = (
            (
                anneal_outp.canvas_arr[:, :, reporter_position_index]
                == reporter_strand_index
            )
        ).sum(axis=-1) / scaffold_len

        return percentage_quencher * percentage_reporter
