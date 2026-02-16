from .base import ReportingMethod
from ..anneal import AnnealOutputs


class StrandIsAtPosition(ReportingMethod):
    """
    Reporting method:

    Check if a given strand (by name) is in the expected scaffold position.

    If the scaffold looks like [A*, B*, C*, D*], and you want to check that a
    strand witht name "0A0" has attacheed to A*, then use:
    StrandIsAtPosition("0A0", 0).

    If you want to check that the last scaffold domain has "1D1" attached, then
    you may use StrandIsAtPosition("1D1", -1).
    """

    desc = "Strand is in desired position"

    def __init__(self, strand_name, position):
        if position < 0:
            # Sub two to account for the fact that it starts with None, None
            self.position = position - 2
        else:
            # Add two to account for the fact that it starts with None, None
            self.position = position + 2

        self.strand_name = strand_name

    def reporter_method(self, anneal_outp: AnnealOutputs):
        # The length of the scaffold -- Minus four since the scaffold (under
        # the hood) must start with two None positions, and end in two None
        # positions
        scaffold_len = len(anneal_outp.canvas_arr[0]) - 4
        rgrows = anneal_outp.system.rgrow_system

        # Check the percentage quencher_strand attached
        strand_index = rgrows.tile_number_from_name(
            self.strand_name
        )

        percentage_match = (
            (
                anneal_outp.canvas_arr[:, :, self.position]
                == strand_index
            )
        ).sum(axis=-1) / scaffold_len

        return percentage_match
