from .base import ReportingMethod
from ..anneal import AnnealOutputs

import rgrow as rg
import numpy as np


class Fluorescence(ReportingMethod):
    """
    Reporting method: Mean fluorescence

    Given the name of the quenching strand and of the reporter strand,
    calculate the mean fluorescence of the system
    """

    def __init__(self, quencher_strand_name, reporter_strand_name):
        self.quencher_n = quencher_strand_name
        self.reporter_n = reporter_strand_name

    desc = "Fluorescence"
    _R = 1.98720425864083e-3
    _BC = 100e-9

    @staticmethod
    def calc_volume(
        temperature, dgds, concentration_strand, concentration_quencher_or_fluorophore
    ):
        """
        Given temperature, dgds, concentration of a strand, and the concentration of
        the fluororophore / quencher, find the volume of [Q - Q'] or [R - R']
        """
        beta = 1 / (Fluorescence._R * (temperature + 273.15))
        delta_g = dgds[0] - dgds[1] * (temperature - 37)
        ep = np.exp(-delta_g * beta)

        minus_b = (
            ep * (concentration_strand +
                  concentration_quencher_or_fluorophore) + 1
        )
        b_squared = (
            ep * (concentration_strand +
                  concentration_quencher_or_fluorophore) + 1
        ) ** 2
        ac = ep * ep * concentration_quencher_or_fluorophore * concentration_strand

        return (minus_b - np.sqrt(b_squared - 4 * ac)) / (2 * ep)

    @staticmethod
    def calc_percentages(
        temperatures, dgds, concentration_strand, concentration_quencher_or_fluorophore
    ):
        """
        Given temperature, dgds, concentration of a strand, and the concentration of
        the fluororophore / quencer, find the volume of [Q - Q'] or [R - R']
        """
        answer = []
        for temp in temperatures:
            answer.append(
                Fluorescence.calc_volume(
                    temp,
                    dgds,
                    concentration_strand,
                    concentration_quencher_or_fluorophore,
                )
                / concentration_quencher_or_fluorophore
            )
        return np.array(answer)

    @staticmethod
    def _percentage_acc(
        anneal_outp: AnnealOutputs, scaffold_position: int, expected_name: str
    ):
        scaffold_len = len(anneal_outp.canvas_arr[0]) - 4
        rgrows = anneal_outp.system.rgrow_system
        expected_index = rgrows.tile_number_from_name(expected_name)
        return (anneal_outp.canvas_arr[:, :, scaffold_position] == expected_index).sum(
            axis=-1
        ) / scaffold_len

    def reporter_method(self, anneal_outp: AnnealOutputs):
        times, temps = anneal_outp.anneal.gen_arrays()

        quencher_position_index = len(anneal_outp.canvas_arr[0][0]) - 4
        reporter_position_index = quencher_position_index + 1

        # Check the percentage quencher_strand and reporter_strand that are
        # attached to the scaffold
        percentage_quencher = Fluorescence._percentage_acc(
            anneal_outp, quencher_position_index, self.quencher_n
        )
        percentage_reporter = Fluorescence._percentage_acc(
            anneal_outp, reporter_position_index, self.reporter_n
        )

        attached_fluo = Fluorescence.calc_percentages(
            temps,
            rg.rgrow.string_dna_dg_ds("ACCATCCCTTCGCATCCCAA"),
            0.9 * Fluorescence._BC,
            0.8 * Fluorescence._BC,
        )

        attached_quench = Fluorescence.calc_percentages(
            temps,
            rg.rgrow.string_dna_dg_ds("ACCATCCCTTCGCATCCCAA"),
            12 * Fluorescence._BC,
            10 * Fluorescence._BC,
        )

        return 1 - (
            percentage_quencher * percentage_reporter * attached_quench * attached_fluo
        )
