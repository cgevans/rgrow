from typing import List, Tuple, Union, cast, Callable
import statsmodels
import statsmodels.stats.proportion
from numpy import mintypecode, ndarray
import numpy as np
from . import rgrow as rg
import dataclasses
import scipy.stats  # FIXME: use statsmodels instead


def _ffs_extract_trajectories(backlist: List[List[int]]) -> ndarray:
    ntraj = len(backlist[-1])
    nsurf = len(backlist)

    backarray = list(np.array(x) for x in backlist)

    trajs = np.full((ntraj, nsurf), -1, dtype=int)

    trajs[:, -1] = np.arange(0, ntraj, dtype=int)

    for j in range(nsurf-2, -1, -1):
        trajs[:, j] = backarray[j+1][trajs[:, j+1]]

    return trajs


@dataclasses.dataclass
class FFSResult:
    nucleation_rate: float
    dimerization_rate: float
    forward_probability: ndarray
    assemblies: List[ndarray]
    num_configs_per_surface: ndarray
    num_trials_per_surface: ndarray
    assembly_size_per_surface: ndarray
    previous_configs: List[List[int]]
    aligned_configs: bool = False

    def align_configs(self,
                      alignment_function: Callable[[ndarray], ndarray]) -> None:
        for assemblies_at_surface in self.assemblies:
            for i in range(0, len(assemblies_at_surface)):
                assemblies_at_surface[i] = alignment_function(
                    assemblies_at_surface[i])
        self.aligned_configs = True

    @property
    def trajectory_indices(self) -> ndarray:
        return _ffs_extract_trajectories(self.previous_configs)

    @property
    def trajectory_configs(self) -> ndarray:
        assert self.has_all_configs

        traj_indices = self.trajectory_indices

        tc = np.ndarray(traj_indices.shape + self.canvas_shape, int)

        for traj_idx in range(0, len(traj_indices)):
            for surf_idx in range(0, len(traj_indices[0])):
                tc[traj_idx, surf_idx, :, :] = (self.assemblies[surf_idx][
                                                traj_indices[traj_idx,
                                                             surf_idx]])

        return tc

    @property
    def canvas_shape(self) -> Tuple[int, int]:
        return cast(Tuple[int, int], self.assemblies[0][0].shape)

    @property
    def iter_configs(self):
        pass

    @property
    def has_all_configs(self) -> bool:
        return len(self.assemblies) > 1

    @property
    def tuple(self) -> Tuple[float, float, ndarray, ndarray, ndarray,
                             ndarray, ndarray, List[List[int]]]:
        return (self.nucleation_rate,
                self.dimerization_rate,
                self.forward_probability,
                self.assemblies,
                self.num_configs_per_surface,
                self.num_trials_per_surface,
                self.assembly_size_per_surface,
                self.previous_configs)

    def __getitem__(self, idx: int) -> Union[float, ndarray, List[List[int]]]:
        """Fakes the old return tuple"""
        return self.tuple[idx]


def committor(system, config, state_type=rg.StateKTAMPeriodic, ci_width=0.05,
              ci_pct=0.95, min_trials=10):
    trials = 0
    successes = 0

    while True:
        state = state_type(config.shape[0], system)
        for y in range(0, config.shape[0]):
            for x in range(0, config.shape[1]):
                state.set_point(system, y, x, config[y, x])
        state.evolve_in_size_range(system, 0, 300, 1_000_000)
        trials += 1
        if state.ntiles >= 300:
            successes += 1
        elif state.ntiles != 0:
            raise ValueError
        ci = statsmodels.stats.proportion.proportion_confint(
            successes, trials, 1-ci_pct, method='jeffreys')
        if ci[1]-ci[0] <= ci_width and trials > min_trials:
            # print(f"Finished after {trials}: p = {successes/trials} {ci}")
            return (successes/trials, ci[0], ci[1], successes, trials)


def committor_in(system, config, state_type=rg.StateKTAMPeriodic,
                 min=0.4, max=0.6,
                 conf_in=0.9, conf_out=0.9):
    trials = 0
    successes = 0
    while True:
        state = state_type(config.shape[0], system)
        for y in range(0, config.shape[0]):
            for x in range(0, config.shape[1]):
                state.set_point(system, y, x, config[y, x])
        state.evolve_in_size_range(system, 0, 300, 1_000_000)
        trials += 1
        if state.ntiles >= 300:
            successes += 1
        elif state.ntiles != 0:
            raise ValueError
        cdf = scipy.stats.beta(successes + 1, trials - successes + 1).cdf
        if cdf(0.6) - cdf(0.4) < 0.1:
            # print(f"Finished (out) after {trials}: p = {successes/trials}")
            return False
        elif cdf(0.6) - cdf(0.4) > 0.9:
            # print(f"Finished (in) after {trials}: p = {successes/trials}")
            return (successes, trials)
