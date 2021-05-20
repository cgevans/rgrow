from typing import List, Optional, Tuple, Union, cast, Callable
import statsmodels
import statsmodels.stats.proportion
from numpy import ndarray
import numpy as np
import pandas as pd
from . import rgrow as rg
import dataclasses


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
                      alignment_function: Callable[
                          [ndarray], ndarray]) -> None:
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
    def iter_configs(self) -> None:
        pass

    @property
    def has_all_configs(self) -> bool:
        return len(self.assemblies) > 1

    def seeds_of_trajectories(self, system: rg.StaticKTAMPeriodic) -> pd.DataFrame:
        trajs = self.trajectory_configs

        seeds = []
        seedinfos = []
        for trajectory in trajs:
            seed, seedinfo = trajectory_seed(system, trajectory)
            seeds.append(seed)
            seedinfos.append(seedinfo)

        p = pd.DataFrame(seedinfos)
        p['config'] = seeds

        return p

    @property
    def tuple(self) -> Tuple[float, float, ndarray, List[ndarray], ndarray,
                             ndarray, ndarray, List[List[int]]]:
        return (self.nucleation_rate,
                self.dimerization_rate,
                self.forward_probability,
                self.assemblies,
                self.num_configs_per_surface,
                self.num_trials_per_surface,
                self.assembly_size_per_surface,
                self.previous_configs)

    def __getitem__(self, idx: int) -> Union[float, ndarray, List[ndarray],
                                             List[List[int]]]:
        """Fakes the old return tuple"""
        return self.tuple[idx]


def trajectory_seed(system, trajectory) -> Tuple[ndarray, pd.DataFrame]:
    """Given a system and trajectory (of configurations), find the seed
    (committor closest to 0.5)"""

    trajs = pd.DataFrame([committor_mid(system, config, ci_width=0.10) for config in trajectory],
                         columns=[
                         "in", "side", "c", "clow", "chigh", "succ", "trials"])

    seed_idx = np.abs(trajs.loc[:, "c"] - 0.5).argmin()

    return (trajectory[seed_idx], trajs.loc[seed_idx, :])


def committor(system, config, state_type=rg.StateKTAMPeriodic, ci_width: float = 0.05,
              ci_pct: float = 0.95, min_trials: float = 10) -> Tuple[float, float,
                                                                     float, int, int]:
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


def committor_mid(system, config, state_type=rg.StateKTAMPeriodic,
                  min=0.4, max=0.6, ci_width=0.05, ci_pct=0.95) -> Tuple[bool,
                                                                         Optional[bool],
                                                                         float,
                                                                         float, float,
                                                                         int, int]:
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
        if ci[1]-ci[0] <= ci_width and trials > 4:
            # print(f"Finished after {trials}: p = {successes/trials} {ci}")
            return (True, None, successes/trials, ci[0], ci[1],
                    successes, trials)
        elif (ci[1] < min) or (ci[0] > max):
            return (False, ci[0] > max, successes/trials, ci[0],
                    ci[1], successes, trials)
