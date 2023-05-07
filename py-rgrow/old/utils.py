from typing import List, Optional, Tuple, Union, cast, Callable
# import statsmodels
# import statsmodels.stats.proportion
from numpy import ndarray
import numpy as np
import pandas as pd
import rgrow.rgrow as rg
import multiprocessing
import multiprocessing.pool
from . import rgrow


def _ffs_extract_trajectories(backlist: List[List[int]]) -> ndarray:
    ntraj = len(backlist[-1])
    nsurf = len(backlist)

    backarray = list(np.array(x) for x in backlist)

    trajs = np.full((ntraj, nsurf), -1, dtype=np.int16)

    trajs[:, -1] = np.arange(0, ntraj, dtype=np.int16)

    for j in range(nsurf-2, -1, -1):
        trajs[:, j] = backarray[j+1][trajs[:, j+1]]

    return trajs


class FFSResult(rgrow.FFSResult):
    # nucleation_rate: float
    # dimerization_rate: float
    # forward_probability: ndarray
    # assemblies: List[ndarray]
    # num_configs_per_surface: ndarray
    # num_trials_per_surface: ndarray
    # assembly_size_per_surface: ndarray
    # previous_configs: List[List[int]]
    # aligned_configs: bool = False
    # _saved_trajectory_configs: Optional[ndarray] = None
    # _saved_trajectory_indices: Optional[ndarray] = None

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
        if self._saved_trajectory_indices is not None:
            return self._saved_trajectory_indices
        else:
            return _ffs_extract_trajectories(self.previous_indices)

    def save_trajectories_and_delete_assemblies(self):
        self._saved_trajectory_configs = self.trajectory_configs
        self._saved_trajectory_indices = self.trajectory_indices
        self.assemblies = []

    @property
    def trajectory_configs(self) -> ndarray:
        if self._saved_trajectory_configs is not None:
            return self._saved_trajectory_configs
        assert self.has_all_configs

        traj_indices = self.trajectory_indices

        # FIXME: might be too small...
        tc = np.array(traj_indices.shape + self.canvas_shape, np.int16)

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

    def seeds_of_trajectories(self, ts: rgrow.TileSet,
                              pool: Optional[multiprocessing.pool.Pool]
                              = None, proppool=False, ci_width=0.1,
                              min=0.4, max=0.6, ci_pct=0.95, max_events=10_000_000) -> pd.DataFrame:
        trajs = self.trajectory_configs

        ts.to_simulation()

        if proppool or (pool is None):
            seeds = []
            seedinfos = []
            for i, trajectory in enumerate(trajs):
                seed, seedinfo = trajectory_seed(system, trajectory, ci_width,
                                                 min, max, ci_pct, max_events, pool)
                print(".", end=None, flush=True)
                seeds.append(seed)
                seedinfos.append(seedinfo)
        else:
            ss = pool.starmap(trajectory_seed, [
                              (system, trajectory, ci_width,
                               min, max, ci_pct, max_events) for trajectory in trajs])
            seeds, seedinfos = zip(*ss)

        p = pd.DataFrame(seedinfos)
        p['config'] = list(seeds)

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

    def __repr__(self) -> str:
        return (f"FFSResult(nucrate={self.nucleation_rate:.4g} M/s, " +
                f"has_all_configs={self.has_all_configs}, " +
                f"num_surfaces={len(self.assembly_size_per_surface)}, " +
                f"num_trajectories={len(self.previous_indices[-1])}"
                ")")

    def __str__(self) -> str:
        return repr(self)


def trajectory_seed(sim, trajectory,
                    ci_width=0.1,
                    min=0.4, max=0.6, ci_pct=0.95, max_events=10_000_000,
                    pool: multiprocessing.pool.Pool = None,) -> Tuple[ndarray, pd.Series]:
    """Given a sim and trajectory (of configurations), find the seed
    (committor closest to 0.5)"""

    if pool is None:
        trajs = pd.DataFrame([committor_mid(sim, config, ci_width=0.10)
                              for config in trajectory],
                             columns=[
            "in", "side", "c", "clow", "chigh", "succ", "trials"])
    else:
        trres = pool.starmap(
            committor_mid, [(sim, config, ci_width, min, max, ci_pct, rg.StaticKTAMPeriodic, max_events) for config in trajectory])
        trajs = pd.DataFrame(trres, columns=[
            "in", "side", "c", "clow", "chigh", "succ", "trials"])

    seed_idx = np.abs(trajs.loc[:, "c"] - 0.5).argmin()

    return (trajectory[seed_idx], trajs.loc[seed_idx, :])


def committor(sim: rgrow.Simulation, config, ci_width: float = 0.05, state_type=rg.StateKTAMPeriodic,
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
        ci: Tuple[float, float] = statsmodels.stats.proportion.proportion_confint(
            successes, trials, 1-ci_pct, method='jeffreys')  # type:ignore
        if ci[1]-ci[0] <= ci_width and trials > min_trials:
            # print(f"Finished after {trials}: p = {successes/trials} {ci}")
            return (successes/trials, ci[0], ci[1], successes, trials)


def committor_mid(sim: rgrow.Simulation, config, ci_width=0.05,
                  min=0.4, max=0.6, ci_pct=0.95,
                  state_type=rg.StateKTAMPeriodic,
                  max_events=10_000_000) -> Tuple[bool,
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
        state.evolve_in_size_range(system, 0, 300, max_events)
        trials += 1
        if state.ntiles >= 300:
            successes += 1
        elif state.ntiles != 0:
            raise ValueError
        ci: Tuple[float, float] = statsmodels.stats.proportion.proportion_confint(
            successes, trials, 1-ci_pct, method='jeffreys')  # type:ignore
        if ci[1]-ci[0] <= ci_width and trials > 4:
            # print(f"Finished after {trials}: p = {successes/trials} {ci}")
            return (True, None, successes/trials, ci[0], ci[1],
                    successes, trials)
        elif (ci[1] < min) or (ci[0] > max):
            return (False, ci[0] > max, successes/trials, ci[0],
                    ci[1], successes, trials)


def ffs_nucleation(system,
                   max_size=200,
                   canvas_size=32,
                   keep_surface_configs=False,
                   cutoff_probability=0.99,
                   cutoff_surfaces=4,
                   min_configs=1_000,
                   varpermean2=1e-4,
                   min_cutoff_size=30,
                   _surface_size_step=1,
                   _surface_init_size=3,
                   _max_init_events=10_000,
                   _max_subseq_events=1_000_000):
    if isinstance(system, rg.StaticKTAMPeriodic):
        restuple = rg.ffs_run_final_p_cvar_cut(
            system,
            varpermean2,
            min_configs,
            max_size,
            cutoff_probability,
            cutoff_surfaces,
            min_cutoff_size,
            canvas_size,
            _max_init_events,
            _max_subseq_events,
            _surface_init_size,
            _surface_size_step,
            keep_surface_configs
        )
        return FFSResult(*restuple, system=system)
    elif isinstance(system, rg.NewKTAMPeriodic):
        restuple = rg.ffs_run_final_p_cvar_cut_new(
            system,
            varpermean2,
            min_configs,
            max_size,
            cutoff_probability,
            cutoff_surfaces,
            min_cutoff_size,
            canvas_size,
            _max_init_events,
            _max_subseq_events,
            _surface_init_size,
            _surface_size_step,
            keep_surface_configs
        )
        return FFSResult(*restuple, system=system)
    else:
        raise TypeError(f"Can't handle system type {type(system)}.")
