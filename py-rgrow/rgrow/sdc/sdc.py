from typing import Mapping
import rgrow as rg
import numpy as np
import yaml

from .anneal import Anneal, AnnealOutputs
import tqdm
import dataclasses
from dataclasses import field
import json
from .strand import SDCStrand
from .reporter_methods import ReportingMethod  # noqa: F401

from rgrow import _system_plot_canvas

@dataclasses.dataclass
class SDCParams:
    """
    Holds all input parameters required to define an SDC system.

    Attributes
    ----------
    k_f : float
        Forward reaction rate constant.
    temperature : float
        Initial simulation temperature in Celsius.
    glue_dg_s : Mapping[str | tuple[str, str], tuple[float, float] | str]
        Dictionary of glue strengths (ΔG, ΔS), indexed by name or tuple of names.
    scaffold : list[str | None]
        Ordered list of glue names representing the scaffold.
    strands : list[SDCStrand]
        List of all strands in solution.
    scaffold_concentration : float
        Effective molar concentration of the scaffold.
    k_n : float
    k_c : float
    junction_penalty_dg : float | None
        Optional ΔG penalty for forming junctions (in kcal/mol).
    junction_penalty_ds : float | None
        Optional ΔS penalty for forming junctions (in kcal/mol/K).
    """

    k_f: float = 1.0e6
    temperature: float = 80.0
    glue_dg_s: (
        Mapping[str | tuple[str, str], tuple[float, float] | str]
        | Mapping[str, tuple[float, float] | str]
        | Mapping[tuple[str, str], tuple[float, float] | str]
    ) = field(default_factory=dict)
    scaffold: list[str | None] = field(
        default_factory=list
    )  # | list[list[str | None]] # FIXME: can't deal with typing for this
    strands: list[SDCStrand] = field(default_factory=list)
    quencher_concentration: float = 0.0
    quencher_name: str | None = None
    fluorophore_concentration: float = 0.0
    reporter_name: str | None = None
    scaffold_concentration: float = 1e-100
    k_n: float = 0.0
    k_c: float = 0.0
    junction_penalty_dg: float | None = None
    junction_penalty_ds: float | None = None

    def __post_init__(self) -> None:
        self.scaffold = [None, None] + self.scaffold + [None, None]

    def __str__(self) -> str:
        strands_info = ""
        for strand in self.strands:
            strands_info += "\n\t" + strand.__str__()
        return f"Forward Rate: {self.k_f}\nStrands: {strands_info}\nScaffold: " + ", ".join(x if x is not None else 'None' for x in self.scaffold[2:-2])

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def write_json(self, filename: str) -> None:
        with open(filename, "w+") as f:
            json.dump(self.to_dict(), f)

    def write_yaml(self, filename: str) -> None:
        with open(filename, "w+") as f:
            yaml.dump(self, f)

    @classmethod
    def from_dict(cls, d: dict) -> "SDCParams":
        if "strands" in d:
            d["strands"] = [SDCStrand(**strand) for strand in d["strands"]]
        if "scaffold" in d:
            if (
                d["scaffold"][0] is None
                and d["scaffold"][1] is None
                and d["scaffold"][-1] is None
                and d["scaffold"][-2] is None
            ):
                d["scaffold"] = d["scaffold"][2:-2]
        return cls(**d)

    @classmethod
    def read_json(cls, filename: str) -> "SDCParams":
        with open(filename, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)

    @classmethod
    def read_yaml(cls, filename: str) -> "SDCParams":
        with open(filename, "r") as f:
            d: SDCParams = yaml.load(f, Loader=yaml.Loader)
        return d



class SDC(rg.rgrow.SDC):
    """
    The actual SDC model

    Attributes
    ----------
    params : SDCParams
        Parameters object used for initialization.
    name : str
        User-provided name of the system.
    """

    params: SDCParams
    name: str | None = None

    def __new__(cls, params: SDCParams, name: str | None = None) -> "SDC":
        self = super().__new__(cls, params)
        self.params = params
        self.name = name
        return self

    @property
    def rgrow_system(self):
        return self

    def __str__(self):
        header_line = f"SDC System {self.name} info:"
        strand_info = f"Parameters:\n{self.params.__str__()}"
        return f"{header_line}\n{strand_info}\n\n"

    def run_anneal(self, anneal: Anneal):
        """"
        Given some Anneal, this will run the system on that anneal.

        This will run the system on a standard square state, with no tracking.

        Parameters
        ----------
        anneal : Anneal

        Returns
        -------
        AnnealOutputs
        """
        times, temperatures = anneal.gen_arrays()
        scaffold_len = len(self.params.scaffold)

        # Here we will keep the state of the canvas at each point in time
        canvas_arr = np.zeros(
            (len(temperatures), anneal.scaffold_count, scaffold_len), dtype=int
        )

        # Now we make a state, and let the time pass ...
        state = rg.State(
            (anneal.scaffold_count, scaffold_len),
            "square",
            "none",
            # The strands defined by the user + null + fluorophore + quencher
            len(self.params.strands) + 3,
        )

        pbar = tqdm.tqdm(enumerate(temperatures), total=len(temperatures))
        for i, t in pbar:
            pbar.set_description(f"{t:7.3f}°C")

            self.set_param("temperature", t)
            self.update_all(state)
            self.evolve(state, for_time=anneal.timestep)
            canvas_arr[i, :, :] = state.canvas_view

        return AnnealOutputs(self, canvas_arr, anneal, state)  # type: ignore[arg-type]

SDC.plot_canvas = _system_plot_canvas  # type: ignore

__all__ = ["SDC", "SDCParams"]