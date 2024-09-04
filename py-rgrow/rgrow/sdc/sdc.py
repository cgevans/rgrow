from typing import Mapping
import rgrow as rg
import numpy as np

from .anneal import Anneal, AnnealOutputs
import tqdm
import dataclasses
import json
from .strand import SDCStrand
from .reporter_methods import ReportingMethod  # noqa: F401


@dataclasses.dataclass
class SDCParams:
    """
    Parameters used to create an SDC system
    """

    k_f: float
    temperature: float
    glue_dg_s: (
        Mapping[str | tuple[str, str], tuple[float, float] | str]
        | Mapping[str, tuple[float, float] | str]
        | Mapping[tuple[str, str], tuple[float, float] | str]
    )
    scaffold: list[
        str | None
    ]  # | list[list[str | None]] # FIXME: can't deal with typing for this
    strands: list[SDCStrand]
    scaffold_concentration: float = 1e-100
    k_n: float = 0.0
    k_c: float = 0.0

    def __post_init__(self) -> None:
        self.scaffold = [None, None] + self.scaffold + [None, None]

    def __str__(self) -> str:
        strands_info = ""
        for strand in self.strands:
            strands_info += "\n\t" + strand.__str__()
        return f"Forward Rate: {self.k_f}\nStrands: {strands_info}\nScaffold: {', '.join(x if x is not None else "None" for x in self.scaffold[2:-2])}"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def write_json(self, filename: str) -> None:
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f)

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


class SDC(rg.rgrow.SDC):
    params: SDCParams
    # Name of the system -- Used for plotting
    name: str

    def __new__(cls, params, system_name):
        self = super().__new__(cls, params)
        self.params = params
        self.name = system_name
        return self

    @property
    def rgrow_system(self):
        return self

    def __str__(self):
        header_line = f"SDC System {self.name} info:"
        strand_info = f"Parameters:\n{self.params.__str__()}"
        return f"{header_line}\n{strand_info}\n\n"

    def run_anneal(self, anneal: Anneal):
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
            len(self.params.strands) + 1,
        )

        for i, t in tqdm.tqdm(enumerate(temperatures), total=len(temperatures)):
            self.set_param("temperature", t)
            self.update_all(state)
            self.evolve(state, for_time=anneal.timestep)
            canvas_arr[i, :, :] = state.canvas_view

        return AnnealOutputs(self, anneal, canvas_arr)
