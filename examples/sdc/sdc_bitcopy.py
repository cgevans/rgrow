import string
from pathlib import Path

from platformdirs import user_data_dir

from rgrow.sdc import SDCStrand
from rgrow.sdc.graphs import Graphing
from rgrow.sdc.reporter_methods.fluorescence import Fluorescence
from rgrow.sdc.reporter_methods.strand_at import StrandIsAtPosition
from rgrow.sdc.sdc import SDCParams, SDC
from rgrow.sdc.anneal import Anneal, AnnealOutputs

SEC = 1
MIN = 60 * SEC
HOUR = 60 * MIN

# Page 47

# The DNA sequences as in the paper supplement.
# Since we cannot type an overline, we will let overline(0) be defined as 0'
sequences = {
    "A": "TCTTTCCAGAGCCTAATTTGCCAG",
    "B": "AGCGTCCAATACTGCGGAATCGTC",
    "C": "ATAAATATTCATTGAATCCCCCTC",
    "D": "AAATGCTTTAAACAGTTCAGAAAA",
    "E": "AAAGAGGACAGATGAACGGTGTAC",

    "0": "CTCATCCTGACC", "0'": "CCTCTTCTCAGC",
    "1": "TCAACTCCGTTC", "1'": "CATCTCCGATCC",
    "2": "AATGCCACCATT", "2'": "TCTTTCCAAGCC",
    "3": "ACAACCCTTGTC", "3'": "TCAATCCTTGCC",
    "4": "CTGTTCCCAACA", "4'": "CACATCCCTGTT",
    "5": "CACTACCAGTCC", "5'": "CCATGTCCCATT",
    "6": "ACACACACTGTC", "6'": "CAACCAACGTTC",
    "7": "TCACTTTCGTCC", "7'": "TCACACTTCGTC",
}

bases = [char for char in string.ascii_uppercase]


def input_strand(inp: int = 0) -> SDCStrand:
    b = bases[0]
    return SDCStrand(
        left_glue=None,
        right_glue=f"{inp}'*",
        btm_glue=f"{b}",
        name=f"{b}{inp}"
    )


def reporter(base):
    return SDCStrand(
        left_glue=None,
        right_glue=None,
        btm_glue=base,
        name="Rep"
    )


def make_bitcopy_system(
        input_strand: SDCStrand,
        length: int = 5,
        inputs: list[int] = [0, 1]
):
    """
    Make a system with the *exact* same parameters as the paper.
    """
    scaffold = [f"{b}*" for b in bases][:length]

    strands = [input_strand]
    for i in range(1, length - 1):
        prime_l = "" if i % 2 == 0 else "'"
        prime_r = "" if i % 2 == 1 else "'"
        strands.extend([
            SDCStrand(
                left_glue=f"{inp}{prime_l}",
                right_glue=f"{inp}{prime_r}*",
                btm_glue=f"{bases[i]}",
                name=f"{bases[i]}{inp}"
            ) for inp in inputs
        ])
    strands.append(reporter(bases[length - 1]))
    return SDCParams(
        1e6, 80.0, sequences, scaffold, strands)


def superfast_anneal():
    """
        Superfast anneal as defined in page 41 of the paper SI.
    """
    return Anneal(
        initial_hold=10 * SEC,
        initial_tmp=80.0,
        delta_time=40 * SEC,
        final_tmp=45.0,
        final_hold=10 * SEC)


def run_system_and_save_output(input_bit: int = 0, other_bit: int = 1, file_name: str | None = None,
                               overwrite: bool = False,
                               anneal=Anneal.standard_long_anneal(scaffold_count=1000)) -> AnnealOutputs:
    r"""
    Set file_name to None if you don't want the data to be saved, file name is the name of the file, without extension.
    The file will be saved in:

    Linux   -> `/home/user/.local/share/rgrow`
    macOS   -> `/Users/user/Library/Application Support/rgrow`
    Windows -> `C:\Users\user\AppData\Roaming\rgrow`
    """

    # If the file already exits, check if we need to overwrite it or not
    if file_name is not None:
        save_path = Path(user_data_dir("rgrow")) / "sdc" / file_name
        if save_path.exists() and not overwrite:
            print(f"[SKIP] File exists: {save_path}, skipping simulation")
            return AnnealOutputs.load_data(file_name)

    system_params = make_bitcopy_system(
        input_strand=input_strand(input_bit),
        inputs=[input_bit, other_bit],
        length=5)
    system = SDC(system_params, name=f"Bit Copy on input {input_bit}")
    anneal_outputs = system.run_anneal(anneal)

    # No need to save the data
    if file_name is None:
        return anneal_outputs

    # Try to save the data
    anneal_outputs.save_data(file_name)
    return anneal_outputs


def positional_graph(std0: AnnealOutputs, std1: AnnealOutputs, x: str = ""):
    # Plot the simulations
    # Here we will plot what percentage of the tiles contained 0 at each spot
    g = Graphing("Bit Copy Experimental System")
    for n, p in [("A0", 0), ("B0", 1), ("C0", 2), ("D0", 3), ("Rep", 4)]:
        g.add_line(std0, StrandIsAtPosition(n, p), label=f"% of {n} on 0")
        g.add_line(std1, StrandIsAtPosition(n, p), label=f"% of {n} on 2")
    g.end(f"/home/angelcr/work/rgrow/py-rgrow/examples/sdc_bitcopy/{x}acc.svg")


def fluo_graph(std0: AnnealOutputs, std1: AnnealOutputs, x: str = ""):
    # Plot the simulations
    # Here we will plot what percentage of the tiles contained 0 at each spot
    g = Graphing("Bit Copy Experimental System")
    fluo = Fluorescence(quencher_strand_name="D1", reporter_strand_name="Rep")
    g.add_line(std0, fluo, label="Fluo input 0", linestyle='dashed')
    g.add_line(std1, fluo, label="Fluo input 2", linestyle='dashed')
    g.end(f"/home/angelcr/work/rgrow/py-rgrow/examples/sdc_bitcopy/{x}fluo.svg")


if __name__ == "__main__":
    # Run the simulations if they have not already been saved
    std0 = run_system_and_save_output(file_name="sdc_short_std_anneal_0.pkl", input_bit=0, other_bit=1, overwrite=True)
    std1 = run_system_and_save_output(file_name="sdc_short_std_anneal_1.pkl", input_bit=1, other_bit=0, overwrite=True)
    std2 = run_system_and_save_output(file_name="sdc_short_std_anneal_2.pkl", input_bit=2, other_bit=0, overwrite=True)
    fluo_graph(std0, std2)
    positional_graph(std0, std2)

    std0_fast = run_system_and_save_output(file_name="sdc_short_fast_anneal_0.pkl", input_bit=0, overwrite=True,
                                           anneal=superfast_anneal())
    std1_fast = run_system_and_save_output(file_name="sdc_short_fast_anneal_1.pkl", input_bit=2, overwrite=True,
                                           anneal=superfast_anneal())
    fluo_graph(std0_fast, std1_fast, x="fast")
    positional_graph(std0_fast, std1_fast, x="fast")
