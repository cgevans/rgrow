from pathlib import Path

from platformdirs import user_data_dir

from rgrow.sdc import SDCStrand
from rgrow.sdc.graphs import Graphing
from rgrow.sdc.reporter_methods.strand_at import StrandIsAtPosition
from rgrow.sdc.sdc import SDCParams, SDC
from rgrow.sdc.anneal import Anneal, AnnealOutputs

MIN, HOUR = 60, 60

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

bases = ["A", "B", "C", "D", "E"]

input_tiles = {
    0: SDCStrand(
        left_glue=None,
        right_glue="0'*",
        btm_glue=f"A",
        name=f"A0"
    ),
    1: SDCStrand(
        left_glue=None,
        right_glue="1'*",
        btm_glue=f"A",
        name=f"A1"
    )
}


def make_bitcopy_system(
        input_strand: SDCStrand,
        length: int = 5,
        complexity: int = 2,
):
    """
    Make a system with the *exact* same parameters as the paper.
    """
    scaffold = [f"{b}*" for b in bases][:length]

    strands = [input_strand]
    for i in range(1, length):
        prime_l = "" if i % 2 == 0 else "'"
        prime_r = "" if i % 2 == 1 else "'"
        strands.extend([
            SDCStrand(
                left_glue=f"{inp}{prime_l}",
                right_glue=f"{inp}{prime_r}*",
                btm_glue=f"{bases[i]}",
                name=f"{bases[i]}{inp}"
            ) for inp in range(0, complexity)
        ])

    return SDCParams(
        1e6, 80.0, sequences, scaffold, strands)
    # glue_dg_s=sequences, strands=strands, scaffold=scaffold)


def superfast_anneal():
    """
        Superfast anneal as defined in page 41 of the paper SI.
    """
    pass


def run_system_and_save_output(input_bit: int = 0, file_name: str | None = None,
                               overwrite: bool = False) -> AnnealOutputs:
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
        input_strand=input_tiles[input_bit],
        length=5, complexity=2)
    system = SDC(system_params, name=f"Bit Copy on input {input_bit}")
    anneal = Anneal.standard_long_anneal(scaffold_count=1000)
    anneal_outputs = system.run_anneal(anneal)

    # No need to save the data
    if file_name is None:
        return anneal_outputs

    # Try to save the data
    anneal_outputs.save_data(file_name)
    return anneal_outputs


if __name__ == "__main__":
    # Run the simulations if they have not already been saved
    std0 = run_system_and_save_output(file_name="sdc_short_std_anneal_0.pkl", input_bit=0)
    std1 = run_system_and_save_output(file_name="sdc_short_std_anneal_1.pkl", input_bit=1)

    # Plot the simulations
    # Here we will plot what percentage of the tiles contained 0 at each spot
    g = Graphing("Bit Copy Experimental System")
    for n, p in [("A0", 0), ("B0", 1), ("C0", 2), ("D0", 3)]:
        g.add_line(std0, StrandIsAtPosition(n, p), label=f"% of {n} at {p}")
        g.add_line(std1, StrandIsAtPosition(n, p), label=f"% of {n} at {p}")
    g.end("/home/angelcr/work/rgrow/py-rgrow/examples/sdc_bitcopy/acc.svg")
