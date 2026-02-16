from dataclasses import dataclass


@dataclass
class SDCStrand:
    """
        Represents a strand that can bind to the scaffold in the SDC model.

        Parameters
        ----------
        concentration : float
            Concentration of the strand (default is 1 μM = 1e-6 M).
        left_glue : str or None
            Glue name (or `None`) for the left attachment point.
        btm_glue : str or None
            Identifier for the central domain of the strand (typically a base like 'A', 'B', etc.).
        right_glue : str or None
            Glue name (or `None`) for the right attachment point.
        name : str or None
            Human-readable identifier (e.g., compact code like "0A1").
        color : str or None
            Optional color for visualization or grouping.
    """

    concentration: float = 1e-6
    left_glue: str | None = None
    btm_glue: str | None = None
    right_glue: str | None = None
    name: str | None = None
    color: str | None = None

    @classmethod
    def basic_from_string(string_representation: str):
        """
        Given some simple string, generate a strand.

        For example:
        - given "0A0", the strand with left glue 0*, right glue 0, and base A will be generated
        - given "-B-", the strand with no glue on the left and right, and base B will be generated
        """

        return SDCStrand(
            left_glue=f"{string_representation[0]}*",
            btm_glue=string_representation[1],
            right_glue=string_representation[2],
            name=string_representation,
        )

    @classmethod
    def pair_glue_from_string(string_representation: str):
        """
        Given some string, generate a strand.

        This function will only work for simple systems (that is, small systems, with not-so high complexity), the input
        MUST be in the following format: f"{left_glue}{base_character}{right_glue}", where the left and right glue are
        either a number, or '-' if no glue is present, and the base_character is A, or B, ..., or Z.

        Some valid strings would be "0A1", "1B1", "0K4", "-A1", "-E-"
        """

        # An even base will have an even right glue, and an odd left glue. An odd base will have an odd right glue and
        # an even left glue
        even_base = (ord(string_representation[1]) - ord("A")) % 2 == 0

        l_postfix = "e" if even_base else "o"
        r_postfix = "o" if even_base else "e"

        lc = string_representation[0]
        rc = string_representation[2]
        l_glue = None if lc == "-" else f"{lc}{l_postfix}"
        r_glue = None if rc == "-" else f"{rc}{r_postfix}"

        return SDCStrand(
            left_glue=l_glue,
            btm_glue=string_representation[1],
            right_glue=r_glue,
            name=string_representation,
        )

    def __str__(self):
        return f"Strand {self.name} at concentration {self.concentration}"
