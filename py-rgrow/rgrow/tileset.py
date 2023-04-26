from typing import Any, Mapping, Sequence, TypeAlias
from rgrow.rgrow import PyTile, PyTileSet
from dataclasses import dataclass


class Tile:
    _s: PyTile

    def __init__(
        self,
        edges: list[str | int],
        name: str | None = None,
        stoic: float | None = None,
        color: str | None = None,
    ):
        self._s = PyTile(edges, name, stoic, color)

    def __repr__(self):
        return self._s.__repr__()

    def __str__(self):
        return self._s.__str__()

    @property
    def edges(self) -> list[str | int | None]:
        """
        The glues on the edges of the tile, in clockwise order starting from the North,
        or the North-facing edge furthest to the West if not a single tile.

        Glues should be either strings, integers (starting at 1), or None or 0 to
        refer to a null glue.
        """
        return self._s.edges

    @edges.setter
    def edges(self, edges: list[str | int]):
        self._s.edges = edges

    @property
    def name(self) -> str | None:
        return self._s.name

    @name.setter
    def name(self, name: str | None):
        self._s.name = name

    @property
    def stoic(self) -> float | None:
        return self._s.stoic

    @stoic.setter
    def stoic(self, stoic: float | None):
        self._s.stoic = stoic

    @property
    def color(self) -> str | None:
        return self._s.color

    @color.setter
    def color(self, color: str | None):
        self._s.color = color


@dataclass
class Bond:
    name: str | int
    strength: float


GlueIdent = TypeAlias[str | int]


class TileSet:
    _s: PyTileSet

    def __init__(
        self,
        tiles: Sequence[Tile] = (),
        bonds: Sequence[Bond] = (),
        glues: Sequence[tuple[GlueIdent, GlueIdent, float]] = (),
        options: Mapping[str, Any] = dict(),
    ):  # fixme: don't use {}
        decodedbonds = [(bond.name, bond.strength) for bond in bonds]

        self._s = PyTileSet(list(tiles), decodedbonds, list(glues), options)

    def __repr__(self):
        return self._s.__repr__()

    def __str__(self):
        return self._s.__str__()

    @property
    def tiles(self) -> list[Tile]:
        return self._s.tiles

    @tiles.setter
    def tiles(self, tiles: list[Tile]):
        self._s.tiles = tiles

    @property
    def bonds(self) -> list[Bond]:
        return self._s.bonds

    @bonds.setter
    def bonds(self, bonds: list[Bond]):
        self._s.bonds = bonds
