# SDC2DSquare — 2D Scaffolded DNA Computer

The `rgrow.sdc2d` module provides a 2D scaffolded DNA tile assembly model on a
square grid. Strands have five glues (North, East, South, West, Bottom); the
lateral glue order matches kTAM (clockwise from north), with the bottom glue
facing the scaffold. All four lateral edges and the scaffold edge share a
single glue namespace.

The model uses unitful (kcal/mol, Molar, Kelvin) thermodynamics with per-glue
`(ΔG_37, ΔS)` parameters, mirroring SDC1D.

## SDC2DParams

::: rgrow.sdc2d.SDC2DParams
    options:
      show_source: true

## SDC2DStrand

::: rgrow.sdc2d.SDC2DStrand
    options:
      show_source: true

## SDC2DSquare

Construct an `SDC2DSquare` system from an `SDC2DParams` instance. In addition
to the kinetic simulation interface (`evolve`, `setup_state`, `get_param`,
`set_param`, etc.), `SDC2DSquare` exposes exact finite-grid thermodynamics
methods. These are exact but exponential in the smaller scaffold dimension.

::: rgrow.sdc2d.SDC2DSquare
    options:
      show_source: false
      members_order: source
