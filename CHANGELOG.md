# 0.20.0

- Changed names from 'covers' to 'blockers' to fit papers.
- Initial implementation of unitful calculations, which should reduce likelihood of bugs, while hopefully being zero-cost for performance.

# 0.19.4

- kcov:
  - Changes to cover+tile rates in the low-cover-concentration case.
- Version used for DNA31 submission.

# 0.19.3

- Numerous covers/covers model improvements and fixes:
  - tile attachment + covers fixes.
  - full concentration calculations for tile-cover complexes (still one tile per cover)
  - add option to allow/disallow partially-blocked attachments
  - rearrange tubes more

# 0.19.2

- Modified tube canvas format, now zig-zagging in the j direction rather than being diagonal.
  The older diagonal format is now TubeDiagonals.

# 0.19.1

- Some convenience functions for covers.

# 0.19.0

- Improvements to SDC support.
- Initial covers model support.

# 0.18.0

- Initial 1D SDC support.

# 0.17.0

- Significant change in Python organization: there are now model-specific System class, rather than one System class.  Tileset definitions will return a specific class, and classes should eventually be able to be created directly.  This should make it much easier to add ways to view and manipulate system objects in model-dependent ways.
- Improvements to Python docstrings and type stubs.
- Numerous code rearrangements for Python.

# 0.16.0

- Improvements to FFS code organization.
- Support for writing FFS results to files.
- Improved FFS polars dataframe output creation:
  - configs now include tracking information, if present.
  - configs no longer breaks if keep_configs is false (shows last configurations)
- Fix Python mismatch display code (some mismatches were not shown).

# 0.15.0

- Start order tracking at 1 (0 is a site that was never filled).
- Added LastAttachTime tracker: keeps track of the last attachment/change event in a location.
- Added PrintEvent tracker for debugging, which prints every event.
- Implement dimer chunk detachment properly for kTAM model (already in oldkTAM).

# 0.14.1

- Fix dimer formation rates in kTAM model.
- Fix k_f specification in kTAM model.

# 0.14.0

- Allow state references to be cloned into normal states (useful for FFS results).
- Add Python typing for FFS results.
- Distinguish stored states and number of configurations for FFS surfaces.
- Make FFS state references more useful.
- Fix rayon use in Python (parallel execution).

# 0.13.1

- Allow Python compilation without ui.
- Fix attachment order tracking.
- Fix tube canvas N/S boundaries.
- PyO3 0.21, with new Bound API.
- Fix python tests.

# 0.13.0

- Restructured API to allow separate, easier-to-use Systems and States.
- Refactored Python API.
- Using enum_dispatch rather than dynamic traits and boxes.
- Added Python plotting function.
- Improved Python typing.
- Fixed tube canvas implementation.
- Added more mutable parameters to kTAM.

# 0.12.2

- Python tests.
- Mismatch calculations for aTAM.
- Add several convenience functions to Python API.
- Improve documentation.

# 0.12.1

Version bump to fix some release automation problems.

# 0.12.0

The version number for Rust has been increased to give consistent version numbering for both Rust and Python.

- Combined the pyrgrow and rgrow repositories, to ease development and ensure version matching.
- Initial support for mid-simulation system parameter changes.
- RateStore multi-update optimizations.
- Significant refactoring of Python API, and more documentation.
- Calculation of mismatches and mismatch locations.
- Display improvements, with mismatches shown at higher display scales, and dynamic scaling.
- Parallel evolution improvements.
- Make UI no longer default for Python.

# 0.7.1

- Type improvements.
- Optimizations. 
- rng instead of SmallRng (easier code, no major performance change)
- Tiles as u32.
- Initial cffi interface.

# 0.7.0

- Improved UI.
- Improved errors.
- Improved evolve bounds specifications.
- Partially merged some Python interface items into this crate.
- Improved xgrow format support.
- Added tests, including the xgrow format.
- Included xgrow example files, and tested all of them.
- Other things I've probably forgotten.

# 0.6.0

- First released version.
- New kTAM model, supporting duples and changes to the system.
- New aTAM model, based on the new kTAM model.
- Simulation trait, allowing model-agnostic code.
- TileSet objects, allowing parsing of tileset inputs.
