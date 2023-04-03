# 0.8.0 (dev)

- Initial support for mid-simulation system parameter changes.
- RateStore multi-update optimizations.

# 0.7.1

- Type improvements.
- Optimizations. 
- thread_rng instead of SmallRng (easier code, no major performance change)
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
