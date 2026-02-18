# Implementing a New Model

This is current as of 2026-03-18, but may rapidly change.  In particular, I am not very happy with how difficult this process would make it for someone to simply make a new model and use it, preferably even in Python, without adding it to rgrow itself, but just using rgrow as a dependency. - CGE

Adding a new model involves:

1. Implementing two traits: `TileBondInfo` and `System`.
2. Registering the model in `SystemEnum`, and (optionally) Python bindings.
3. Adding some way to construct the model, and (optionally) making that accessible te Python.

The `SDC1DBindReplace` model is particularly simple, and the source for it may be useful.  

## Core model implementation

Models are generally stored in src/models/, in separate files.  Then, you'll need to implement:

- `TileBondInfo`: these methods should be reasonably straightforward, but there isn't an easy way to automatically implement them (TODO: some actually can be...).
- `System`: this trait has the actual simulation code:
   - `system_info`: straightforward.
   - `event_rate_at_point`: given a point on the canvas, return the total rate of all possible events at that point. This is primarily used to update rate arrays.
   - `choose_event_at_point`: given a point and the remaining random rate variable choice, choose amongst the possible events at that point.  The structure should mirror `event_rate_at_point`: iterate through the same events, preferably in the same order, subtracting each event's rate from `acc`, and return the event when `acc` drops to zero or below.  If the rates from these two disagree, panics or other weirdness may occur, so consider calling the same internal functions from both.  
   - `update_after_event`: after an event is performed (the state is already mutated), update the points in the state whose rates may have changed.  There is the `State::update_multiple` method that helps with updating multiple points in a state.
   - `seed_locs`: return the initial tile placements for a new state. Return an empty `Vec` if the model has no concept of a seed. (TODO: this may be given a default implementation or even removed)
   - `calc_mismatch_locations`: return an array indicating mismatches. Can be left as `todo!()` initially if not needed.

## Connecting to the rest of rgrow

Three files need changes to integrate the model into rgrow:

### `rgrow-rust/src/models/mod.rs`

Add the module declaration:

```rust
pub mod sdc1d_bindreplace;
```

### `rgrow-rust/src/system.rs`

Add two things:

1. An import at the top:

    ```rust
    use crate::models::sdc1d_bindreplace::SDC1DBindReplace;
    ```

2. A variant in the `SystemEnum` enum (which uses `enum_dispatch`):

    ```rust
    #[enum_dispatch(DynSystem, TileBondInfo)]
    pub enum SystemEnum {
        KTAM,
        OldKTAM,
        ATAM,
        SDC,
        SDC1DBindReplace,  // <-- add this
        KBlock,
    }
    ```

3. If Python bindings are enabled, add the conversion arm in `IntoPyObject`:

    ```rust
    SystemEnum::SDC1DBindReplace(x) => x.into_bound_py_any(py),
    ```

### `rgrow-python/src/lib.rs` (for Python bindings)

Export the class:

```rust
#[pymodule_export]
use rgrow::models::sdc1d_bindreplace::SDC1DBindReplace;
```

## Add Python bindings (optional)

If you want the model to be constructable from Python, add a `#[pymethods]` block, for example:

```rust
#[cfg(feature = "python")]
#[pymethods]
impl SDC1DBindReplace {
    #[new]
    fn py_new(params: SDCParams) -> Self {
        Self::from_params(params)
    }
}
```
