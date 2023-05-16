use std::ops::DerefMut;

use crate::base::RgrowError;

use crate::state::State;

pub fn run_window(parsed: &crate::tileset::TileSet) -> Result<Box<dyn State>, RgrowError> {
    let sys = parsed.create_dynsystem()?;
    let mut state = parsed.create_state()?;

    sys.setup_state(state.deref_mut())?;

    let bounds = parsed.get_bounds();

    let block = parsed.options.block;

    sys.evolve_in_window(state.deref_mut(), block, bounds)?;

    Ok(state)
}
