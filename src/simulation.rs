//$ Simulations hold both a model and a state, so that they can be handled without knowing the specific model, state, or canvas being used.

use ndarray::ArrayView2;
use rand::prelude::SmallRng;

use crate::base::{NumEvents, NumTiles};
use crate::state::{State, StateStatus};
use crate::{
    base::Tile,
    state::{NullStateTracker, StateTracked},
    system::{StepOutcome, System},
};

struct EvolveBounds {
    events: Option<NumEvents>,
    time: Option<f64>,
    size_min: Option<NumTiles>,
    size_max: Option<NumTiles>,
}

enum EvolveOutcome {
    Events,
    Time,
    SizeMin,
    SizeMax,
    NoStep(StepOutcome),
}

trait CanvasArray {
    fn as_array(&self) -> ArrayView2<Tile>;
}

/// The Simulation trait is designed for holding a single state.
trait Simulation: CanvasArray {
    fn take_step(&mut self, max_time: f64) -> StepOutcome;
    fn evolve(&mut self, bounds: EvolveBounds) -> EvolveOutcome;
}

struct RefSim<'a, St: State + StateTracked<NullStateTracker>, Sy: System<St>> {
    system: &'a Sy,
    state: St,
    rng: SmallRng,
}

impl<'a, St: State + StateTracked<NullStateTracker>, Sy: System<St>> CanvasArray
    for RefSim<'a, St, Sy>
{
    fn as_array(&self) -> ArrayView2<Tile> {
        todo!()
    }
}

impl<'a, St: State + StateTracked<NullStateTracker>, Sy: System<St>> Simulation
    for RefSim<'a, St, Sy>
{
    fn take_step(&mut self, max_time: f64) -> StepOutcome {
        self.system
            .state_step(&mut self.state, &mut self.rng, max_time)
    }

    fn evolve(&mut self, bounds: EvolveBounds) -> EvolveOutcome {
        todo!()
    }
}
