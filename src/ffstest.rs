use super::*;
use ndarray::prelude::*;
use ndarray::Zip;

pub struct FFSSystem<S: System<CanvasSquare>> {
    system: S,
    states: Vec<Vec<State2DQT<S,NullStateTracker>>>,
}

impl<S: System<CanvasSquare>> FFSSystem<S> {
    fn gen_first_order(&mut self) -> &mut Self {
        todo!()
    }

    fn gen_next_order(&mut self) -> &mut Self {
        todo!()
    }
}

pub fn ffstest() {
    // We'll start with a 24x24 square.
    let shape = Array2::<usize>::from_shape_fn((24, 24), |(i, j)| i * 24 + j + 1);

    let max: usize = 24 * 24 + 1;
    let GSE = 6.5;
    let GMC = 10.0;

    let ns_strengths = {
        let mut ns = Array2::<f64>::zeros((max, max));
        Zip::from(shape.slice(s![..-1, ..]))
            .and(shape.slice(s![1.., ..]))
            .apply(|&n, &s| {
                ns[(n, s)] = 1.;
            });
        ns
    };

    let we_strengths = {
        let mut we = Array2::<f64>::zeros((max, max));
        Zip::from(shape.slice(s![.., ..-1]))
            .and(shape.slice(s![.., 1..]))
            .apply(|&w, &e| {
                we[(w, e)] = 1.;
            });
        we
    };

    let tile_concs = {
        let mut tc = Array1::<f64>::ones(max);
        tc[0] = 0.0;
        for t in shape.slice(s![10..14, 10..14]).iter() { tc[*t] = 10.0 };
        tc *= f64::exp(-GMC);
        tc
    };


}
