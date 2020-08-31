extern crate ndarray;
extern crate num_traits;
use numpy::IntoPyArray;
use ndarray::prelude::*;
use ndarray::{FoldWhile, Zip};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::prelude::*;
use std::{convert::TryInto};
type NumTiles = usize;
type NumEvents = u64;
type Point = (usize, usize);
type Tile = usize;
type Rate = f64;
type Energy = f64;

pub trait StateEvolve {
    fn evolve_in_size_range(
        &mut self,
        minsize: NumTiles,
        maxsize: NumTiles,
        maxevents: NumEvents,
    ) -> &Self;
}

pub trait StateStep {
    fn take_step(&mut self) -> &Self;
}

pub trait StateCreate<C: Canvas, S: System<C>> {
    fn create(canvas: &Array2<Tile>, sys: &S) -> Self;
}

pub trait StateStatus {
    fn ntiles(&self) -> NumTiles;

    fn total_events(&self) -> NumEvents;

    //fn time(&self) -> Time;

    //fn last_step_time(&self) -> Time;
}

pub trait Canvas {
    unsafe fn uv_n(&self, p: Point) -> Tile;
    unsafe fn uv_e(&self, p: Point) -> Tile;
    unsafe fn uv_s(&self, p: Point) -> Tile;
    unsafe fn uv_w(&self, p: Point) -> Tile;
    unsafe fn uv_p(&self, p: Point) -> Tile;
    fn inbounds(&self, p: Point) -> bool;
}
pub trait System<C: Canvas> {
    fn event_rate_at_point(&self, canvas: &C, p: Point) -> Rate;

    fn choose_event_at_point(&self, canvas: &C, p: Point, acc: Rate) -> Tile;
}

#[derive(Clone)]
pub struct StaticATAM {
    tile_rates: Array1<Rate>,
    strength_ns: Array2<Energy>,
    strength_we: Array2<Energy>,
    tau: Energy
}

#[derive(Clone)]
pub struct StaticKTAM {
    tile_rates: Array1<Rate>,
    energy_ns: Array2<Energy>,
    energy_we: Array2<Energy>,
}

impl StaticATAM {
    pub fn new(
        tile_concs: Array1<f64>,
        tile_edges: Array2<usize>,
        glue_strengths: Array1<f64>,
        tau: f64,
    ) -> Self {
        let ntiles = tile_concs.len();
        assert!(ntiles == tile_edges.nrows());
        let mut strength_we: Array2<Energy> = Array2::zeros((ntiles, ntiles));
        let mut strength_ns: Array2<Energy> = Array2::zeros((ntiles, ntiles));
        for ti1 in 0..ntiles {
            for ti2 in 0..ntiles {
                let t1 = tile_edges.row(ti1);
                let t2 = tile_edges.row(ti2);
                if t1[2] == t2[0] {
                    strength_ns[(ti1, ti2)] = glue_strengths[t1[2]];
                }
                if t1[1] == t2[3] {
                    strength_we[(ti1, ti2)] = glue_strengths[t1[1]];
                }
            }
        }
        return StaticATAM {
            tile_rates: tile_concs,
            strength_ns,
            strength_we,
            tau
        };
    }  
}

impl StaticKTAM {
    pub fn new(
        tile_concs: Array1<f64>,
        tile_edges: Array2<usize>,
        glue_strengths: Array1<f64>,
        gse: f64,
    ) -> Self {
        let ntiles = tile_concs.len();
        assert!(ntiles == tile_edges.nrows());
        let mut energy_we: Array2<Energy> = Array2::zeros((ntiles, ntiles));
        let mut energy_ns: Array2<Energy> = Array2::zeros((ntiles, ntiles));
        for ti1 in 0..ntiles {
            for ti2 in 0..ntiles {
                let t1 = tile_edges.row(ti1);
                let t2 = tile_edges.row(ti2);
                if t1[2] == t2[0] {
                    energy_ns[(ti1, ti2)] = gse * glue_strengths[t1[2]];
                }
                if t1[1] == t2[3] {
                    energy_we[(ti1, ti2)] = gse * glue_strengths[t1[1]];
                }
            }
        }
        return StaticKTAM {
            tile_rates: tile_concs,
            energy_ns,
            energy_we,
        };
    }
}

impl<S> StateCreate<Canvas2D, S> for State2DQT<S> where S: System<Canvas2D> + Clone {
    fn create(canvas: &Array2<Tile>, sys: &S) -> Self {
        assert!(canvas.nrows().is_power_of_two());

        let p: u32 = (1+canvas.nrows().trailing_zeros()).try_into().unwrap();
        println!("P: {:?}", p);

        let mut rates = Vec::<Array2<Rate>>::new();

        for i in (1..p).rev() {
            rates.push(Array2::<Rate>::zeros((2usize.pow(i), 2usize.pow(i))))
        }

        let size = canvas.nrows();

        let ncanvas = Canvas2D {
            canvas: canvas.to_owned(),
            size,
        };

        let mut ret = State2DQT::<S> {
            rates: rates,
            canvas: ncanvas,
            system: sys.to_owned(), // FIXME: not ideal
            ntiles: canvas.fold(0, |x, y| x + (if *y == 0 { 0 } else { 1 })),
            total_rate: 0.,
            total_events: 0,
        };

        for y in 1..size - 1 {
            for x in 1..size - 1 {
                // FIXME: not at all ideal
                ret.update_rates_single((y, x));
            }
        }

        ret
    }
}


impl<C> System<C> for StaticATAM
where
    C: Canvas,
{
    fn event_rate_at_point(&self, canvas: &C, p: Point) -> Rate {
        if !canvas.inbounds(p) {
            return 0.0;
        }

        // Bound is previously checked.
        let tile = unsafe { canvas.uv_p(p) };

        let tn = unsafe { canvas.uv_n(p) };
        let te = unsafe { canvas.uv_e(p) };
        let ts = unsafe { canvas.uv_s(p) };
        let tw = unsafe { canvas.uv_w(p) };

        if tile != 0 {
            // Deletion
            0.0
        } else {
            // Insertion

            Zip::from(self.strength_ns.row(tn))
                .and(self.strength_we.column(te))
                .and(self.strength_we.row(tw))
                .and(self.strength_ns.column(ts))
                .and(&self.tile_rates)
                .fold(0., |acc, &n, &e, &s, &w, &r| {
                    if n + e + s + w >= self.tau {
                        acc + r
                    } else {
                        acc
                    }
                })
        }
    }

    fn choose_event_at_point(&self, canvas: &C, p: Point, acc: Rate) -> Tile {
        if !canvas.inbounds(p) {
            panic!("Oh dear!");
        }

        // Bound is previously checked.
        let tile = unsafe { canvas.uv_p(p) };

        if tile != 0 {
            // Deletion is easy!
            panic!("We shouldn't be deleting in the aTAM!")
        } else {
            let tn = unsafe { canvas.uv_n(p) };
            let te = unsafe { canvas.uv_e(p) };
            let ts = unsafe { canvas.uv_s(p) };
            let tw = unsafe { canvas.uv_w(p) };

            // Insertion is hard!
            let r = Zip::indexed(self.strength_ns.row(tn))
                .and(self.strength_we.column(te))
                .and(self.strength_ns.column(ts))
                .and(self.strength_we.row(tw))
                .and(&self.tile_rates)
                .fold_while((acc, 0), |(acc, _v), i, &n, &e, &s, &w, &r| {
                    if n + e + s + w >= self.tau {
                        if acc - r > 0. {
                            FoldWhile::Continue((acc - r, 0))
                        } else {
                            FoldWhile::Done((acc - r, i))
                        }
                    } else {
                        FoldWhile::Continue((acc, 0))
                    }
                });

            match r {
                FoldWhile::Done((_acc, i)) => i,

                FoldWhile::Continue((_acc, _i)) => panic!(),
            }
        }
    }
}


impl<C> System<C> for StaticKTAM
where
    C: Canvas,
{
    fn event_rate_at_point(&self, canvas: &C, p: Point) -> Rate {
        if !canvas.inbounds(p) {
            return 0.0;
        }

        // Bound is previously checked.
        let tile = unsafe { canvas.uv_p(p) };

        let tn = unsafe { canvas.uv_n(p) };
        let te = unsafe { canvas.uv_e(p) };
        let ts = unsafe { canvas.uv_s(p) };
        let tw = unsafe { canvas.uv_w(p) };

        if tile != 0 {
            // Deletion

            // Bound is previously checked
            let bound_energy: Energy = self.energy_ns[(tile, ts)]
                + self.energy_ns[(tn, tile)]
                + self.energy_we[(tile, te)]
                + self.energy_we[(tw, tile)];

            Rate::exp(-bound_energy)
        } else {
            // Insertion

            Zip::from(self.energy_ns.row(tn))
                .and(self.energy_we.column(te))
                .and(self.energy_ns.column(ts))
                .and(self.energy_we.row(tw))
                .and(&self.tile_rates)
                .fold(0., |acc, &n, &e, &s, &w, &r| {
                    if (n != 0.) | (e != 0.) | (s != 0.) | (w != 0.) {
                        acc + r
                    } else {
                        acc
                    }
                })
        }
    }

    fn choose_event_at_point(&self, canvas: &C, p: Point, acc: Rate) -> Tile {
        if !canvas.inbounds(p) {
            panic!("Oh dear!");
        }

        // Bound is previously checked.
        let tile = unsafe { canvas.uv_p(p) };

        if tile != 0 {
            // Deletion is easy!
            0
        } else {
            let tn = unsafe { canvas.uv_n(p) };
            let te = unsafe { canvas.uv_e(p) };
            let ts = unsafe { canvas.uv_s(p) };
            let tw = unsafe { canvas.uv_w(p) };

            // Insertion is hard!
            let r = Zip::indexed(self.energy_ns.row(tn))
                .and(self.energy_we.column(te))
                .and(self.energy_ns.column(ts))
                .and(self.energy_we.row(tw))
                .and(&self.tile_rates)
                .fold_while((acc, 0), |(acc, _v), i, &n, &e, &s, &w, &r| {
                    if (n != 0.) | (e != 0.) | (s != 0.) | (w != 0.) {
                        if acc - r > 0. {
                            FoldWhile::Continue((acc - r, 0))
                        } else {
                            FoldWhile::Done((acc - r, i))
                        }
                    } else {
                        FoldWhile::Continue((acc, 0))
                    }
                });

            match r {
                FoldWhile::Done((_acc, i)) => i,

                FoldWhile::Continue((_acc, _i)) => panic!(),
            }
        }
    }
}

impl<T> StateEvolve for T
where
    T: StateStep + StateStatus,
{
    fn evolve_in_size_range(
        &mut self,
        minsize: NumTiles,
        maxsize: NumTiles,
        maxevents: NumEvents,
    ) -> &Self {
        let mut events: NumEvents = 0;

        while events < maxevents {
            self.take_step();

            if (self.ntiles() <= minsize) | (self.ntiles() >= maxsize) {
                return self;
            }

            events += 1;
        }

        panic!("Too many events!");
    }
}

#[derive(Clone)]
pub struct State2DQT<S: System<Canvas2D>> {
    rates: Vec<Array2<Rate>>,
    canvas: Canvas2D,
    system: S,
    ntiles: NumTiles,
    total_rate: Rate,
    total_events: NumEvents,
    //time: f64
}


#[derive(Clone)]
pub struct Canvas2D {
    canvas: Array2<Tile>,
    size: usize,
}

impl Canvas for Canvas2D {
    unsafe fn uv_n(&self, p: Point) -> Tile {
        *self.canvas.uget((p.0 - 1, p.1))
    }

    unsafe fn uv_e(&self, p: Point) -> Tile {
        *self.canvas.uget((p.0, p.1 + 1))
    }

    unsafe fn uv_s(&self, p: Point) -> Tile {
        *self.canvas.uget((p.0 + 1, p.1))
    }

    unsafe fn uv_w(&self, p: Point) -> Tile {
        *self.canvas.uget((p.0, p.1 - 1))
    }

    unsafe fn uv_p(&self, p: Point) -> Tile {
        *self.canvas.uget(p)
    }

    fn inbounds(&self, p: Point) -> bool {
        return (p.0 >= 1) & (p.1 >= 1) & (p.0 < self.size - 1) & (p.1 < self.size - 1);
    }
}

#[macro_use]
macro_rules! plusar {
    ( $p:expr  ) => {
        &[
            $p,
            ($p.0 - 1, $p.1),
            ($p.0 + 1, $p.1),
            ($p.0, $p.1 - 1),
            ($p.0, $p.1 + 1),
        ]
    };
}

impl<S> State2DQT<S>
where
    S: System<Canvas2D>,
{
    fn choose_event_point(&self) -> (Point, Rate) {
        let mut threshold = self.total_rate * random::<Rate>();

        let mut x: usize = 0;
        let mut y: usize = 0;

        for r in self.rates.iter().rev() {
            y *= 2;
            x *= 2;
            if threshold - r[(y, x)] <= 0. {
                continue;
            } else {
                threshold -= r[(y, x)];
                x += 1;
            }
            if threshold - r[(y, x)] <= 0. {
                continue;
            } else {
                threshold -= r[(y, x)];
                y += 1;
                x -= 1
            }
            if threshold - r[(y, x)] <= 0. {
                continue;
            } else {
                threshold -= r[(y, x)];
                x += 1;
            }
            if threshold - r[(y, x)] <= 0. {
                continue;
            } else {
                panic!();
            }
        }
        return ((y, x), threshold);
    }

    fn do_event_at_location(&mut self, p: Point, acc: Rate) -> &Self {
        let newtile = self.system.choose_event_at_point(&self.canvas, p, acc);

        if newtile == 0 {
            self.ntiles -= 1
        } else {
            self.ntiles += 1
        };

        self.total_events += 1;

        // Repeatedly checked!
        unsafe { *self.canvas.canvas.uget_mut(p) = newtile };

        self.update_rates_ps(p)
    }

    fn update_rates_ps(&mut self, p: Point) -> &Self {
        for ps in plusar!(p) {
            // Propagate everything for now (FIXME) until I can think through this process.
            let mut rtiter = self.rates.iter_mut();

            // The base level
            let mut rt = rtiter.next().unwrap();
            let mut np: (usize, usize) = *ps;
            let mut ip: (usize, usize);

            rt[*ps] = self.system.event_rate_at_point(&self.canvas, *ps);

            for rn in rtiter {
                np = (np.0 / 2, np.1 / 2);
                ip = (np.0 * 2, np.1 * 2);
                // *rn.uget_mut(np) = rt
                //     .slice(s![2 * np.0..2 * np.0 + 2, 2 * np.1..2 * np.1 + 2])
                //     .sum();
                unsafe {
                    *rn.uget_mut(np) = *rt.uget(ip)
                    + *rt.uget((ip.0, ip.1+1))
                    + *rt.uget((ip.0+1, ip.1))
                    + *rt.uget((ip.0+1, ip.1+1));
                }
                rt = rn;
            }

            self.total_rate = rt.sum();

        }


        return self;
    }

    fn update_rates_single(&mut self, p: Point) -> &Self {
        let mut rtiter = self.rates.iter_mut();
        let mut rt = rtiter.next().unwrap();
        let mut np: (usize, usize) = p.clone();
        
        rt[p] = self.system.event_rate_at_point(&self.canvas, p);

        for rn in rtiter {
        np = (np.0 / 2, np.1 / 2);
        rn[np] = rt
                .slice(s![2 * np.0..2 * np.0 + 2, 2 * np.1..2 * np.1 + 2])
                .sum();
        rt = rn;
        }

        self.total_rate = rt.sum();

        return self
    }
    }


impl<S> StateStep for State2DQT<S>
where
    S: System<Canvas2D>,
{
    fn take_step(&mut self) -> &Self {
        // Decide on a point.
        let (p, acc) = self.choose_event_point();

        // Do the event.
        self.do_event_at_location(p, acc);

        // Update rates around the point (point, and NESW)
        // Only works without fission and without double tiles!
        self.update_rates_ps(p);

        return self;
    }
}

impl<S> StateStatus for State2DQT<S>
where
    S: System<Canvas2D>,
{
    fn ntiles(&self) -> NumTiles {
        self.ntiles
    }

    fn total_events(&self) -> NumEvents {
        self.total_events
    }
}

// #[pymodule]
// fn rgrow(_py: Python, m: &PyModule) -> PyResult<()> {

//     #[pyclass]
//     #[derive(Clone)]
//     struct PyStaticKTAM {
//         sys: StaticKTAM
//     }


//     #[pyclass]
//     #[derive(Clone)]
//     struct PyStaticATAM {
//         sys: StaticATAM
//     }

//     #[pyclass]
//     #[derive(Clone)]
//     struct PyStateKTAM {
//         state: State2DQT<StaticKTAM>
//     }

//     #[pyclass]
//     #[derive(Clone)]
//     struct PyStateATAM {
//         state: State2DQT<StaticATAM>
//     }

//     #[pymethods]
//     impl PyStaticKTAM {
//         #[new]
//         fn new(tile_concs: PyReadonlyArray1<f64>, tile_edges: PyReadonlyArray2<Tile>, 
//             glue_strengths: PyReadonlyArray1<Energy>, gse: Energy) -> PyResult<Self> {
//             let sys = StaticKTAM::new(tile_concs.to_owned_array(), 
//                         tile_edges.to_owned_array(), 
//                         glue_strengths.to_owned_array(), gse);

//             Ok(PyStaticKTAM { sys })
//         }
//     }

//     #[pymethods]
//     impl PyStaticATAM {
//         #[new]
//         fn new(tile_concs: PyReadonlyArray1<f64>, tile_edges: PyReadonlyArray2<Tile>, 
//             glue_strengths: PyReadonlyArray1<Energy>, tau: Energy) -> PyResult<Self> {
//             let sys = StaticATAM::new(tile_concs.to_owned_array(), 
//                         tile_edges.to_owned_array(), 
//                         glue_strengths.to_owned_array(), tau);

//             Ok(PyStaticATAM { sys })
//         }
//     }

//     #[pymethods]
//     impl PyStateKTAM {
//         #[new]
//         fn new(canvas: PyReadonlyArray2<usize>, system: PyStaticKTAM) -> PyResult<Self> {
//             let state = State2DQT::<StaticKTAM>::create(&canvas.to_owned_array(), &system.sys);

//             Ok(Self { state })
//         }

//         fn take_step(&mut self) -> PyResult<()> {
//             self.state.take_step();

//             Ok(())
//         }

//         fn evolve_in_size_range(&mut self, minsize: usize, maxsize: usize, maxevents: u64) -> PyResult<()> {
//             self.state.evolve_in_size_range(minsize, maxsize, maxevents);

//             Ok(())
//         }

//         fn to_array(&self, py: Python) -> Py<PyArray2<usize>> {
//             self.state.canvas.canvas.to_owned().into_pyarray(py).into()
//         }

//         fn rates(&self, level: usize, py: Python) -> Py<PyArray2<f64>> {
//             self.state.rates[level].to_owned().into_pyarray(py).into()
//         }
//     }

//     #[pymethods]
//     impl PyStateATAM {
//         #[new]
//         fn new(canvas: PyReadonlyArray2<usize>, system: PyStaticATAM) -> PyResult<Self> {
//             let state = State2DQT::<StaticATAM>::create(&canvas.to_owned_array(), &system.sys);

//             Ok(Self { state })
//         }

//         fn take_step(&mut self) -> PyResult<()> {
//             self.state.take_step();

//             Ok(())
//         }

//         fn evolve_in_size_range(&mut self, minsize: usize, maxsize: usize, maxevents: u64) -> PyResult<()> {
//             self.state.evolve_in_size_range(minsize, maxsize, maxevents);

//             Ok(())
//         }

//         fn to_array(&self, py: Python) -> Py<PyArray2<usize>> {
//             self.state.canvas.canvas.to_owned().into_pyarray(py).into()
//         }

//         fn rates(&self, level: usize, py: Python) -> Py<PyArray2<f64>> {
//             self.state.rates[level].to_owned().into_pyarray(py).into()
//         }
//     }
//     m.add_class::<PyStaticATAM>()?;
//     m.add_class::<PyStaticKTAM>()?;
//     m.add_class::<PyStateKTAM>()?;
//     m.add_class::<PyStateATAM>()?;


//     Ok(())
// }