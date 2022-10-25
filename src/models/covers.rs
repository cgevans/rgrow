use std::collections::HashMap;

use ndarray::{Array1, Array2};
use rand::{rngs::SmallRng, Rng};

use super::oldktam::OldKTAM;
use crate::{
    base::{NumEvents, NumTiles, Point, Rate, Tile},
    canvas::{PointSafe2, PointSafeHere},
    models::oldktam::Seed,
    state::{State, StateCreate},
    system::{ChunkSize, DimerInfo, Event, StepOutcome, System, SystemWithDimers, TileBondInfo},
    tileset::{FromTileSet, ParsedSeed, TileSet},
};

#[derive(Debug, Clone)]
pub(crate) enum CoverType {
    NonCover,
    Cover,
    Composite,
}

#[derive(Debug, Clone)]
pub(crate) struct CoverAttach {
    pub(crate) like_tile: Tile,
    pub(crate) new_tile: Tile,
}

#[derive(Debug, Clone)]
pub(crate) struct CompositeDetach {
    pub(crate) like_tile: Tile,
    pub(crate) new_tile: Tile,
}

enum PossibleChoice {
    Remainder(Rate),
    Event(Event),
}

#[derive(Debug, Clone)]
pub struct StaticKTAMCover<S: State> {
    pub inner: OldKTAM<S>,
    pub(crate) tile_is_cover: Vec<CoverType>,
    pub(crate) cover_attach_info: Vec<Vec<CoverAttach>>,
    pub(crate) composite_detach_info: Vec<Vec<CompositeDetach>>,
}

impl<S: State> System<S> for StaticKTAMCover<S> {
    fn update_after_event(&self, state: &mut S, event: &Event) {
        match event {
            Event::None => {
                panic!("Being asked to update after a dead event.")
            }
            Event::MonomerAttachment(p, _)
            | Event::MonomerDetachment(p)
            | Event::MonomerChange(p, _) => match self.inner.chunk_size {
                ChunkSize::Single => {
                    let points = [
                        state.move_sa_n(*p),
                        state.move_sa_w(*p),
                        PointSafeHere(p.0),
                        state.move_sa_e(*p),
                        state.move_sa_s(*p),
                    ];
                    self.update_points(state, &points);
                }
                ChunkSize::Dimer => {
                    let mut points = Vec::with_capacity(10);
                    points.extend_from_slice(&[
                        state.move_sa_n(*p),
                        state.move_sa_w(*p),
                        PointSafeHere(p.0),
                        state.move_sa_e(*p),
                        state.move_sa_s(*p),
                        state.move_sa_nw(*p),
                        state.move_sa_ne(*p),
                        state.move_sa_sw(*p),
                    ]);

                    let w = state.move_sa_w(*p);
                    let n = state.move_sa_n(*p);

                    if state.inbounds(w.0) {
                        points.push(PointSafeHere(state.move_sh_w(w)));
                    }
                    if state.inbounds(n.0) {
                        points.push(PointSafeHere(state.move_sh_n(n)));
                    }

                    self.update_points(state, &points);
                }
            },
            Event::PolymerDetachment(v) => {
                let mut points = Vec::new();
                for p in v {
                    points.extend(self.inner.points_to_update_around(state, p));
                }
                points.sort_unstable();
                points.dedup();
                self.update_points(state, &points);
            }
            Event::PolymerAttachment(v) | Event::PolymerChange(v) => {
                let mut points = Vec::new();
                for (p, _) in v {
                    points.extend(self.inner.points_to_update_around(state, p));
                }
                points.sort_unstable();
                points.dedup();
                self.update_points(state, &points);
            }
        }
    }

    fn event_rate_at_point(&self, state: &S, p: PointSafeHere) -> Rate {
        let t = state.v_sh(p) as usize;

        if !state.inbounds(p.0) {
            return 0.;
        }

        let sp = PointSafe2(p.0);

        match self.tile_is_cover[t] {
            CoverType::NonCover => self.inner.event_rate_at_point(state, p),
            CoverType::Cover => {
                self.inner.event_rate_at_point(state, p)
                    + self.cover_to_composite_rate(state, sp, t)
            }
            CoverType::Composite => self.composite_to_cover_rate(state, sp, t),
        }
    }

    fn choose_event_at_point(&self, state: &S, p: PointSafe2, acc: Rate) -> Event {
        let t = state.tile_at_point(p) as usize;

        match self.tile_is_cover[t] {
            CoverType::NonCover => self.inner.choose_event_at_point(state, p, acc),
            CoverType::Cover => match self.choose_cover_to_composite(state, p, t, acc) {
                PossibleChoice::Remainder(acc) => self.inner.choose_event_at_point(state, p, acc),
                PossibleChoice::Event(e) => e,
            },
            CoverType::Composite => match self.choose_composite_to_cover(state, p, t, acc) {
                PossibleChoice::Remainder(_) => {
                    panic!("Ran out of rate for composite.")
                }
                PossibleChoice::Event(e) => e,
            },
        }
    }

    fn seed_locs(&self) -> Vec<(PointSafe2, Tile)> {
        self.inner.seed_locs()
    }

    fn calc_mismatch_locations(&self, state: &S) -> Array2<usize> {
        self.inner.calc_mismatch_locations(state)
    }

    fn state_step(&self, state: &mut S, rng: &mut SmallRng, max_time_step: f64) -> StepOutcome {
        let time_step = -f64::ln(rng.gen()) / state.total_rate();
        if time_step > max_time_step {
            state.add_time(max_time_step);
            return StepOutcome::NoEventIn(max_time_step);
        }
        let (point, remainder) = state.choose_point(rng); // todo: resultify
        let event = self.choose_event_at_point(state, PointSafe2(point), remainder); // FIXME
        if let Event::None = event {
            state.add_time(time_step);
            return StepOutcome::DeadEventAt(time_step);
        }

        self.perform_event(state, &event);
        self.update_after_event(state, &event);
        state.add_time(time_step);
        StepOutcome::HadEventAt(time_step)
    }

    fn evolve_in_size_range_events_max(
        &mut self,
        state: &mut S,
        minsize: NumTiles,
        maxsize: NumTiles,
        maxevents: NumEvents,
        rng: &mut SmallRng,
    ) {
        let mut events: NumEvents = 0;

        while (events < maxevents) & (state.ntiles() < maxsize) & (state.ntiles() > minsize) {
            match self.state_step(state, rng, 1e100) {
                StepOutcome::HadEventAt(_) => {
                    events += 1;
                }
                StepOutcome::NoEventIn(_) => {
                    println!("Timeout {:?}", state);
                }
                StepOutcome::DeadEventAt(_) => {
                    println!("Dead");
                }
                StepOutcome::ZeroRate => {
                    panic!()
                }
            }
        }
    }

    fn set_point(&self, state: &mut S, point: Point, tile: Tile) {
        assert!(state.inbounds(point));

        let point = PointSafe2(point);

        state.set_sa(&point, &tile);

        let event = Event::MonomerAttachment(point, tile);

        self.update_after_event(state, &event);
    }

    fn perform_event(&self, state: &mut S, event: &Event) {
        match event {
            Event::None => panic!("Being asked to perform null event."),
            Event::MonomerAttachment(point, tile) | Event::MonomerChange(point, tile) => {
                state.set_sa(point, tile);
            }
            Event::MonomerDetachment(point) => {
                state.set_sa(point, &0usize);
            }
            Event::PolymerAttachment(changelist) | Event::PolymerChange(changelist) => {
                for (point, tile) in changelist {
                    state.set_sa(point, tile);
                }
            }
            Event::PolymerDetachment(changelist) => {
                for point in changelist {
                    state.set_sa(point, &0usize);
                }
            }
        }
    }
}

impl<S: State> SystemWithDimers<S> for StaticKTAMCover<S> {
    fn calc_dimers(&self) -> Vec<DimerInfo> {
        self.inner.calc_dimers()
    }
}

impl<S: State> StaticKTAMCover<S> {
    fn cover_to_composite_rate(&self, state: &S, p: PointSafe2, t: usize) -> Rate {
        let cc = &self.cover_attach_info[t as usize];

        let mut total_rate = 0.;
        for c in cc {
            if self
                .inner
                .bond_strength_of_tile_at_point(state, p, c.like_tile)
                > 0.
            {
                total_rate +=
                    self.inner.k_f_hat() * self.inner.tile_adj_concs[c.like_tile as usize];
            }
        }

        total_rate
    }
    fn choose_cover_to_composite(
        &self,
        state: &S,
        p: PointSafe2,
        t: usize,
        mut acc: Rate,
    ) -> PossibleChoice {
        let cc = &self.cover_attach_info[t as usize];

        for c in cc {
            if self
                .inner
                .bond_strength_of_tile_at_point(state, p, c.like_tile)
                > 0.
            {
                acc -= self.inner.k_f_hat() * self.inner.tile_adj_concs[c.like_tile as usize];
                if acc <= 0. {
                    return PossibleChoice::Event(Event::MonomerChange(p, c.new_tile));
                }
            }
        }

        PossibleChoice::Remainder(acc)
    }
    fn composite_to_cover_rate(&self, state: &S, p: PointSafe2, t: usize) -> Rate {
        let cc = &self.composite_detach_info[t as usize];

        let mut total_rate = 0.;
        for c in cc {
            total_rate += self.inner.k_f_hat()
                * f64::exp(
                    -self
                        .inner
                        .bond_strength_of_tile_at_point(state, p, c.like_tile),
                );
        }

        total_rate
    }
    fn choose_composite_to_cover(
        &self,
        state: &S,
        p: PointSafe2,
        t: usize,
        mut acc: Rate,
    ) -> PossibleChoice {
        let cc = &self.composite_detach_info[t as usize];

        for c in cc {
            acc -= self.inner.k_f_hat()
                * f64::exp(
                    -self
                        .inner
                        .bond_strength_of_tile_at_point(state, p, c.like_tile),
                );
            if acc <= 0. {
                return PossibleChoice::Event(Event::MonomerChange(p, c.new_tile));
            }
        }

        PossibleChoice::Remainder(acc)
    }
}

impl<C: State> TileBondInfo for StaticKTAMCover<C> {
    fn tile_color(&self, tile_number: Tile) -> [u8; 4] {
        self.inner.tile_colors[tile_number as usize]
    }

    fn tile_name(&self, tile_number: Tile) -> &str {
        self.inner.tile_names[tile_number as usize].as_str()
    }

    fn bond_name(&self, _bond_number: usize) -> &str {
        todo!()
    }

    fn tile_colors(&self) -> &Vec<[u8; 4]> {
        self.inner.tile_colors()
    }

    fn tile_names(&self) -> Vec<String> {
        self.inner.tile_names()
    }

    fn bond_names(&self) -> Vec<String> {
        todo!()
    }
}

impl<St: State + StateCreate> FromTileSet for StaticKTAMCover<St> {
    fn from_tileset(tileset: &TileSet) -> Self {
        let mut tsc: TileSet = (*tileset).to_owned();

        let cs = tsc.cover_strands.as_ref().unwrap();

        let mut tile_is_cover = Vec::with_capacity(tsc.tiles.len() + cs.len());
        let mut cover_attach_info = Vec::with_capacity(tsc.tiles.len() + cs.len());
        let mut composite_detach_info = Vec::with_capacity(tsc.tiles.len() + cs.len());

        let mut extratiles = Vec::new();

        cover_attach_info.push(Vec::new());
        composite_detach_info.push(Vec::new());
        tile_is_cover.push(CoverType::NonCover);

        for _ in 0..tsc.tiles.len() {
            tile_is_cover.push(CoverType::NonCover);
            cover_attach_info.push(Vec::new());
            composite_detach_info.push(Vec::new());
        }
        for c in cs {
            tile_is_cover.push(CoverType::Cover);
            composite_detach_info.push(Vec::new());
            cover_attach_info.push(Vec::new());
            extratiles.push(c.to_tile());
        }

        let coverbegin = tsc.tiles.len() + 1;
        let mut comp = coverbegin + cs.len();

        for i in 0..cs.len() {
            for j in i..cs.len() {
                // Same direction: can't attach at the same place at the same time.
                if cs[i].dir == cs[j].dir {
                    continue;
                }

                assert!(comp == coverbegin + extratiles.len());
                extratiles.push(cs[i].make_composite(&cs[j]));

                cover_attach_info[coverbegin + i].push(CoverAttach {
                    like_tile: (coverbegin + i),
                    new_tile: comp,
                });
                cover_attach_info[coverbegin + j].push(CoverAttach {
                    like_tile: (coverbegin + j),
                    new_tile: comp,
                });

                tile_is_cover.push(CoverType::Composite);
                composite_detach_info.push(vec![
                    CompositeDetach {
                        like_tile: (coverbegin + i),
                        new_tile: (coverbegin + j),
                    },
                    CompositeDetach {
                        like_tile: (coverbegin + j),
                        new_tile: (coverbegin + i),
                    },
                ]);

                comp += 1;
            }
        }

        tsc.tiles.extend(extratiles);

        for tile in tsc.tiles.iter() {
            println!("{:?}", tile);
        }

        assert!(comp == tsc.tiles.len() + 1);

        let (gluemap, gluestrengthmap) = tsc.number_glues().unwrap();

        let tile_edges = tsc.tile_edge_process(&gluemap);
        let mut tile_concs = tsc.tile_stoics();
        tile_concs *= f64::exp(-tsc.options.gmc + tsc.options.alpha);

        let mut glue_strength_vec = Vec::<f64>::new();

        for (i, (j, v)) in gluestrengthmap.into_iter().enumerate() {
            assert!(j == i);
            glue_strength_vec.push(v);
        }

        let seed = match &tsc.options.seed {
            ParsedSeed::Single(y, x, v) => Seed::SingleTile {
                point: (*y, *x),
                tile: *v,
            },
            ParsedSeed::None() => Seed::None(),
            ParsedSeed::Multi(vec) => {
                let mut hm = HashMap::default();
                hm.extend(vec.iter().map(|(y, x, v)| ((*y, *x), *v)));
                Seed::MultiTile(hm)
            }
        };

        let inner = OldKTAM::from_ktam(
            tsc.tile_stoics(),
            tile_edges,
            Array1::from(glue_strength_vec),
            tsc.options.gse,
            tsc.options.gmc,
            Some(tsc.options.alpha),
            tsc.options.kf,
            Some(seed),
            Some(tsc.options.fission),
            tsc.options.chunk_handling,
            tsc.options.chunk_size,
            Some(tsc.tile_names()),
            Some(tsc.tile_colors()),
        );

        StaticKTAMCover {
            inner,
            tile_is_cover,
            cover_attach_info,
            composite_detach_info,
        }
    }
}
