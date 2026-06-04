//! Model-introspection and editing surface shared by the desktop GUI and
//! the wasm browser front-end.
//!
//! The [`EditableSystem`] trait (defined in [`super::dispatch`]) is the one
//! place model-specific concentration / glue / blocker editing lives, so
//! both `rgrow-wasm` (which owns a [`SystemEnum`](super::SystemEnum)
//! directly) and the subprocess GUI (which holds a concrete `S: System` in
//! its evolve loop) drive edits through identical code. The plain data
//! structs below are `serde`-serializable so the desktop IPC layer can ship
//! them across the socket and the wasm layer can hand them to JS.

use serde::{Deserialize, Serialize};

use crate::base::{GrowError, Tile};
use crate::colors::Color;
use crate::models::atam::ATAM;
use crate::models::kblock::KBlock;
use crate::models::ktam::KTAM;
use crate::models::oldktam::OldKTAM;
use crate::models::sdc1d::SDC;
use crate::models::sdc1d_bindreplace::SDC1DBindReplace;
use crate::models::sdc2d::SDC2DSquare;
use crate::units::{KcalPerMol, KcalPerMolKelvin, Molar};

use super::dispatch::{EditableSystem, TileBondInfo};
use super::types::NeededUpdate;

/// Capability flags for the editing UI. A front-end reads this once after a
/// sim loads and decides which cells become editable, avoiding any
/// model-name string matching in the UI layer.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EditableFeatures {
    pub tile_concentration: bool,
    pub tile_edge_glue: bool,
    pub glue_interaction: bool,
    pub blocker: bool,
}

/// Describes the per-pair interaction editing schema for the loaded model —
/// what the numbers mean, how to label them, and whether the second
/// (entropy) column exists.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InteractionSchema {
    pub label_dg: String,
    pub has_ds: bool,
    pub label_ds: Option<String>,
}

/// One non-zero entry in the model's pair interaction matrix. `dg` is the
/// model's primary number — KTAM: dimensionless strength; SDC2D / KBlock:
/// ΔG in kcal/mol. `ds` is only populated for SDC2D (ΔS in kcal/(mol·K)).
/// `matching` flags KTAM's special case where `(g, g)` reads from
/// `glue_strengths[g]` instead of `glue_links[(g, g)]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlueInteractionData {
    pub a: u32,
    pub a_name: String,
    pub b: u32,
    pub b_name: String,
    pub matching: bool,
    pub dg: f64,
    pub ds: Option<f64>,
}

/// Per-glue blocker info for the KBlock blocker panel. `concentration` is
/// the user-set total blocker concentration (M); `free_concentration` is
/// the equilibrium free-blocker concentration computed from the tile/glue
/// usages and the blocker–glue ΔG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockerData {
    pub glue_id: u32,
    pub glue_name: String,
    pub concentration: f64,
    pub free_concentration: f64,
}

// ── KTAM ────────────────────────────────────────────────────────────────

impl EditableSystem for KTAM {
    fn editable_features(&self) -> EditableFeatures {
        EditableFeatures {
            tile_concentration: true,
            tile_edge_glue: true,
            glue_interaction: true,
            blocker: false,
        }
    }

    fn interaction_schema(&self) -> InteractionSchema {
        InteractionSchema {
            label_dg: "Strength".to_string(),
            has_ds: false,
            label_ds: None,
        }
    }

    fn tile_concentrations(&self) -> Option<Vec<f64>> {
        Some(self.tile_concs.to_vec())
    }

    fn glue_interactions(&self) -> Vec<GlueInteractionData> {
        let names = self.bond_names();
        let name_at = |id: usize| -> String { names.get(id).cloned().unwrap_or_default() };
        let mut out: Vec<GlueInteractionData> = Vec::new();
        let n = self.glue_strengths.len();
        for g in 1..n {
            let v = self.glue_strengths[g];
            if v != 0.0 {
                out.push(GlueInteractionData {
                    a: g as u32,
                    a_name: name_at(g),
                    b: g as u32,
                    b_name: name_at(g),
                    matching: true,
                    dg: v,
                    ds: None,
                });
            }
        }
        let m = self.glue_links.nrows().min(self.glue_links.ncols());
        for a in 1..m {
            for b in (a + 1)..m {
                let v = self.glue_links[(a, b)];
                if v != 0.0 {
                    out.push(GlueInteractionData {
                        a: a as u32,
                        a_name: name_at(a),
                        b: b as u32,
                        b_name: name_at(b),
                        matching: false,
                        dg: v,
                        ds: None,
                    });
                }
            }
        }
        out
    }

    fn set_tile_concentration(&mut self, id: usize, value: f64) -> Result<NeededUpdate, GrowError> {
        if !value.is_finite() || value < 0.0 {
            return Err(GrowError::NotSupported(
                "setTileConcentration: value must be a non-negative finite number".to_string(),
            ));
        }
        if id == 0 || id >= self.tile_concs.len() {
            return Err(GrowError::NotSupported(
                "setTileConcentration: tile id out of range".to_string(),
            ));
        }
        self.tile_concs[id] = value;
        self.update_system();
        Ok(NeededUpdate::NonZero)
    }

    fn set_tile_edge_glue(
        &mut self,
        id: usize,
        side: usize,
        glue_id: Option<usize>,
    ) -> Result<NeededUpdate, GrowError> {
        if side >= 4 {
            return Err(GrowError::NotSupported(
                "setTileEdgeGlue: side must be 0..=3".to_string(),
            ));
        }
        let g = glue_id.unwrap_or(0);
        if id == 0 || id >= self.tile_edges.nrows() {
            return Err(GrowError::NotSupported(
                "setTileEdgeGlue: tile id out of range".to_string(),
            ));
        }
        if g >= self.glue_strengths.len() {
            return Err(GrowError::NotSupported(
                "setTileEdgeGlue: glue id out of range".to_string(),
            ));
        }
        self.tile_edges[(id, side)] = g;
        self.update_system();
        Ok(NeededUpdate::NonZero)
    }

    fn set_glue_interaction(
        &mut self,
        a: usize,
        b: usize,
        dg: f64,
        _ds: Option<f64>,
    ) -> Result<NeededUpdate, GrowError> {
        if !dg.is_finite() {
            return Err(GrowError::NotSupported(
                "setGlueInteraction: dg must be finite".to_string(),
            ));
        }
        if a == 0 || b == 0 {
            return Err(GrowError::NotSupported(
                "setGlueInteraction: glue id 0 is the null glue".to_string(),
            ));
        }
        let n = self.glue_strengths.len();
        if a >= n || b >= n {
            return Err(GrowError::NotSupported(
                "setGlueInteraction: glue id out of range".to_string(),
            ));
        }
        if a == b {
            self.glue_strengths[a] = dg;
        } else {
            self.glue_links[(a, b)] = dg;
            self.glue_links[(b, a)] = dg;
        }
        self.update_system();
        Ok(NeededUpdate::NonZero)
    }
}

// ── SDC2DSquare ───────────────────────────────────────────────────────────

impl EditableSystem for SDC2DSquare {
    fn editable_features(&self) -> EditableFeatures {
        EditableFeatures {
            tile_concentration: true,
            tile_edge_glue: true,
            glue_interaction: true,
            blocker: false,
        }
    }

    fn interaction_schema(&self) -> InteractionSchema {
        InteractionSchema {
            label_dg: "ΔG (kcal/mol)".to_string(),
            has_ds: true,
            label_ds: Some("ΔS (kcal/(mol·K))".to_string()),
        }
    }

    fn tile_concentrations(&self) -> Option<Vec<f64>> {
        Some(
            self.strand_concentration
                .iter()
                .map(|c| f64::from(*c))
                .collect(),
        )
    }

    fn glue_interactions(&self) -> Vec<GlueInteractionData> {
        let names = self.bond_names();
        let name_at = |id: usize| -> String { names.get(id).cloned().unwrap_or_default() };
        let mut out: Vec<GlueInteractionData> = Vec::new();
        let m = self
            .delta_g_matrix
            .nrows()
            .min(self.delta_g_matrix.ncols())
            .min(self.entropy_matrix.nrows())
            .min(self.entropy_matrix.ncols());
        for a in 1..m {
            for b in a..m {
                let dg: f64 = self.delta_g_matrix[(a, b)].into();
                let ds: f64 = self.entropy_matrix[(a, b)].into();
                if dg != 0.0 || ds != 0.0 {
                    out.push(GlueInteractionData {
                        a: a as u32,
                        a_name: name_at(a),
                        b: b as u32,
                        b_name: name_at(b),
                        matching: a == b,
                        dg,
                        ds: Some(ds),
                    });
                }
            }
        }
        out
    }

    fn set_tile_concentration(&mut self, id: usize, value: f64) -> Result<NeededUpdate, GrowError> {
        if !value.is_finite() || value < 0.0 {
            return Err(GrowError::NotSupported(
                "setTileConcentration: value must be a non-negative finite number".to_string(),
            ));
        }
        if id == 0 || id >= self.strand_concentration.len() {
            return Err(GrowError::NotSupported(
                "setTileConcentration: strand id out of range".to_string(),
            ));
        }
        self.strand_concentration[id] = Molar::from(value);
        self.update_system();
        Ok(NeededUpdate::NonZero)
    }

    fn set_tile_edge_glue(
        &mut self,
        id: usize,
        side: usize,
        glue_id: Option<usize>,
    ) -> Result<NeededUpdate, GrowError> {
        if side >= 4 {
            return Err(GrowError::NotSupported(
                "setTileEdgeGlue: side must be 0..=3".to_string(),
            ));
        }
        let g = glue_id.unwrap_or(0);
        // strand_glues columns are NORTH/EAST/SOUTH/WEST (0..=3); the bottom
        // (scaffold) glue lives in column 4 and isn't editable from the
        // per-side cells.
        if id == 0 || id >= self.strand_glues.nrows() {
            return Err(GrowError::NotSupported(
                "setTileEdgeGlue: strand id out of range".to_string(),
            ));
        }
        if g >= self.glue_names.len() {
            return Err(GrowError::NotSupported(
                "setTileEdgeGlue: glue id out of range".to_string(),
            ));
        }
        self.strand_glues[(id, side)] = g;
        self.update_system();
        Ok(NeededUpdate::NonZero)
    }

    fn set_glue_interaction(
        &mut self,
        a: usize,
        b: usize,
        dg: f64,
        ds: Option<f64>,
    ) -> Result<NeededUpdate, GrowError> {
        if !dg.is_finite() {
            return Err(GrowError::NotSupported(
                "setGlueInteraction: dg must be finite".to_string(),
            ));
        }
        if let Some(d) = ds {
            if !d.is_finite() {
                return Err(GrowError::NotSupported(
                    "setGlueInteraction: ds must be finite".to_string(),
                ));
            }
        }
        if a == 0 || b == 0 {
            return Err(GrowError::NotSupported(
                "setGlueInteraction: glue id 0 is the null glue".to_string(),
            ));
        }
        let nrows = self.delta_g_matrix.nrows();
        let ncols = self.delta_g_matrix.ncols();
        if a >= nrows || b >= ncols || a >= self.entropy_matrix.nrows() {
            return Err(GrowError::NotSupported(
                "setGlueInteraction: glue id out of range".to_string(),
            ));
        }
        let dg_val = KcalPerMol::from(dg);
        self.delta_g_matrix[(a, b)] = dg_val;
        self.delta_g_matrix[(b, a)] = dg_val;
        if let Some(d) = ds {
            let ds_val = KcalPerMolKelvin::from(d);
            self.entropy_matrix[(a, b)] = ds_val;
            self.entropy_matrix[(b, a)] = ds_val;
        }
        self.update_system();
        Ok(NeededUpdate::NonZero)
    }
}

// ── KBlock ────────────────────────────────────────────────────────────────

impl EditableSystem for KBlock {
    fn editable_features(&self) -> EditableFeatures {
        EditableFeatures {
            tile_concentration: true,
            tile_edge_glue: false,
            glue_interaction: true,
            blocker: true,
        }
    }

    fn interaction_schema(&self) -> InteractionSchema {
        InteractionSchema {
            label_dg: "ΔG (kcal/mol)".to_string(),
            has_ds: false,
            label_ds: None,
        }
    }

    fn tile_concentrations(&self) -> Option<Vec<f64>> {
        Some(
            self.tile_concentration
                .iter()
                .map(|c| f64::from(*c))
                .collect(),
        )
    }

    fn free_tile_concentrations(&self) -> Option<Vec<f64>> {
        // `unblocked_tile_concentration` adjusts the raw conc by the
        // equilibrium blocker occupancy on each side, so the result is the
        // "available" (fully-unblocked) tile concentration shown in the UI.
        Some(
            (0..self.tile_concentration.len())
                .map(|i| f64::from(self.unblocked_tile_concentration(i)))
                .collect(),
        )
    }

    fn glue_interactions(&self) -> Vec<GlueInteractionData> {
        let names = self.bond_names();
        let name_at = |id: usize| -> String { names.get(id).cloned().unwrap_or_default() };
        let mut out: Vec<GlueInteractionData> = Vec::new();
        let links = self.glue_links();
        let m = links.nrows().min(links.ncols());
        for a in 1..m {
            for b in a..m {
                let dg: f64 = links[(a, b)].into();
                if dg != 0.0 {
                    out.push(GlueInteractionData {
                        a: a as u32,
                        a_name: name_at(a),
                        b: b as u32,
                        b_name: name_at(b),
                        matching: a == b,
                        dg,
                        ds: None,
                    });
                }
            }
        }
        out
    }

    fn blocker_list(&self) -> Vec<BlockerData> {
        let frees = self.free_blocker_concentrations();
        let n = self
            .glue_names
            .len()
            .min(self.blocker_concentrations.len())
            .min(frees.len());
        let mut out: Vec<BlockerData> = Vec::new();
        for gi in 1..n {
            let name = &self.glue_names[gi];
            if name.is_empty() {
                continue;
            }
            out.push(BlockerData {
                glue_id: gi as u32,
                glue_name: name.clone(),
                concentration: f64::from(self.blocker_concentrations[gi]),
                free_concentration: f64::from(frees[gi]),
            });
        }
        out
    }

    fn panel_tri_colors(&self, base_id: usize) -> [Color; 4] {
        // KBlock's color lookup expects an encoded TileState (base index in
        // the high bits, blocker mask in the low 4 bits). The tileset panel
        // iterates base indices, so shift to address the right color slot;
        // a low-bit mask of 0 means "no blockers attached," which is what we
        // want for the panel's reference sprite.
        self.tile_style((base_id << 4) as Tile).tri_colors
    }

    fn canvas_id_shift(&self) -> u32 {
        // KBlock stores `TileState = base << 4 | blocker_mask` in the canvas.
        4
    }

    fn set_tile_concentration(&mut self, id: usize, value: f64) -> Result<NeededUpdate, GrowError> {
        if !value.is_finite() || value < 0.0 {
            return Err(GrowError::NotSupported(
                "setTileConcentration: value must be a non-negative finite number".to_string(),
            ));
        }
        if id == 0 || id >= self.tile_concentration.len() {
            return Err(GrowError::NotSupported(
                "setTileConcentration: tile id out of range".to_string(),
            ));
        }
        self.tile_concentration[id] = Molar::from(value);
        // Recompute energies / free-blocker concentrations so attachment
        // rates stay consistent with the new total.
        self.update();
        Ok(NeededUpdate::NonZero)
    }

    fn set_glue_interaction(
        &mut self,
        a: usize,
        b: usize,
        dg: f64,
        _ds: Option<f64>,
    ) -> Result<NeededUpdate, GrowError> {
        if !dg.is_finite() {
            return Err(GrowError::NotSupported(
                "setGlueInteraction: dg must be finite".to_string(),
            ));
        }
        if a == 0 || b == 0 {
            return Err(GrowError::NotSupported(
                "setGlueInteraction: glue id 0 is the null glue".to_string(),
            ));
        }
        let links = self.glue_links();
        if a >= links.nrows() || b >= links.ncols() {
            return Err(GrowError::NotSupported(
                "setGlueInteraction: glue id out of range".to_string(),
            ));
        }
        self.set_glue_link(a, b, KcalPerMol::from(dg));
        self.update();
        Ok(NeededUpdate::NonZero)
    }

    fn set_blocker_concentration(
        &mut self,
        glue_id: usize,
        value: f64,
    ) -> Result<NeededUpdate, GrowError> {
        if !value.is_finite() || value < 0.0 {
            return Err(GrowError::NotSupported(
                "setBlockerConcentration: value must be a non-negative finite number".to_string(),
            ));
        }
        if glue_id == 0 || glue_id >= self.blocker_concentrations.len() {
            return Err(GrowError::NotSupported(
                "setBlockerConcentration: glue id out of range".to_string(),
            ));
        }
        self.blocker_concentrations[glue_id] = Molar::from(value);
        self.update();
        Ok(NeededUpdate::NonZero)
    }
}

// ── Models without editing support: default (no-op) impls ─────────────────
//
// `EditableSystem` is a supertrait of `System`, so every model must provide
// an impl. These use the trait defaults: no editable features, empty
// interaction/blocker lists, and `not_supported` errors for every setter.

impl EditableSystem for ATAM {}
impl EditableSystem for OldKTAM {}
impl EditableSystem for SDC {}
impl EditableSystem for SDC1DBindReplace {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ktam_tile_concentration_roundtrip() {
        let mut k = KTAM::new_sized(5, 5);
        let needed = k.set_tile_concentration(2, 1.5e-7).unwrap();
        assert_eq!(needed, NeededUpdate::NonZero);
        let concs = k.tile_concentrations().unwrap();
        assert!((concs[2] - 1.5e-7).abs() < 1e-20);
        // empty tile (id 0), out-of-range, and negative values are rejected.
        assert!(k.set_tile_concentration(0, 1.0).is_err());
        assert!(k.set_tile_concentration(99, 1.0).is_err());
        assert!(k.set_tile_concentration(2, -1.0).is_err());
    }

    #[test]
    fn ktam_glue_interaction_is_symmetric() {
        let mut k = KTAM::new_sized(5, 5);
        k.set_glue_interaction(1, 2, 3.0, None).unwrap();
        assert_eq!(k.glue_links[(1, 2)], 3.0);
        assert_eq!(k.glue_links[(2, 1)], 3.0);
        // A self-pair writes glue_strengths, not the link matrix.
        k.set_glue_interaction(3, 3, 2.5, None).unwrap();
        assert_eq!(k.glue_strengths[3], 2.5);
        // The null glue (id 0) is rejected.
        assert!(k.set_glue_interaction(0, 1, 1.0, None).is_err());
    }

    #[test]
    fn ktam_features_and_shift() {
        let k = KTAM::new_sized(2, 2);
        let f = k.editable_features();
        assert!(f.tile_concentration && f.tile_edge_glue && f.glue_interaction);
        assert!(!f.blocker);
        assert_eq!(k.canvas_id_shift(), 0);
    }
}
