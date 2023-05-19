#[cfg(feature = "python")]
#[pymethods]
impl Tile {
    #[new]
    fn new(
        edges: Vec<GlueIdent>,
        name: Option<String>,
        stoic: Option<f64>,
        color: Option<String>,
        shape: Option<TileShape>,
    ) -> Tile {
        Tile {
            name,
            edges,
            stoic,
            color,
            shape,
        }
    }

    fn __repr__(&self) -> String {
        let mut f = String::from("Tile(");
        if let Some(ref name) = self.name {
            write!(f, "name=\"{}\", ", name).unwrap();
        }
        write!(f, "edges=[").unwrap();
        for edge in &self.edges {
            write!(f, "{}, ", edge).unwrap();
        }
        f.pop();
        f.pop(); // FIXME
        write!(f, "]").unwrap();
        if let Some(stoic) = self.stoic {
            write!(f, ", stoic={}", stoic).unwrap();
        }
        if let Some(ref color) = self.color {
            write!(f, ", color=\"{}\"", color).unwrap();
        }
        if let Some(ref shape) = self.shape {
            write!(f, ", shape={}", shape).unwrap();
        }
        write!(f, ")").unwrap();
        f
    }
}
