// #![feature(stmt_expr_attributes)]

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
extern crate ndarray;

#[cfg(not(target_arch = "wasm32"))]
use clap::crate_version;
#[cfg(not(target_arch = "wasm32"))]
use clap::Parser;

#[cfg(not(target_arch = "wasm32"))]
use rgrow::base::RgrowError;
#[cfg(not(target_arch = "wasm32"))]
use rgrow::ffs;
#[cfg(not(target_arch = "wasm32"))]
use rgrow::tileset::TileSet;

#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;

// Everything past this line is the desktop CLI. The wasm target ships
// only the library; `main` above is a no-op stub for that case.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser)]
#[clap(version = crate_version!(), author = "Constantine Evans <cevans@costinet.org")]
struct Opts {
    #[clap(subcommand)]
    subcmd: SubCommand,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser)]
enum SubCommand {
    /// Run a tileset in the GUI
    Run(PO),
    /// Run FFS to compute a nucleation rate
    NucRate(FFSOptions),
    /// Internal: GUI subprocess (hidden from help)
    #[clap(name = "gui-subprocess", hide = true)]
    GuiSubprocess(GuiSubprocessArgs),
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser)]
#[clap(version)]
struct GuiSubprocessArgs {
    socket_path: Option<String>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser)]
struct FFSOptions {
    /// Input tileset file
    input: String,
    #[arg(long, default_value_t = 0.01)]
    varpermean2: f64,
    #[arg(long, default_value_t = 1_000)]
    min_configs: usize,
    #[arg(long, default_value_t = 100)]
    target_size: u32,
    #[arg(long, default_value_t = 0.99)]
    cutoff_probability: f64,
    #[arg(long, default_value_t = 4)]
    cutoff_surfaces: usize,
    #[arg(long, default_value_t = 30)]
    min_cutoff_size: u32,
}

#[cfg(not(target_arch = "wasm32"))]
impl From<FFSOptions> for ffs::FFSRunConfig {
    fn from(opts: FFSOptions) -> Self {
        Self {
            var_per_mean2: opts.varpermean2,
            min_configs: opts.min_configs,
            target_size: opts.target_size,
            cutoff_probability: opts.cutoff_probability,
            cutoff_number: opts.cutoff_surfaces,
            min_cutoff_size: opts.min_cutoff_size,
            ..Default::default()
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser)]
struct EO {}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser)]
struct PO {
    /// Input tileset file
    input: String,
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    let opts = Opts::parse();

    match opts.subcmd {
        SubCommand::Run(po) => {
            let parsed = TileSet::from_file(po.input)?;
            parsed.run_window()?;
            Ok(())
        }
        SubCommand::NucRate(po) => {
            nucrate(po)?;
            Ok(())
        }
        #[cfg(feature = "gui")]
        SubCommand::GuiSubprocess(args) => {
            let socket_path = args
                .socket_path
                .ok_or_else(|| anyhow::anyhow!("Usage: rgrow gui-subprocess <socket_path>"))?;
            rgrow::gui::run_gui_subprocess(&socket_path).map_err(|e| anyhow::anyhow!("{}", e))?;
            Ok(())
        }
        #[cfg(not(feature = "gui"))]
        SubCommand::GuiSubprocess(_) => {
            eprintln!("GUI support is not enabled. Rebuild with the 'gui' feature.");
            std::process::exit(1);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn nucrate(po: FFSOptions) -> Result<(), RgrowError> {
    let tileset: TileSet =
        serde_saphyr::from_reader(File::open(po.input.clone()).expect("Input file not found."))
            .expect("Input file parse erorr.");

    let ffsrun = tileset.run_ffs(&po.into())?;

    println!("Nuc rate: {:e}", f64::from(ffsrun.nucleation_rate()));
    let forward_vec_string = ffsrun
        .forward_vec()
        .iter()
        .map(|x| format!("{x:.2e}"))
        .collect::<Vec<String>>()
        .join(", ");
    println!("Forward probabilities: [{forward_vec_string}]");
    Ok(())
}
