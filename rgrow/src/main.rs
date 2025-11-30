// #![feature(stmt_expr_attributes)]
extern crate ndarray;

use clap::Parser;

use rgrow::base::RgrowError;
use rgrow::ffs;
use rgrow::tileset::TileSet;

use std::fs::File;

#[derive(Parser)]
#[clap(version = "0.1.0", author = "Constantine Evans <cevans@costinet.org")]
struct Opts {
    #[clap(subcommand)]
    subcmd: SubCommand,
}

#[derive(Parser)]
enum SubCommand {
    Run(PO),
    NucRate(FFSOptions),
}

#[derive(Parser)]
struct FFSOptions {
    input: String,
    #[arg(short, long, default_value_t = 0.01)]
    varpermean2: f64,
    #[arg(short, long, default_value_t = 1_000)]
    min_configs: usize,
    #[arg(short, long, default_value_t = 100)]
    target_size: u32,
    #[arg(short, long, default_value_t = 0.99)]
    cutoff_probability: f64,
    #[arg(short, long, default_value_t = 4)]
    cutoff_surfaces: usize,
    #[arg(short, long, default_value_t = 30)]
    min_cutoff_size: u32,
}

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

#[derive(Parser)]
struct EO {}

#[derive(Parser)]
struct PO {
    input: String,
}

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
    }
}

fn nucrate(po: FFSOptions) -> Result<(), RgrowError> {
    let tileset: TileSet =
        serde_yaml::from_reader(File::open(po.input.clone()).expect("Input file not found."))
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
