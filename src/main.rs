#![feature(stmt_expr_attributes)]
extern crate ndarray;

use clap::Parser;

use rgrow::base::GrowError;
use rgrow::{parser_xgrow, tileset::TileSet};

use std::fs::File;

#[cfg(feature = "ui")]
use rgrow::ui::run_window;

#[derive(Parser)]
#[clap(version = "0.1.0", author = "Constantine Evans <cevans@costinet.org")]
struct Opts {
    #[clap(subcommand)]
    subcmd: SubCommand,
}

#[derive(Parser)]
enum SubCommand {
    //Run(EO),
    //RunSubs(EO),
    //Parse(PO),
    //RunAtam(PO),
    Run(PO),
    NucRate(FFSOptions),
    RunXgrow(PO),
    //FissionTest(EO)
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

#[derive(Parser)]
struct EO {}

#[derive(Parser)]
struct PO {
    input: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts = Opts::parse();

    match opts.subcmd {
        SubCommand::Run(po) =>
        #[cfg(feature = "ui")]
        {
            let file = match File::open(po.input) {
                Ok(f) => f,
                Err(e) => return Err(Box::new(rgrow::tileset::ParserError::Io { source: e })),
            };
            let parsed: TileSet = serde_yaml::from_reader(file)?;
            run_window(&parsed)?;
            Ok(())
        }
        SubCommand::NucRate(po) => {
            nucrate(po)?;
            Ok(())
        }
        SubCommand::RunXgrow(po) =>
        #[cfg(feature = "ui")]
        {
            let parsed = parser_xgrow::parse_xgrow(po.input)?;
            run_window(&parsed)?;
            Ok(())
        }
    }
}

fn nucrate(po: FFSOptions) -> Result<(), GrowError> {
    let tileset: TileSet =
        serde_yaml::from_reader(File::open(po.input).expect("Input file not found."))
            .expect("Input file parse erorr.");

    let ffsrun = tileset.run_ffs(
        po.varpermean2,
        po.min_configs,
        po.target_size,
        po.cutoff_probability,
        po.cutoff_surfaces,
        po.min_cutoff_size,
        10_000,
        1_000_000,
        2,
        1,
        false,
        None,
    )?;

    println!("Nuc rate: {:e}", ffsrun.nucleation_rate());
    println!("Forwards: {:?}", ffsrun.forward_vec());
    Ok(())
}
