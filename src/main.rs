#![feature(stmt_expr_attributes)]
extern crate ndarray;

use clap::Parser;

use rgrow::base::GrowError;
use rgrow::models::ktam::NewKTAM;
use rgrow::{ffs, parser::TileSet, parser_xgrow};

use rgrow::{canvas, state};

use rgrow::parser::FromTileSet;

use serde_yaml;
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
    NucRate(PO),
    RunXgrow(PO),
    //FissionTest(EO)
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
                Err(e) => return Err(Box::new(rgrow::parser::ParserError::Io { source: e })),
            };
            let parsed: TileSet = serde_yaml::from_reader(file)?;
            run_window(parsed)?;
            Ok(())
        }
        SubCommand::NucRate(po) => {
            nucrate(po.input)?;
            Ok(())
        }
        SubCommand::RunXgrow(po) =>
        #[cfg(feature = "ui")]
        {
            let parsed = parser_xgrow::parse_xgrow(po.input)?;
            run_window(parsed)?;
            Ok(())
        }
    }
}

fn nucrate(input: String) -> Result<(), GrowError> {
    let parsed: TileSet =
        serde_yaml::from_reader(File::open(input).expect("Input file not found."))
            .expect("Input file parse erorr.");

    let system = NewKTAM::<state::QuadTreeState<canvas::CanvasPeriodic, state::NullStateTracker>>::from_tileset(&parsed);

    let size = match parsed.options.size {
        rgrow::parser::Size::Single(x) => x,
        rgrow::parser::Size::Pair((x, y)) => x.max(y),
    };

    let ffsrun = ffs::FFSRun::create(system, 1000, 30, size, 1_000, 50_000, 3, 2);

    println!("Nuc rate: {:e}", ffsrun.nucleation_rate());
    println!("Forwards: {:?}", ffsrun.forward_vec());
    Ok(())
}
