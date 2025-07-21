//! A parser for the original Xgrow tileset files.  Note that this tries to fit Xgrow's behavior closely, so parsing is occasionally weird
//! (eg, *no* whitespace is needed to separate things)

use crate::{
    base::Glue,
    system::FissionHandling,
    tileset::{CanvasType, Model, Size, TileShape},
};

use super::base::{GlueIdent, TileIdent};
use super::tileset;
use nom::{
    branch::alt,
    bytes::complete::{is_not, tag},
    character::complete::multispace1,
    character::complete::{digit1, not_line_ending, space0},
    combinator::all_consuming,
    combinator::{map, map_res, opt},
    error::ParseError,
    multi::many1,
    number::complete::recognize_float,
    sequence::{delimited, preceded, tuple},
    IResult,
};

use std::fs::File;
use std::io::prelude::*;

type GlueVec = Vec<(GlueIdent, GlueIdent, f64)>;

fn std_delim<'a, P, O, E: ParseError<&'a str>>(
    parser: P,
) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
where
    P: FnMut(&'a str) -> IResult<&'a str, O, E>,
{
    delimited(rsc, parser, rsc)
}

fn comment<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, &'a str, E> {
    let (input, comment) = preceded(tag("%"), not_line_ending)(input)?;
    Ok((input, comment))
}

fn rsc<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, (), E> {
    let (input, _) = opt(many1(alt((comment, multispace1))))(input)?;
    Ok((input, ()))
}

fn glue(input: &str) -> IResult<&str, GlueIdent> {
    fn glue_num(input: &str) -> IResult<&str, GlueIdent> {
        let (input, n) = map_res(digit1, |n: &str| n.parse::<Glue>())(input)?;
        Ok((input, GlueIdent::Num(n)))
    }

    fn glue_name(input: &str) -> IResult<&str, GlueIdent> {
        map(is_not(" \n\t}"), |n: &str| GlueIdent::Name(n.to_string()))(input)
    }

    alt((glue_num, glue_name))(input)
}

fn string_f64(input: &str) -> IResult<&str, f64> {
    map_res(recognize_float, |x: &str| x.parse::<f64>())(input)
}

fn take_u32(input: &str) -> IResult<&str, u32> {
    let (input, n) = map_res(digit1, |n: &str| n.parse::<u32>())(input)?;
    Ok((input, n))
}

fn tile(input: &str) -> IResult<&str, tileset::Tile> {
    let (input, edges) = std_delim(delimited(tag("{"), many1(std_delim(glue)), tag("}")))(input)?;

    let (input, stoic) = opt(std_delim(delimited(tag("["), string_f64, tag("]"))))(input)?;

    let (input, color) = opt(map(
        std_delim(delimited(tag("("), is_not(")"), tag(")"))),
        |n: &str| n.to_string(),
    ))(input)?;

    Ok((
        input,
        tileset::Tile {
            name: None,
            edges,
            stoic,
            color,
            shape: Some(TileShape::Single),
        },
    ))
}

fn tilelist(input: &str) -> IResult<&str, Vec<tileset::Tile>> {
    preceded(
        std_delim(tag("tile edges=")),
        std_delim(delimited(tag("{"), many1(std_delim(tile)), tag("}"))),
    )(input)
}

fn parse(input: &str) -> IResult<&str, tileset::TileSet> {
    // Consume initial comments
    let (input, _) = opt(std_delim(tag("tile edges matches {{N E S W}*}")))(input)?;

    let (input, _tilenum) =
        std_delim(preceded(tag("num tile types="), preceded(space0, take_u32)))(input)?;

    let (input, _gluenum) = std_delim(preceded(
        tag("num binding types="),
        preceded(space0, take_u32),
    ))(input)?;

    let (input, bondnames) = opt(std_delim(preceded(
        tag("binding type names="),
        std_delim(delimited(
            tag("{"),
            many1(std_delim(is_not(" \n\t}"))),
            tag("}"),
        )),
    )))(input)?;

    let (input, tiles) = tilelist(input)?;

    let (input, bondstrengths) = opt(std_delim(preceded(
        tag("binding strengths="),
        std_delim(delimited(tag("{"), many1(std_delim(string_f64)), tag("}"))),
    )))(input)?;

    let (input, (mut options, glues)) = xgrow_args(input)?;

    // todo: checks

    let bonds = match bondnames {
        Some(b) => match bondstrengths {
            Some(bs) => bs
                .iter()
                .zip(b)
                .map(|(s, n)| tileset::Bond {
                    name: GlueIdent::Name(n.to_string()),
                    strength: *s,
                })
                .collect(),

            None => b
                .iter()
                .map(|n| tileset::Bond {
                    name: GlueIdent::Name(n.to_string()),
                    strength: 1.0,
                })
                .collect(),
        },
        None => match bondstrengths {
            Some(bs) => bs
                .iter()
                .enumerate()
                .map(|(i, s)| tileset::Bond {
                    name: GlueIdent::Num((i + 1) as Glue),
                    strength: *s,
                })
                .collect(),

            None => Vec::new(),
        },
    };

    all_consuming(rsc)(input)?;

    options.tiles = tiles;
    options.bonds = bonds;
    options.glues = glues;

    Ok((input, options))
}

enum XgrowArgs<'a> {
    Block(usize),
    Size(usize),
    Gse(f64),
    Gmc(f64),
    UpdateRate(u64),
    Unhandled(&'a str),
    Seed(tileset::Seed),
    T(f64),
    GlueLink(u32, u32, f64),
    Periodic(bool),
    HDoubleTile(TileIdent, TileIdent),
    VDoubleTile(TileIdent, TileIdent),
}

fn arg_block(input: &str) -> IResult<&str, XgrowArgs<'_>> {
    map(preceded(tag("block="), take_u32), |x| {
        XgrowArgs::Block(x as usize)
    })(input)
}

fn arg_size(input: &str) -> IResult<&str, XgrowArgs<'_>> {
    map(preceded(tag("size="), take_u32), |x| {
        XgrowArgs::Size(x as usize)
    })(input)
}

fn arg_gluelink(input: &str) -> IResult<&str, XgrowArgs<'_>> {
    let (input, (g1, _, g2)) = preceded(
        tag("g"),
        delimited(tag("("), tuple((take_u32, tag(","), take_u32)), tag(")")),
    )(input)?;

    let (input, v) = preceded(tag("="), string_f64)(input)?;

    Ok((input, XgrowArgs::GlueLink(g1, g2, v)))
}

fn arg_gse(input: &str) -> IResult<&str, XgrowArgs<'_>> {
    map(preceded(tag("Gse="), string_f64), XgrowArgs::Gse)(input)
}

fn arg_gmc(input: &str) -> IResult<&str, XgrowArgs<'_>> {
    map(preceded(tag("Gmc="), string_f64), XgrowArgs::Gmc)(input)
}

fn arg_update_rate(input: &str) -> IResult<&str, XgrowArgs<'_>> {
    map(preceded(tag("update_rate="), take_u32), |x| {
        XgrowArgs::UpdateRate(x as u64)
    })(input)
}

fn arg_seed(input: &str) -> IResult<&str, XgrowArgs<'_>> {
    preceded(
        tag("seed="),
        map(
            tuple((take_u32, tag(","), take_u32, tag(","), take_u32)),
            |(y, _, x, _, t)| {
                XgrowArgs::Seed(tileset::Seed::Single(y as usize, x as usize, t.into()))
            },
        ),
    )(input)
}

fn arg_hdoubletile(input: &str) -> IResult<&str, XgrowArgs<'_>> {
    preceded(
        tag("doubletile="),
        map(tuple((take_u32, tag(","), take_u32)), |(x, _, y)| {
            XgrowArgs::HDoubleTile(x.into(), y.into())
        }),
    )(input)
}

fn arg_vdoubletile(input: &str) -> IResult<&str, XgrowArgs<'_>> {
    preceded(
        tag("vdoubletile="),
        map(tuple((take_u32, tag(","), take_u32)), |(x, _, y)| {
            XgrowArgs::VDoubleTile(x.into(), y.into())
        }),
    )(input)
}

fn arg_periodic(input: &str) -> IResult<&str, XgrowArgs<'_>> {
    alt((
        map(tag("periodic=False"), |_| XgrowArgs::Periodic(false)),
        map(tag("periodic=True"), |_| XgrowArgs::Periodic(true)),
        map(tag("periodic"), |_| XgrowArgs::Periodic(true)),
    ))(input)
}

fn arg_threshold(input: &str) -> IResult<&str, XgrowArgs<'_>> {
    map(preceded(tag("T="), string_f64), XgrowArgs::T)(input)
}

fn unhandled_option(input: &str) -> IResult<&str, XgrowArgs<'_>> {
    map(is_not(" \t\r\n%"), XgrowArgs::Unhandled)(input)
}

fn xgrow_args(input: &str) -> IResult<&str, (tileset::TileSet, GlueVec)> {
    let mut args = tileset::TileSet::default();

    let parsers = (
        arg_block,
        arg_size,
        arg_gmc,
        arg_gse,
        arg_update_rate,
        arg_seed,
        arg_threshold,
        arg_gluelink,
        arg_periodic,
        arg_hdoubletile,
        arg_vdoubletile,
        unhandled_option,
    );

    let mut i2 = input;
    args.size = Some(Size::Single(132));

    let mut gluelinks = Vec::new();

    while let Ok((input, x)) = std_delim(alt(parsers))(i2) {
        match x {
            XgrowArgs::Block(n) => {
                args.block = Some(n);
            }
            XgrowArgs::Size(n) => {
                args.size = Some(crate::tileset::Size::Single(n));
            }
            XgrowArgs::Gse(x) => {
                args.gse = Some(x);
            }
            XgrowArgs::Gmc(x) => {
                args.gmc = Some(x);
            }
            XgrowArgs::Unhandled(u) => {
                println!("Warning: \"{u}\" unhandled.");
            }
            XgrowArgs::UpdateRate(x) => {
                args.update_rate = Some(x);
            }
            XgrowArgs::Seed(x) => {
                args.seed = Some(x);
            }
            XgrowArgs::T(x) => {
                args.model = Some(Model::ATAM);
                args.threshold = Some(x);
            }
            XgrowArgs::GlueLink(g1, g2, v) => gluelinks.push((g1.into(), g2.into(), v)),
            XgrowArgs::Periodic(b) => {
                if b {
                    args.canvas_type = Some(CanvasType::Periodic);
                } else {
                    args.canvas_type = Some(CanvasType::Square);
                }
            }
            XgrowArgs::HDoubleTile(t1, t2) => {
                match args.hdoubletiles {
                    None => args.hdoubletiles = Some(vec![(t1, t2)]),
                    Some(ref mut v) => v.push((t1, t2)),
                };
            }
            XgrowArgs::VDoubleTile(t1, t2) => {
                match args.vdoubletiles {
                    None => args.vdoubletiles = Some(vec![(t1, t2)]),
                    Some(ref mut v) => v.push((t1, t2)),
                };
            }
        }
        i2 = input;
    }

    let Some(Size::Single(size)) = args.size else {
        panic!()
    };

    if args.seed.is_none() {
        args.seed = Some(tileset::Seed::Single(size - 3, size - 3, 1.into()));
    }

    if let Some(tileset::Seed::Single(x, y, ref t)) = args.seed {
        if ((x > size - 3) || (y > size - 3))
            || (x < 2)
            || (y < 2)
                && (args.canvas_type == Some(CanvasType::Square) || args.canvas_type.is_none())
        {
            let nx = (x).clamp(2, size - 3);
            let ny = (y).clamp(2, size - 3);
            println!("Warning: seed position {x}, {y} is out of bounds for rgrow.  Adjusting to {nx}, {ny}.");
            args.seed = Some(tileset::Seed::Single(nx, ny, t.clone()));
        }
    }

    args.fission = Some(FissionHandling::NoFission);

    Ok((i2, (args, gluelinks)))
}

pub fn parse_xgrow<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<tileset::TileSet> {
    let mut f = File::open(path)?;

    let mut tilestring = String::new();
    f.read_to_string(&mut tilestring)?;

    parse_xgrow_string(&tilestring)
}

pub fn parse_xgrow_string(tilestring: &str) -> anyhow::Result<tileset::TileSet> {
    parse(tilestring)
        .map_err(|x| x.to_owned().into())
        .map(|x| x.1)
}
