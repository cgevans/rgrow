//! A parser for the original Xgrow tileset files.  Note that this tries to fit Xgrow's behavior closely, so parsing is occasionally weird
//! (eg, *no* whitespace is needed to separate things)

use crate::system::FissionHandling;

use super::parser;
use super::parser::GlueIdent;
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

use std::io::prelude::*;
use std::{error::Error, fs::File};

fn std_delim<'a, P, O, E: ParseError<&'a str>>(
    parser: P,
) -> impl Fn(&'a str) -> IResult<&'a str, O, E>
where
    P: Fn(&'a str) -> IResult<&'a str, O, E>,
{
    delimited(rsc, parser, rsc)
}

fn comment<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&str, &str, E> {
    let (input, comment) = preceded(tag("%"), not_line_ending)(input)?;
    Ok((input, comment))
}

fn rsc<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, (), E> {
    let (input, _) = opt(many1(alt((comment, multispace1))))(input)?;
    Ok((input, ()))
}

fn glue(input: &str) -> IResult<&str, parser::GlueIdent> {
    fn glue_num(input: &str) -> IResult<&str, parser::GlueIdent> {
        let (input, n) = map_res(digit1, |n: &str| n.parse::<u32>())(input)?;
        Ok((input, GlueIdent::Num(n)))
    }

    fn glue_name(input: &str) -> IResult<&str, parser::GlueIdent> {
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

fn tile(input: &str) -> IResult<&str, parser::Tile> {
    let (input, edges) = std_delim(delimited(tag("{"), many1(std_delim(glue)), tag("}")))(input)?;

    let (input, stoic) = opt(std_delim(delimited(tag("["), string_f64, tag("]"))))(input)?;

    let (input, color) = opt(map(
        std_delim(delimited(tag("("), is_not(")"), tag(")"))),
        |n: &str| n.to_string(),
    ))(input)?;

    Ok((
        input,
        parser::Tile {
            name: None,
            edges,
            stoic,
            color,
        },
    ))
}

fn tilelist(input: &str) -> IResult<&str, Vec<parser::Tile>> {
    preceded(
        std_delim(tag("tile edges=")),
        std_delim(delimited(tag("{"), many1(std_delim(tile)), tag("}"))),
    )(input)
}

fn parse(input: &str) -> IResult<&str, parser::TileSet> {
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

    let (input, bondstrengths) = std_delim(preceded(
        tag("binding strengths="),
        std_delim(delimited(tag("{"), many1(std_delim(string_f64)), tag("}"))),
    ))(input)?;

    // todo: checks

    let bonds = match bondnames {
        Some(b) => bondstrengths
            .iter()
            .zip(b)
            .map(|(s, n)| parser::Bond {
                name: parser::GlueIdent::Name(n.to_string()),
                strength: *s,
            })
            .collect(),
        None => bondstrengths
            .iter()
            .enumerate()
            .map(|(i, s)| parser::Bond {
                name: parser::GlueIdent::Num((i + 1) as u32),
                strength: *s,
            })
            .collect(),
    };

    // Todo: glue defs go here

    let (input, options) = xgrow_args(input)?;

    all_consuming(rsc)(input)?;

    Ok((
        input,
        parser::TileSet {
            tiles,
            bonds,
            options,
            cover_strands: None,
        },
    ))

    //Ok((input, ()));
}

enum XgrowArgs<'a> {
    Block(usize),
    Size(usize),
    Gse(f64),
    Gmc(f64),
    UpdateRate(u64),
    Unhandled(&'a str),
    Seed(parser::ParsedSeed),
}

fn arg_block(input: &str) -> IResult<&str, XgrowArgs> {
    map(preceded(tag("block="), take_u32), |x| {
        XgrowArgs::Block(x as usize)
    })(input)
}

fn arg_size(input: &str) -> IResult<&str, XgrowArgs> {
    map(preceded(tag("size="), take_u32), |x| {
        XgrowArgs::Size(x as usize)
    })(input)
}

fn arg_gse(input: &str) -> IResult<&str, XgrowArgs> {
    map(preceded(tag("Gse="), string_f64), |x| XgrowArgs::Gse(x))(input)
}

fn arg_gmc(input: &str) -> IResult<&str, XgrowArgs> {
    map(preceded(tag("Gmc="), string_f64), |x| XgrowArgs::Gmc(x))(input)
}

fn arg_update_rate(input: &str) -> IResult<&str, XgrowArgs> {
    map(preceded(tag("update_rate="), take_u32), |x| {
        XgrowArgs::UpdateRate(x as u64)
    })(input)
}

fn arg_seed(input: &str) -> IResult<&str, XgrowArgs> {
    preceded(
        tag("seed="),
        map(
            tuple((take_u32, tag(","), take_u32, tag(","), take_u32)),
            |(x, _, y, _, t)| {
                XgrowArgs::Seed(parser::ParsedSeed::Single(
                    y as usize, x as usize, t as usize,
                ))
            },
        ),
    )(input)
}

fn unhandled_option(input: &str) -> IResult<&str, XgrowArgs> {
    map(is_not(" \t\r\n%"), |x| XgrowArgs::Unhandled(x))(input)
}

fn xgrow_args(input: &str) -> IResult<&str, parser::Args> {
    let mut args = parser::Args::default();

    let parsers = (
        arg_block,
        arg_size,
        arg_gmc,
        arg_gse,
        arg_update_rate,
        arg_seed,
        unhandled_option,
    );

    let mut i2 = input;

    while let Ok((input, x)) = std_delim(alt(parsers))(i2) {
        match x {
            XgrowArgs::Block(n) => {
                args.block = n;
            }
            XgrowArgs::Size(n) => {
                args.size = n;
            }
            XgrowArgs::Gse(x) => {
                args.gse = x;
            }
            XgrowArgs::Gmc(x) => {
                args.gmc = x;
            }
            XgrowArgs::Unhandled(u) => {
                println!("Warning: \"{}\" unhandled.", u);
            }
            XgrowArgs::UpdateRate(x) => {
                args.update_rate = x;
            }
            XgrowArgs::Seed(x) => {
                args.seed = x;
            }
        }
        i2 = input;
    }

    if let parser::ParsedSeed::None() = args.seed {
        args.seed = parser::ParsedSeed::Single(args.size - 2, args.size - 2, 1);
    }

    args.fission = FissionHandling::NoFission;

    Ok((i2, args))
}

pub fn parse_xgrow(file: String) -> Result<parser::TileSet, Box<dyn Error>> {
    let mut f = File::open(file)?;

    let mut tilestring = String::new();
    f.read_to_string(&mut tilestring)?;

    let (_, parsed) = parse(tilestring.as_str()).unwrap();

    Ok(parsed)
}
