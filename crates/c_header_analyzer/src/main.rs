use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Header files to analyze
    #[arg(required = true)]
    headers: Vec<PathBuf>,

    /// Output JSON file (defaults to stdout)
    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let info = c_header_analyzer::analyze_headers(&args.headers)?;

    if let Some(out) = args.output {
        let f = std::fs::File::create(out)?;
        serde_json::to_writer_pretty(f, &info)?;
    } else {
        let s = serde_json::to_string_pretty(&info)?;
        println!("{}", s);
    }

    Ok(())
}
