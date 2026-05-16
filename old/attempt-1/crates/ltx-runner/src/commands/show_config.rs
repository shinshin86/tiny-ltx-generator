use anyhow::Result;

use crate::{cli::ShowConfigArgs, config::EngineConfig, json_output};

pub async fn run(args: ShowConfigArgs) -> Result<()> {
    let config = EngineConfig::load(args.config.as_deref())?;
    if args.json {
        json_output::print_json(&config)?;
    } else {
        eprintln!("{}", toml::to_string_pretty(&config)?);
    }
    Ok(())
}
