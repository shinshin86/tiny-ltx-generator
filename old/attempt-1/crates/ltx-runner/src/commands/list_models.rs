use anyhow::Result;

use crate::{
    cli::ListModelsArgs, config::EngineConfig, json_output, model_registry::ModelRegistry,
};

pub async fn run(args: ListModelsArgs) -> Result<()> {
    let config = EngineConfig::load(None)?;
    let path = args.model_registry.unwrap_or(config.model_registry);
    let registry = ModelRegistry::load(&path)?;
    if args.json {
        json_output::print_json(&registry)?;
    } else {
        for (id, entry) in registry.models {
            eprintln!("{id}: {}", entry.display_name);
        }
    }
    Ok(())
}
