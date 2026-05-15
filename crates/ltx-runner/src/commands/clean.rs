use anyhow::Result;
use serde_json::json;
use std::{fs, path::Path};

use crate::{cli::CleanArgs, config::EngineConfig, error::AppError, json_output};

pub async fn run(args: CleanArgs) -> Result<()> {
    if !args.yes {
        return Err(AppError::Validation(
            "clean requires --yes for non-interactive execution".to_string(),
        )
        .into());
    }
    let config = EngineConfig::load(None)?;
    let out_dir = args.out_dir.unwrap_or(config.output_dir);
    let tmp_dir = args.tmp_dir.unwrap_or(config.tmp_dir);
    let mut removed = Vec::new();
    for path in [&tmp_dir, &format!("{out_dir}/jobs")] {
        if Path::new(path).exists() {
            fs::remove_dir_all(path)?;
            removed.push(path.to_string());
        }
    }
    if args.json {
        json_output::print_json(&json!({"removed": removed}))?;
    } else {
        eprintln!("removed: {:?}", removed);
    }
    Ok(())
}
