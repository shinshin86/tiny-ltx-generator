use anyhow::Result;
use serde_json::json;
use std::{fs, path::Path};

use crate::{cli::InitArgs, config::EngineConfig, json_output, storage};

pub async fn run(args: InitArgs) -> Result<()> {
    let mut config = EngineConfig::load(args.config.as_deref())?;
    if let Some(path) = args.output_dir {
        config.output_dir = path;
    }
    if let Some(path) = args.tmp_dir {
        config.tmp_dir = path;
    }
    if let Some(path) = args.model_dir {
        config.model_dir = path;
    }
    storage::ensure_dirs(&[&config.output_dir, &config.tmp_dir, &config.model_dir])?;
    if Path::new("/content/drive").exists() {
        storage::ensure_dirs(&[
            &format!("{}/models", config.drive_root),
            &format!("{}/outputs", config.drive_root),
        ])?;
    }
    if !Path::new("configs/colab_engine.toml").exists() {
        fs::copy(
            "configs/colab_engine.example.toml",
            "configs/colab_engine.toml",
        )
        .ok();
    }
    if !Path::new("configs/model_registry.toml").exists() {
        fs::copy(
            "configs/model_registry.example.toml",
            "configs/model_registry.toml",
        )
        .ok();
    }
    if args.download_models {
        eprintln!("--download-models is handled by scripts/bootstrap_colab.sh via scripts/download_models_colab.sh. Set HF_TOKEN only when Hugging Face access requires it.");
    }
    if args.copy_models_to_content {
        eprintln!("--copy-models-to-content currently prepares directories only; copy or symlink exact paths from Drive after configuring the registry.");
    }
    let summary = json!({
        "output_dir": config.output_dir,
        "tmp_dir": config.tmp_dir,
        "model_dir": config.model_dir,
        "model_registry": "configs/model_registry.toml",
    });
    if args.json {
        json_output::print_json(&summary)?;
    } else {
        eprintln!("initialized tiny-ltx-generator paths");
    }
    Ok(())
}
