use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{env, fs, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub output_dir: String,
    pub tmp_dir: String,
    pub model_dir: String,
    pub model_registry: String,
    pub python_worker: String,
    pub drive_root: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_uses_colab_paths() {
        let config = EngineConfig::default();
        assert_eq!(config.output_dir, "/content/outputs");
        assert_eq!(config.tmp_dir, "/content/ltx_tmp");
        assert_eq!(config.model_dir, "/content/models");
    }
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            output_dir: "/content/outputs".to_string(),
            tmp_dir: "/content/ltx_tmp".to_string(),
            model_dir: "/content/models".to_string(),
            model_registry: "configs/model_registry.toml".to_string(),
            python_worker: "py-worker/worker.py".to_string(),
            drive_root: "/content/drive/MyDrive/tiny-ltx-generator".to_string(),
        }
    }
}

impl EngineConfig {
    pub fn load(path: Option<&str>) -> Result<Self> {
        let path = path
            .map(ToOwned::to_owned)
            .or_else(|| env::var("LTX_ENGINE_CONFIG").ok())
            .unwrap_or_else(|| "configs/colab_engine.toml".to_string());
        let mut config = if Path::new(&path).exists() {
            let raw = fs::read_to_string(&path).with_context(|| format!("read config {path}"))?;
            toml::from_str::<EngineConfig>(&raw).with_context(|| format!("parse config {path}"))?
        } else {
            EngineConfig::default()
        };
        if let Ok(value) = env::var("LTX_OUTPUT_DIR") {
            config.output_dir = value;
        }
        if let Ok(value) = env::var("LTX_TMP_DIR") {
            config.tmp_dir = value;
        }
        if let Ok(value) = env::var("LTX_MODEL_DIR") {
            config.model_dir = value;
        }
        if let Ok(value) = env::var("LTX_MODEL_REGISTRY") {
            config.model_registry = value;
        }
        Ok(config)
    }
}
