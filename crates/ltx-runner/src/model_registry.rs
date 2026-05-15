use anyhow::{Context, Result};
use ltx_core::{ModelEntry, ModelId, ProfileId};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistry {
    pub models: BTreeMap<String, ModelEntry>,
}

impl ModelRegistry {
    pub fn load(path: &str) -> Result<Self> {
        let raw =
            fs::read_to_string(path).with_context(|| format!("read model registry {path}"))?;
        toml::from_str::<Self>(&raw).with_context(|| format!("parse model registry {path}"))
    }

    pub fn select(&self, requested: ModelId, profile: ProfileId) -> Option<(ModelId, ModelEntry)> {
        if requested != ModelId::Auto {
            let key = model_key(requested);
            return self
                .models
                .get(key)
                .cloned()
                .map(|entry| (requested, entry));
        }
        preferred_models(profile).into_iter().find_map(|id| {
            self.models
                .get(model_key(id))
                .filter(|entry| is_configured(entry))
                .cloned()
                .map(|entry| (id, entry))
        })
    }
}

pub fn model_key(model: ModelId) -> &'static str {
    match model {
        ModelId::Auto => "auto",
        ModelId::Ltx2_3Full => "ltx2_3_full",
        ModelId::Ltx2_3Fp8 => "ltx2_3_fp8",
        ModelId::Ltx2_3DistilledFp8 => "ltx2_3_distilled_fp8",
        ModelId::Ltx2_3Distilled => "ltx2_3_distilled",
        ModelId::Ltxv13bDistilledFp8 => "ltxv_13b_distilled_fp8",
    }
}

pub fn preferred_models(profile: ProfileId) -> Vec<ModelId> {
    match profile {
        ProfileId::ColabTiny | ProfileId::ColabEco | ProfileId::NoGpu | ProfileId::Auto => vec![
            ModelId::Ltx2_3DistilledFp8,
            ModelId::Ltx2_3Distilled,
            ModelId::Ltx2_3Fp8,
            ModelId::Ltxv13bDistilledFp8,
        ],
        ProfileId::ColabBalanced => vec![
            ModelId::Ltx2_3Fp8,
            ModelId::Ltx2_3DistilledFp8,
            ModelId::Ltx2_3Distilled,
            ModelId::Ltx2_3Full,
        ],
        ProfileId::ColabQuality => vec![
            ModelId::Ltx2_3Fp8,
            ModelId::Ltx2_3Full,
            ModelId::Ltx2_3DistilledFp8,
            ModelId::Ltx2_3Distilled,
        ],
    }
}

pub fn missing_paths(entry: &ModelEntry) -> Vec<String> {
    let mut missing = Vec::new();
    if entry
        .checkpoint_path
        .as_deref()
        .map(str::trim)
        .unwrap_or("")
        .is_empty()
    {
        missing.push("checkpoint_path".to_string());
    }
    missing.extend(
        [
            entry.checkpoint_path.as_deref(),
            entry.gemma_root.as_deref(),
            entry.text_encoder_path.as_deref(),
            entry.vae_path.as_deref(),
            entry.config_path.as_deref(),
        ]
        .into_iter()
        .flatten()
        .filter(|path| !path.trim().is_empty() && !Path::new(path).exists())
        .map(ToOwned::to_owned),
    );
    missing
}

fn is_configured(entry: &ModelEntry) -> bool {
    entry
        .checkpoint_path
        .as_deref()
        .map(|path| !path.trim().is_empty())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use tempfile::NamedTempFile;

    fn entry(path: Option<String>) -> ModelEntry {
        ModelEntry {
            display_name: "test".to_string(),
            checkpoint_path: path,
            gemma_root: Some("".to_string()),
            text_encoder_path: None,
            vae_path: None,
            spatial_upsampler_path: None,
            temporal_upsampler_path: None,
            config_path: None,
            supports_audio: false,
            supports_t2v: true,
            supports_i2v: true,
            supports_fp8_cast: false,
            supports_fp8_scaled_mm: false,
            quantization: Some("none".to_string()),
            preferred_profiles: vec![ProfileId::ColabTiny],
            notes: None,
            extra: BTreeMap::new(),
        }
    }

    #[test]
    fn missing_paths_ignores_empty_paths() {
        assert_eq!(
            missing_paths(&entry(Some("".to_string()))),
            vec!["checkpoint_path"]
        );
    }

    #[test]
    fn missing_paths_reports_absent_files() {
        let missing = missing_paths(&entry(Some(
            "/definitely/missing/model.safetensors".to_string(),
        )));
        assert_eq!(missing, vec!["/definitely/missing/model.safetensors"]);
    }

    #[test]
    fn missing_paths_accepts_existing_files() {
        let file = NamedTempFile::new().unwrap();
        assert!(missing_paths(&entry(Some(file.path().display().to_string()))).is_empty());
    }

    #[test]
    fn auto_model_order_prefers_distilled_for_tiny() {
        let models = preferred_models(ProfileId::ColabTiny);
        assert_eq!(models[0], ModelId::Ltx2_3DistilledFp8);
    }

    #[test]
    fn auto_selection_skips_unconfigured_models() {
        let configured = NamedTempFile::new().unwrap();
        let mut models = BTreeMap::new();
        models.insert(
            "ltx2_3_distilled_fp8".to_string(),
            entry(Some("".to_string())),
        );
        models.insert(
            "ltx2_3_distilled".to_string(),
            entry(Some(configured.path().display().to_string())),
        );
        let registry = ModelRegistry { models };
        let (model, _) = registry
            .select(ModelId::Auto, ProfileId::ColabTiny)
            .unwrap();
        assert_eq!(model, ModelId::Ltx2_3Distilled);
    }
}
