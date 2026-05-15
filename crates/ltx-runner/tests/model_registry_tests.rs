use std::fs;

#[test]
fn model_registry_parses_example() {
    let raw = fs::read_to_string("../../configs/model_registry.example.toml").unwrap();
    let registry: ltx_runner_test_types::ModelRegistry = toml::from_str(&raw).unwrap();
    assert!(registry.models.contains_key("ltx2_3_distilled_fp8"));
    assert!(registry.models.contains_key("sulphur_2_dev_fp8mixed"));
}

#[test]
fn model_registry_quantization_flags_are_consistent() {
    let raw = fs::read_to_string("../../configs/model_registry.example.toml").unwrap();
    let registry: ltx_runner_test_types::ModelRegistry = toml::from_str(&raw).unwrap();
    for (id, model) in registry.models {
        match model.quantization.as_deref().unwrap_or("none") {
            "none" => {}
            "fp8-cast" => assert!(
                model.supports_fp8_cast,
                "{id} advertises unsupported fp8-cast"
            ),
            "fp8-scaled-mm" => assert!(
                model.supports_fp8_scaled_mm,
                "{id} advertises unsupported fp8-scaled-mm"
            ),
            other => panic!("{id} has unknown quantization mode {other}"),
        }
    }
}

#[test]
fn colab_download_scripts_track_default_low_vram_variant() {
    let download_script = fs::read_to_string("../../scripts/download_models_colab.sh").unwrap();
    let registry_writer =
        fs::read_to_string("../../scripts/write_downloaded_model_registry.py").unwrap();
    let example_registry = fs::read_to_string("../../configs/model_registry.example.toml").unwrap();

    assert!(download_script.contains("VARIANT=\"${LTX_DOWNLOAD_VARIANT:-ltx2_3_distilled_fp8}\""));
    assert!(download_script.contains("ltx-2.3-22b-distilled.safetensors"));
    assert!(download_script.contains("sulphur_2_dev_fp8mixed"));
    assert!(registry_writer.contains("ltx-2.3-22b-distilled.safetensors"));
    assert!(registry_writer.contains("sulphur_dev_fp8mixed.safetensors"));
    assert!(registry_writer.contains("spatial_path=\"\""));
    assert!(example_registry.contains("ltx-2.3-22b-distilled.safetensors"));
    assert!(example_registry.contains("sulphur_dev_fp8mixed.safetensors"));

    assert!(!download_script.contains("ltx-2.3-22b-distilled-1.1.safetensors"));
    assert!(!registry_writer.contains("ltx-2.3-22b-distilled-1.1.safetensors"));
    assert!(!example_registry.contains("ltx-2.3-22b-distilled-1.1.safetensors"));
}

mod ltx_runner_test_types {
    use ltx_core::ModelEntry;
    use serde::Deserialize;
    use std::collections::BTreeMap;

    #[derive(Debug, Deserialize)]
    pub struct ModelRegistry {
        pub models: BTreeMap<String, ModelEntry>,
    }
}
