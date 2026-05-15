use std::fs;

#[test]
fn model_registry_parses_example() {
    let raw = fs::read_to_string("../../configs/model_registry.example.toml").unwrap();
    let registry: ltx_runner_test_types::ModelRegistry = toml::from_str(&raw).unwrap();
    assert!(registry.models.contains_key("ltx2_3_distilled_fp8"));
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

mod ltx_runner_test_types {
    use ltx_core::ModelEntry;
    use serde::Deserialize;
    use std::collections::BTreeMap;

    #[derive(Debug, Deserialize)]
    pub struct ModelRegistry {
        pub models: BTreeMap<String, ModelEntry>,
    }
}
