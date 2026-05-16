use std::fs;

#[test]
fn model_registry_parses_example() {
    let raw = fs::read_to_string("../../configs/model_registry.example.toml").unwrap();
    let registry: ltx_runner_test_types::ModelRegistry = toml::from_str(&raw).unwrap();
    assert!(registry
        .models
        .contains_key("ltx2_3_dev_fp8_distilled_lora"));
    assert!(registry.models.contains_key("ltx2_3_distilled_fp8"));
    assert!(registry.models.contains_key("sulphur_2_dev_fp8mixed"));
}

#[test]
fn model_registry_quantization_flags_are_consistent() {
    let raw = fs::read_to_string("../../configs/model_registry.example.toml").unwrap();
    let registry: ltx_runner_test_types::ModelRegistry = toml::from_str(&raw).unwrap();
    for (id, model) in &registry.models {
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

    let sulphur = registry.models.get("sulphur_2_dev_fp8mixed").unwrap();
    assert_eq!(sulphur.quantization.as_deref(), Some("none"));
    assert!(!sulphur.supports_fp8_cast);

    for (id, model) in &registry.models {
        if model
            .checkpoint_path
            .as_deref()
            .unwrap_or("")
            .to_ascii_lowercase()
            .contains("fp8")
        {
            assert_eq!(
                model.quantization.as_deref().unwrap_or("none"),
                "none",
                "{id} is an FP8 checkpoint entry and must not request fp8-cast"
            );
        }
    }

    let comfy_style = registry
        .models
        .get("ltx2_3_dev_fp8_distilled_lora")
        .unwrap();
    assert_eq!(comfy_style.quantization.as_deref(), Some("none"));
    assert!(!comfy_style.supports_fp8_cast);
    assert!(comfy_style
        .lora_path
        .as_deref()
        .unwrap_or("")
        .contains("distilled-lora"));
    assert_eq!(
        comfy_style.extra.get("backend").map(String::as_str),
        Some("comfy_ltx")
    );
    assert!(comfy_style
        .text_encoder_path
        .as_deref()
        .unwrap_or("")
        .contains("gemma_3_12B_it_fp4_mixed"));
    assert_eq!(comfy_style.lora_strength, Some(0.5));
}

#[test]
fn colab_download_scripts_track_default_low_vram_variant() {
    let download_script = fs::read_to_string("../../scripts/download_models_colab.sh").unwrap();
    let bootstrap_script = fs::read_to_string("../../scripts/bootstrap_colab.sh").unwrap();
    let registry_writer =
        fs::read_to_string("../../scripts/write_downloaded_model_registry.py").unwrap();
    let example_registry = fs::read_to_string("../../configs/model_registry.example.toml").unwrap();

    assert!(download_script
        .contains("VARIANT=\"${LTX_DOWNLOAD_VARIANT:-ltx2_3_dev_fp8_distilled_lora}\""));
    assert!(download_script.contains("ltx-2.3-22b-dev-fp8.safetensors"));
    assert!(download_script.contains("ltx-2.3-22b-distilled-lora-384.safetensors"));
    assert!(download_script.contains("gemma_3_12B_it_fp4_mixed.safetensors"));
    assert!(download_script.contains("ltx-2.3-spatial-upscaler-x2-1.1.safetensors"));
    assert!(bootstrap_script.contains("scripts/setup_comfy_ltx_colab.sh"));
    assert!(download_script.contains("ltx-2.3-22b-distilled.safetensors"));
    assert!(download_script.contains("sulphur_2_dev_fp8mixed"));
    assert!(registry_writer.contains("ltx-2.3-22b-distilled.safetensors"));
    assert!(registry_writer.contains("gemma_3_12B_it_fp4_mixed.safetensors"));
    assert!(registry_writer.contains("backend = \"comfy_ltx\""));
    assert!(registry_writer.contains("sulphur_dev_fp8mixed.safetensors"));
    assert!(registry_writer.contains("spatial_path=\"\""));
    assert!(registry_writer.contains("dev FP8 checkpoint plus distilled LoRA"));
    assert!(example_registry.contains("ltx-2.3-22b-distilled.safetensors"));
    assert!(example_registry.contains("gemma_3_12B_it_fp4_mixed.safetensors"));
    assert!(example_registry.contains("backend = \"comfy_ltx\""));
    assert!(example_registry.contains("ltx-2.3-spatial-upscaler-x2-1.1.safetensors"));
    assert!(example_registry.contains("sulphur_dev_fp8mixed.safetensors"));
    assert!(example_registry.contains("dev FP8 checkpoint plus distilled LoRA"));

    assert!(!download_script.contains("ltx-2.3-22b-distilled-1.1.safetensors"));
    assert!(!registry_writer.contains("ltx-2.3-22b-distilled-1.1.safetensors"));
    assert!(!example_registry.contains("ltx-2.3-22b-distilled-1.1.safetensors"));
}

#[test]
fn comfy_headless_runner_uses_native_ltxv_clip_loader() {
    let runner = fs::read_to_string("../../scripts/comfy_headless_ltx.py").unwrap();

    assert!(runner.contains("\"class_type\": \"LTXAVTextEncoderLoader\""));
    assert!(runner.contains("\"class_type\": \"LTXVEmptyLatentAudio\""));
    assert!(runner.contains("\"class_type\": \"LTXVConcatAVLatent\""));
    assert!(runner.contains("\"class_type\": \"LTXVLatentUpsampler\""));
    assert!(runner.contains("\"class_type\": \"LTXVAudioVAEDecode\""));
    assert!(runner.contains("\"class_type\": \"ManualSigmas\""));
    assert!(runner.contains("\"audio\": [\"30\", 0]"));
    assert!(runner.contains("\"22\": {\"class_type\": \"RandomNoise\", \"inputs\": {\"noise_seed\": second_stage_seed}}"));
    assert!(runner.contains("second_stage_seed = int(model.get(\"second_stage_seed\") or 42)"));
    assert!(runner.contains("if not _same_file(produced, output_path):"));
    assert!(runner.contains("0.99375, 0.9875, 0.98125, 0.975"));
    assert!(runner.contains("0.85, 0.7250, 0.4219, 0.0"));
    assert!(!runner.contains("\"class_type\": \"CLIPLoader\""));
    assert!(!runner.contains("\"class_type\": \"LTXVGemmaCLIPModelLoader\""));
    assert!(runner.contains("init_api_nodes=False"));
}

#[test]
fn balanced_examples_do_not_request_fps_above_profile_cap() {
    let batch = fs::read_to_string("../../configs/batch.example.jsonl").unwrap();
    for (idx, line) in batch.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line).unwrap();
        if value.get("profile").and_then(|v| v.as_str()) == Some("colab_balanced") {
            let fps = value.get("fps").and_then(|v| v.as_u64()).unwrap();
            assert!(
                fps <= 12,
                "configs/batch.example.jsonl line {} requests fps {} above colab_balanced cap",
                idx + 1,
                fps
            );
        }
    }

    let readme = fs::read_to_string("../../README.md").unwrap();
    let batch_doc = fs::read_to_string("../../docs/batch_generation.md").unwrap();
    assert!(!readme.contains("--fps 24 \\\n  --out-dir /content/outputs \\\n  --jsonl-events"));
    assert!(!batch_doc.contains(r#""fps":24,"profile":"colab_balanced""#));
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
