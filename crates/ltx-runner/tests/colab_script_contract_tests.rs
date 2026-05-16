use std::fs;

#[test]
fn setup_script_installs_comfy_ltx_and_downloads_template_models_to_comfy_dirs() {
    let script = fs::read_to_string("../../scripts/setup_comfy_ltx23_colab.sh").unwrap();

    assert!(script.contains("https://github.com/comfyanonymous/ComfyUI.git"));
    assert!(script.contains("https://github.com/Lightricks/ComfyUI-LTXVideo.git"));
    assert!(script.contains("Comfy-Org/ltx-2"));
    assert!(script.contains("Lightricks/LTX-2.3-fp8"));
    assert!(script.contains("Lightricks/LTX-2.3"));
    assert!(script.contains("models/checkpoints"));
    assert!(script.contains("models/loras"));
    assert!(script.contains("models/text_encoders"));
    assert!(script.contains("models/latent_upscale_models"));
    assert!(script.contains("ltx-2.3-22b-dev-fp8.safetensors"));
    assert!(script.contains("ltx-2.3-22b-distilled-lora-384.safetensors"));
    assert!(script.contains("ltx-2.3-spatial-upscaler-x2-1.1.safetensors"));
    assert!(script.contains("gemma_3_12B_it_fp4_mixed.safetensors"));
    assert!(script.contains("HF_TOKEN"));
}

#[test]
fn setup_script_remains_colab_local_and_does_not_add_service_or_mcp_behavior() {
    let script = fs::read_to_string("../../scripts/setup_comfy_ltx23_colab.sh").unwrap();

    assert!(!script.contains("ngrok"));
    assert!(!script.contains("cloudflared"));
    assert!(!script.contains("jupyter"));
    assert!(!script.contains("server.py"));
    assert!(!script.contains("mcp"));
    assert!(!script.contains("/content/drive"));
}

#[test]
fn cat_smoke_runs_the_template_path_and_real_adapter() {
    let script = fs::read_to_string("../../scripts/run_ltx23_cat_smoke_colab.sh").unwrap();

    assert!(script.contains("cargo test --workspace"));
    assert!(script.contains("cargo build --release -p ltx-runner"));
    assert!(script.contains("ltx-runner prepare-run"));
    assert!(script.contains("fixtures/ltx23_comfy_template.json"));
    assert!(script.contains("fixtures/ltx23_template_manifest.example.json"));
    assert!(script.contains("scripts/comfy_headless_adapter.py"));
    assert!(script.contains("768"));
    assert!(script.contains("512"));
    assert!(script.contains("cat walking"));
    assert!(script.contains("test -s"));
    assert!(!script.contains("--dry-run"));
}
