use std::fs;

#[test]
fn adapter_is_thin_and_does_not_rebuild_comfy_graph_by_hand() {
    let script = fs::read_to_string("../../scripts/comfy_headless_adapter.py").unwrap();

    assert!(script.contains("adapter_request.json"));
    assert!(script.contains("REQUIRED_REQUEST_FIELDS"));
    assert!(script.contains("\"comfyui_headless\""));
    assert!(!script.contains("PromptExecutor"));
    assert!(!script.contains("prompt = {"));
    assert!(!script.contains("\"class_type\""));
    assert!(!script.contains("_build_prompt"));
}

#[test]
fn adapter_supports_dry_run_before_colab_execution_is_enabled() {
    let script = fs::read_to_string("../../scripts/comfy_headless_adapter.py").unwrap();

    assert!(script.contains("--dry-run"));
    assert!(script.contains("\"missing_workflow\""));
    assert!(script.contains("\"not_implemented\""));
}
