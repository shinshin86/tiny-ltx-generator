use ltx_core::template_patcher::{
    apply_template_patch, find_subgraph_node, PatchError, TemplatePatchRequest,
};
use serde_json::Value;

fn request() -> TemplatePatchRequest {
    TemplatePatchRequest {
        prompt: "a small orange cat walking through a sunlit room".to_string(),
        negative_prompt: "low quality, noisy, distorted".to_string(),
        width: 768,
        height: 512,
        duration_seconds: 4,
        fps: 12,
        seed: 12345,
        checkpoint: "ltx-2.3-22b-dev-fp8.safetensors".to_string(),
        text_encoder: "gemma_3_12B_it_fp4_mixed.safetensors".to_string(),
        distilled_lora: "ltx-2.3-22b-distilled-lora-384.safetensors".to_string(),
        lora_strength: 0.5,
        spatial_upscaler: "ltx-2.3-spatial-upscaler-x2-1.1.safetensors".to_string(),
    }
}

#[test]
fn template_patch_updates_only_manifested_user_controls() {
    let workflow = include_str!("../../../fixtures/ltx23_comfy_template_minimal.json");
    let manifest = include_str!("../../../fixtures/ltx23_template_manifest.example.json");

    let patched = apply_template_patch(workflow, manifest, &request()).expect("patch should pass");

    assert_eq!(patched.changed_controls.len(), 17);
    assert_widget(&patched.workflow, 266, 0, Value::String(request().prompt));
    assert_widget(&patched.workflow, 240, 0, Value::String(request().prompt));
    assert_widget(
        &patched.workflow,
        247,
        0,
        Value::String(request().negative_prompt),
    );
    assert_widget(&patched.workflow, 257, 0, Value::from(768));
    assert_widget(&patched.workflow, 258, 0, Value::from(512));
    assert_widget(&patched.workflow, 225, 0, Value::from(4));
    assert_widget(&patched.workflow, 260, 0, Value::from(12));
    assert_widget(
        &patched.workflow,
        236,
        0,
        Value::String("ltx-2.3-22b-dev-fp8.safetensors".to_string()),
    );
    assert_widget(
        &patched.workflow,
        243,
        0,
        Value::String("gemma_3_12B_it_fp4_mixed.safetensors".to_string()),
    );
    assert_widget(
        &patched.workflow,
        243,
        1,
        Value::String("ltx-2.3-22b-dev-fp8.safetensors".to_string()),
    );
    assert_widget(&patched.workflow, 237, 0, Value::from(12345));
    assert_widget(
        &patched.workflow,
        237,
        1,
        Value::String("fixed".to_string()),
    );
    assert_widget(&patched.workflow, 216, 0, Value::from(42));
    assert_widget(
        &patched.workflow,
        216,
        1,
        Value::String("fixed".to_string()),
    );
}

#[test]
fn canonical_comfy_template_can_be_patched_by_manifest() {
    let workflow = include_str!("../../../fixtures/ltx23_comfy_template.json");
    let manifest = include_str!("../../../fixtures/ltx23_template_manifest.example.json");

    let patched =
        apply_template_patch(workflow, manifest, &request()).expect("canonical patch should pass");

    assert_eq!(patched.changed_controls.len(), 17);
    assert_widget(&patched.workflow, 266, 0, Value::String(request().prompt));
    assert_widget(
        &patched.workflow,
        247,
        0,
        Value::String(request().negative_prompt),
    );
    assert_widget(&patched.workflow, 257, 0, Value::from(768));
    assert_widget(&patched.workflow, 258, 0, Value::from(512));
    assert_widget(&patched.workflow, 237, 0, Value::from(12345));
    assert_widget(
        &patched.workflow,
        237,
        1,
        Value::String("fixed".to_string()),
    );
}

#[test]
fn manifest_node_type_mismatch_is_rejected_before_generation() {
    let workflow = include_str!("../../../fixtures/ltx23_comfy_template_minimal.json");
    let manifest = include_str!("../../../fixtures/ltx23_template_manifest.example.json")
        .replace("\"PrimitiveStringMultiline\"", "\"CLIPTextEncode\"");

    let err =
        apply_template_patch(workflow, &manifest, &request()).expect_err("bad manifest must fail");

    assert!(matches!(err, PatchError::NodeTypeMismatch { .. }));
    assert!(err.to_string().contains("prompt"));
}

#[test]
fn missing_manifest_control_is_rejected_before_generation() {
    let workflow = include_str!("../../../fixtures/ltx23_comfy_template_minimal.json");
    let manifest = r#"{"controls":{}}"#;

    let err = apply_template_patch(workflow, manifest, &request())
        .expect_err("missing controls must fail");

    assert_eq!(err, PatchError::MissingControl("prompt".to_string()));
}

fn assert_widget(workflow: &Value, node_id: u64, widget_index: usize, expected: Value) {
    let node = find_subgraph_node(workflow, node_id).expect("node should exist");
    let actual = node
        .get("widgets_values")
        .and_then(Value::as_array)
        .and_then(|widgets| widgets.get(widget_index))
        .expect("widget should exist");
    assert_eq!(actual, &expected);
}
