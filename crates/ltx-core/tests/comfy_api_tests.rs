use ltx_core::comfy_api::workflow_to_api_prompt;
use ltx_core::template_patcher::{apply_template_patch, TemplatePatchRequest};
use serde_json::Value;

#[test]
fn canonical_workflow_converts_to_comfy_api_prompt_dependency_closure() {
    let workflow = include_str!("../../../fixtures/ltx23_comfy_template.json");
    let api = workflow_to_api_prompt(workflow).expect("canonical workflow should convert");

    assert_eq!(api.output_node_ids, vec!["75"]);
    assert!(api.prompt.get("75").is_some());
    assert!(api.prompt.get("242").is_some());
    assert!(api.prompt.get("236").is_some());
    assert!(
        api.prompt.get("267").is_none(),
        "subgraph wrapper must not be executed"
    );
    assert_no_reroute_nodes(&api.prompt);
    assert_eq!(api.prompt["75"]["class_type"], "SaveVideo");
    assert_eq!(api.prompt["75"]["inputs"]["video"], link("242", 0));
    assert_eq!(
        api.prompt["243"]["inputs"]["device"],
        Value::String("default".to_string())
    );
    assert_eq!(
        api.prompt["209"]["inputs"]["sampler_name"],
        Value::String("euler_ancestral_cfg_pp".to_string())
    );
    assert_eq!(
        api.prompt["246"]["inputs"]["sampler_name"],
        Value::String("euler_cfg_pp".to_string())
    );
}

#[test]
fn patched_workflow_converts_with_requested_values() {
    let workflow = include_str!("../../../fixtures/ltx23_comfy_template.json");
    let manifest = include_str!("../../../fixtures/ltx23_template_manifest.example.json");
    let patched = apply_template_patch(
        workflow,
        manifest,
        &TemplatePatchRequest {
            prompt: "a cat walking".to_string(),
            negative_prompt: "bad quality".to_string(),
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
        },
    )
    .unwrap();

    let api = ltx_core::comfy_api::workflow_value_to_api_prompt(&patched.workflow)
        .expect("patched workflow should convert");

    assert_eq!(api.prompt["266"]["inputs"]["value"], "a cat walking");
    assert_eq!(api.prompt["247"]["inputs"]["text"], "bad quality");
    assert_eq!(api.prompt["237"]["inputs"]["noise_seed"], 12345);
    assert_eq!(api.prompt["257"]["inputs"]["value"], 768);
    assert_eq!(api.prompt["258"]["inputs"]["value"], 512);
    assert_eq!(api.prompt["75"]["inputs"]["video"], link("242", 0));
    assert_no_reroute_nodes(&api.prompt);
}

fn link(node_id: &str, slot: u64) -> Value {
    Value::Array(vec![
        Value::String(node_id.to_string()),
        Value::Number(slot.into()),
    ])
}

fn assert_no_reroute_nodes(prompt: &Value) {
    let prompt = prompt.as_object().expect("api prompt must be an object");
    for (id, node) in prompt {
        assert_ne!(
            node["class_type"], "Reroute",
            "node {id} must not be executed"
        );
    }
}
