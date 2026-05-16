use serde_json::Value;
use std::process::Command;

#[test]
fn validate_template_command_reports_contract_json() {
    let output = Command::new(env!("CARGO_BIN_EXE_ltx-runner"))
        .args([
            "validate-template",
            "--workflow",
            fixture("ltx23_comfy_template_minimal.json").as_str(),
            "--manifest",
            fixture("ltx23_template_manifest.example.json").as_str(),
            "--json",
        ])
        .output()
        .expect("runner should execute");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("stdout must be JSON");
    assert_eq!(value["valid"], true);
    assert_eq!(value["subgraph_name"], "Text to Video (LTX-2.3)");
    assert!(value["required_node_types"]
        .as_array()
        .unwrap()
        .contains(&Value::String("LTXAVTextEncoderLoader".to_string())));
}

#[test]
fn patch_template_command_outputs_patched_workflow_json() {
    let output = Command::new(env!("CARGO_BIN_EXE_ltx-runner"))
        .args([
            "patch-template",
            "--workflow",
            fixture("ltx23_comfy_template_minimal.json").as_str(),
            "--manifest",
            fixture("ltx23_template_manifest.example.json").as_str(),
            "--prompt",
            "a cat walking",
            "--negative-prompt",
            "bad quality",
            "--width",
            "768",
            "--height",
            "512",
            "--duration-seconds",
            "4",
            "--fps",
            "12",
            "--seed",
            "12345",
            "--checkpoint",
            "ltx-2.3-22b-dev-fp8.safetensors",
            "--text-encoder",
            "gemma_3_12B_it_fp4_mixed.safetensors",
            "--distilled-lora",
            "ltx-2.3-22b-distilled-lora-384.safetensors",
            "--lora-strength",
            "0.5",
            "--spatial-upscaler",
            "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        ])
        .output()
        .expect("runner should execute");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: Value = serde_json::from_slice(&output.stdout).expect("stdout must be JSON");
    assert_eq!(
        widget(&value, 266, 0),
        &Value::String("a cat walking".to_string())
    );
    assert_eq!(
        widget(&value, 247, 0),
        &Value::String("bad quality".to_string())
    );
    assert_eq!(widget(&value, 257, 0), &Value::from(768));
    assert_eq!(widget(&value, 237, 0), &Value::from(12345));
    assert_eq!(widget(&value, 237, 1), &Value::String("fixed".to_string()));
}

fn fixture(name: &str) -> String {
    format!("{}/../../fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn widget(workflow: &Value, node_id: u64, widget_index: usize) -> &Value {
    workflow["definitions"]["subgraphs"][0]["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|node| node["id"].as_u64() == Some(node_id))
        .unwrap()["widgets_values"]
        .as_array()
        .unwrap()
        .get(widget_index)
        .unwrap()
}
