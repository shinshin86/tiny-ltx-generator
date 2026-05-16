use serde_json::Value;
use std::fs;
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
fn validate_template_command_accepts_canonical_comfy_template() {
    let output = Command::new(env!("CARGO_BIN_EXE_ltx-runner"))
        .args([
            "validate-template",
            "--workflow",
            fixture("ltx23_comfy_template.json").as_str(),
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
    assert_eq!(value["node_count"], 46);
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

#[test]
fn prepare_run_writes_adapter_request_and_required_job_inputs() {
    let temp = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_ltx-runner"))
        .args([
            "prepare-run",
            "--workflow",
            fixture("ltx23_comfy_template.json").as_str(),
            "--manifest",
            fixture("ltx23_template_manifest.example.json").as_str(),
            "--out-dir",
            temp.path().to_str().unwrap(),
            "--job-id",
            "cat-smoke-001",
            "--comfy-root",
            "/content/ComfyUI",
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
    let summary: Value = serde_json::from_slice(&output.stdout).expect("stdout must be JSON");
    let job_dir = temp.path().join("jobs/cat-smoke-001");
    assert_eq!(summary["job_id"], "cat-smoke-001");
    assert_eq!(summary["status"], "prepared");
    assert!(job_dir.join("patched_workflow.json").exists());
    assert!(job_dir.join("api_prompt.json").exists());
    assert!(job_dir.join("adapter_request.json").exists());
    assert!(job_dir.join("resolved_request.json").exists());
    assert!(job_dir.join("metadata.json").exists());
    assert!(job_dir.join("events.jsonl").exists());
    assert_eq!(
        fs::read_to_string(job_dir.join("prompt.txt")).unwrap(),
        "a cat walking\n"
    );

    let adapter: Value =
        serde_json::from_str(&fs::read_to_string(job_dir.join("adapter_request.json")).unwrap())
            .unwrap();
    assert_eq!(adapter["backend"], "comfyui_headless");
    assert_eq!(adapter["comfy_root"], "/content/ComfyUI");
    assert_eq!(
        adapter["api_prompt_path"].as_str().unwrap(),
        job_dir.join("api_prompt.json").to_string_lossy()
    );
    assert_eq!(adapter["output_node_ids"][0], "75");
    assert_eq!(
        adapter["output_path"].as_str().unwrap(),
        job_dir.join("output.mp4").to_string_lossy()
    );
    assert!(adapter["required_success_files"]
        .as_array()
        .unwrap()
        .contains(&Value::String("visual_check.json".to_string())));

    let patched: Value =
        serde_json::from_str(&fs::read_to_string(job_dir.join("patched_workflow.json")).unwrap())
            .unwrap();
    assert_eq!(
        widget(&patched, 266, 0),
        &Value::String("a cat walking".to_string())
    );

    let api_prompt: Value =
        serde_json::from_str(&fs::read_to_string(job_dir.join("api_prompt.json")).unwrap())
            .unwrap();
    assert_eq!(api_prompt["75"]["class_type"], "SaveVideo");
    assert_eq!(
        api_prompt["75"]["inputs"]["video"],
        Value::Array(vec![Value::String("242".to_string()), Value::from(0)])
    );
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
