use crate::workflow_contract::{validate_ltx23_template, ContractError};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub struct TemplatePatchRequest {
    pub prompt: String,
    pub negative_prompt: String,
    pub width: u64,
    pub height: u64,
    pub duration_seconds: u64,
    pub fps: u64,
    pub seed: u64,
    pub checkpoint: String,
    pub text_encoder: String,
    pub distilled_lora: String,
    pub lora_strength: f64,
    pub spatial_upscaler: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchedWorkflow {
    pub workflow: Value,
    pub changed_controls: Vec<String>,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum PatchError {
    #[error(transparent)]
    Contract(#[from] ContractError),
    #[error("manifest JSON is invalid: {0}")]
    InvalidManifest(String),
    #[error("workflow JSON is invalid: {0}")]
    InvalidWorkflow(String),
    #[error("manifest is missing control `{0}`")]
    MissingControl(String),
    #[error("control `{control}` points to missing node id {node_id}")]
    MissingNode { control: String, node_id: u64 },
    #[error("control `{control}` expected node type `{expected}` but found `{actual}`")]
    NodeTypeMismatch {
        control: String,
        expected: String,
        actual: String,
    },
    #[error("control `{control}` points to missing widget index {widget_index}")]
    MissingWidget {
        control: String,
        widget_index: usize,
    },
}

#[derive(Debug, Deserialize)]
struct Manifest {
    controls: BTreeMap<String, ControlBinding>,
}

#[derive(Debug, Deserialize, Clone)]
struct ControlBinding {
    node_id: u64,
    node_type: String,
    widget_index: usize,
}

pub fn apply_template_patch(
    workflow_raw: &str,
    manifest_raw: &str,
    request: &TemplatePatchRequest,
) -> Result<PatchedWorkflow, PatchError> {
    validate_ltx23_template(workflow_raw)?;
    let manifest: Manifest = serde_json::from_str(manifest_raw)
        .map_err(|err| PatchError::InvalidManifest(err.to_string()))?;
    let mut workflow: Value = serde_json::from_str(workflow_raw)
        .map_err(|err| PatchError::InvalidWorkflow(err.to_string()))?;
    let mut changed_controls = Vec::new();

    let values = patch_values(request);
    for (control, value) in values {
        patch_control(&mut workflow, &manifest, control, value)?;
        changed_controls.push(control.to_string());
    }

    validate_ltx23_template(&workflow.to_string())?;
    Ok(PatchedWorkflow {
        workflow,
        changed_controls,
    })
}

fn patch_values(request: &TemplatePatchRequest) -> Vec<(&'static str, Value)> {
    vec![
        ("prompt", json!(request.prompt)),
        ("positive_prompt", json!(request.prompt)),
        ("negative_prompt", json!(request.negative_prompt)),
        ("width", json!(request.width)),
        ("height", json!(request.height)),
        ("duration_seconds", json!(request.duration_seconds)),
        ("fps", json!(request.fps)),
        ("checkpoint", json!(request.checkpoint)),
        ("text_encoder", json!(request.text_encoder)),
        ("text_encoder_checkpoint", json!(request.checkpoint)),
        ("distilled_lora", json!(request.distilled_lora)),
        ("lora_strength", json!(request.lora_strength)),
        ("spatial_upscaler", json!(request.spatial_upscaler)),
        ("first_stage_seed", json!(request.seed)),
        ("first_stage_seed_mode", json!("fixed")),
        ("second_stage_seed", json!(42)),
        ("second_stage_seed_mode", json!("fixed")),
    ]
}

fn patch_control(
    workflow: &mut Value,
    manifest: &Manifest,
    control: &'static str,
    value: Value,
) -> Result<(), PatchError> {
    let binding = manifest
        .controls
        .get(control)
        .ok_or_else(|| PatchError::MissingControl(control.to_string()))?;
    let node = find_subgraph_node_mut(workflow, binding.node_id).ok_or_else(|| {
        PatchError::MissingNode {
            control: control.to_string(),
            node_id: binding.node_id,
        }
    })?;
    let actual_type = node
        .get("type")
        .or_else(|| node.get("class_type"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if actual_type != binding.node_type {
        return Err(PatchError::NodeTypeMismatch {
            control: control.to_string(),
            expected: binding.node_type.clone(),
            actual: actual_type,
        });
    }

    let widgets = node
        .get_mut("widgets_values")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| PatchError::MissingWidget {
            control: control.to_string(),
            widget_index: binding.widget_index,
        })?;
    let widget =
        widgets
            .get_mut(binding.widget_index)
            .ok_or_else(|| PatchError::MissingWidget {
                control: control.to_string(),
                widget_index: binding.widget_index,
            })?;
    *widget = value;
    Ok(())
}

fn find_subgraph_node_mut(workflow: &mut Value, node_id: u64) -> Option<&mut Value> {
    workflow
        .get_mut("definitions")?
        .get_mut("subgraphs")?
        .as_array_mut()?
        .first_mut()?
        .get_mut("nodes")?
        .as_array_mut()?
        .iter_mut()
        .find(|node| node.get("id").and_then(Value::as_u64) == Some(node_id))
}

pub fn find_subgraph_node(workflow: &Value, node_id: u64) -> Option<&Value> {
    workflow
        .get("definitions")?
        .get("subgraphs")?
        .as_array()?
        .first()?
        .get("nodes")?
        .as_array()?
        .iter()
        .find(|node| node.get("id").and_then(Value::as_u64) == Some(node_id))
}
