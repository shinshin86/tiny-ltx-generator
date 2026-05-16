use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComfyApiPrompt {
    pub prompt: Value,
    pub output_node_ids: Vec<String>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ComfyApiError {
    #[error("workflow JSON is invalid: {0}")]
    InvalidWorkflow(String),
    #[error("workflow is missing a subgraph")]
    MissingSubgraph,
    #[error("workflow is missing top-level SaveVideo")]
    MissingSaveVideo,
    #[error("workflow is missing subgraph output source")]
    MissingSubgraphOutput,
    #[error("workflow link is missing: {0}")]
    MissingLink(u64),
    #[error("workflow node is missing: {0}")]
    MissingNode(i64),
    #[error("node `{node_type}` has no API widget mapping")]
    MissingWidgetMapping { node_type: String },
    #[error("node `{node_type}` is missing widget index {widget_index}")]
    MissingWidget {
        node_type: String,
        widget_index: usize,
    },
}

#[derive(Debug, Clone)]
struct Link {
    origin_id: i64,
    origin_slot: u64,
}

pub fn workflow_to_api_prompt(workflow_raw: &str) -> Result<ComfyApiPrompt, ComfyApiError> {
    let workflow: Value = serde_json::from_str(workflow_raw)
        .map_err(|err| ComfyApiError::InvalidWorkflow(err.to_string()))?;
    workflow_value_to_api_prompt(&workflow)
}

pub fn workflow_value_to_api_prompt(workflow: &Value) -> Result<ComfyApiPrompt, ComfyApiError> {
    let subgraph = workflow
        .pointer("/definitions/subgraphs/0")
        .ok_or(ComfyApiError::MissingSubgraph)?;
    let save_video = top_level_nodes(workflow)
        .into_iter()
        .find(|node| node_type(node) == Some("SaveVideo"))
        .ok_or(ComfyApiError::MissingSaveVideo)?;
    let save_video_id = node_id(save_video)?;
    let output_source = subgraph_output_source(subgraph)?;

    let mut node_by_id = BTreeMap::new();
    for node in subgraph_nodes(subgraph) {
        node_by_id.insert(node_id(node)?, node);
    }
    node_by_id.insert(save_video_id, save_video);

    let link_by_id = link_map(subgraph)?;
    let mut needed = BTreeSet::new();
    collect_dependencies(
        output_source.origin_id,
        &node_by_id,
        &link_by_id,
        &mut needed,
    )?;
    needed.insert(save_video_id);

    let mut prompt = Map::new();
    for id in needed {
        let node = node_by_id.get(&id).ok_or(ComfyApiError::MissingNode(id))?;
        let class_type = node_type(node)
            .ok_or(ComfyApiError::MissingNode(id))?
            .to_string();
        let mut inputs = if id == save_video_id {
            Map::new()
        } else {
            node_inputs(node, &link_by_id)?
        };
        if id == save_video_id {
            inputs.insert(
                "video".to_string(),
                json!([
                    output_source.origin_id.to_string(),
                    output_source.origin_slot
                ]),
            );
        }
        add_widget_inputs(node, &class_type, &mut inputs)?;
        prompt.insert(
            id.to_string(),
            json!({
                "class_type": class_type,
                "inputs": inputs,
            }),
        );
    }

    Ok(ComfyApiPrompt {
        prompt: Value::Object(prompt),
        output_node_ids: vec![save_video_id.to_string()],
    })
}

fn top_level_nodes(workflow: &Value) -> Vec<&Value> {
    workflow
        .get("nodes")
        .and_then(Value::as_array)
        .map(|nodes| nodes.iter().collect())
        .unwrap_or_default()
}

fn subgraph_nodes(subgraph: &Value) -> Vec<&Value> {
    subgraph
        .get("nodes")
        .and_then(Value::as_array)
        .map(|nodes| nodes.iter().collect())
        .unwrap_or_default()
}

fn subgraph_output_source(subgraph: &Value) -> Result<Link, ComfyApiError> {
    let output_link_ids = subgraph
        .get("outputs")
        .and_then(Value::as_array)
        .and_then(|outputs| outputs.first())
        .and_then(|output| output.get("linkIds"))
        .and_then(Value::as_array)
        .ok_or(ComfyApiError::MissingSubgraphOutput)?;
    let output_link_id = output_link_ids
        .first()
        .and_then(Value::as_u64)
        .ok_or(ComfyApiError::MissingSubgraphOutput)?;
    let links = link_map(subgraph)?;
    links
        .get(&output_link_id)
        .cloned()
        .ok_or(ComfyApiError::MissingSubgraphOutput)
}

fn link_map(graph: &Value) -> Result<BTreeMap<u64, Link>, ComfyApiError> {
    let mut links = BTreeMap::new();
    for link in graph
        .get("links")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(id) = link.get("id").and_then(Value::as_u64) else {
            continue;
        };
        let Some(origin_id) = link.get("origin_id").and_then(Value::as_i64) else {
            continue;
        };
        let Some(origin_slot) = link.get("origin_slot").and_then(Value::as_u64) else {
            continue;
        };
        links.insert(
            id,
            Link {
                origin_id,
                origin_slot,
            },
        );
    }
    Ok(links)
}

fn collect_dependencies(
    node_id_value: i64,
    node_by_id: &BTreeMap<i64, &Value>,
    link_by_id: &BTreeMap<u64, Link>,
    needed: &mut BTreeSet<i64>,
) -> Result<(), ComfyApiError> {
    if node_id_value < 0 || !needed.insert(node_id_value) {
        return Ok(());
    }
    let node = node_by_id
        .get(&node_id_value)
        .ok_or(ComfyApiError::MissingNode(node_id_value))?;
    for input in input_array(node) {
        if let Some(link_id) = input.get("link").and_then(Value::as_u64) {
            let link = link_by_id
                .get(&link_id)
                .ok_or(ComfyApiError::MissingLink(link_id))?;
            collect_dependencies(link.origin_id, node_by_id, link_by_id, needed)?;
        }
    }
    Ok(())
}

fn node_inputs(
    node: &Value,
    link_by_id: &BTreeMap<u64, Link>,
) -> Result<Map<String, Value>, ComfyApiError> {
    let mut inputs = Map::new();
    let widgets = widget_values(node);
    let mut widget_cursor = 0;
    for input in input_array(node) {
        let name = input
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if let Some(link_id) = input.get("link").and_then(Value::as_u64) {
            let link = link_by_id
                .get(&link_id)
                .ok_or(ComfyApiError::MissingLink(link_id))?;
            if link.origin_id >= 0 {
                inputs.insert(name, json!([link.origin_id.to_string(), link.origin_slot]));
                continue;
            }
        }
        if input.get("widget").is_some() {
            if let Some(value) = widgets.get(widget_cursor) {
                inputs.insert(name, value.clone());
                widget_cursor += 1;
            }
        }
    }
    Ok(inputs)
}

fn add_widget_inputs(
    node: &Value,
    class_type: &str,
    inputs: &mut Map<String, Value>,
) -> Result<(), ComfyApiError> {
    let widgets = widget_values(node);
    for (index, name) in widget_mapping(class_type)? {
        if inputs.contains_key(name) {
            continue;
        }
        let value = widgets
            .get(index)
            .ok_or_else(|| ComfyApiError::MissingWidget {
                node_type: class_type.to_string(),
                widget_index: index,
            })?
            .clone();
        inputs.insert(name.to_string(), value);
    }
    Ok(())
}

fn widget_mapping(class_type: &str) -> Result<Vec<(usize, &'static str)>, ComfyApiError> {
    let mapping = match class_type {
        "PrimitiveStringMultiline" => vec![(0, "value")],
        "PrimitiveInt" => vec![(0, "value")],
        "PrimitiveBoolean" => vec![(0, "value")],
        "CheckpointLoaderSimple" => vec![(0, "ckpt_name")],
        "LTXAVTextEncoderLoader" => vec![(0, "text_encoder"), (1, "ckpt_name"), (2, "device")],
        "CLIPTextEncode" => vec![(0, "text")],
        "LoraLoaderModelOnly" => vec![(0, "lora_name"), (1, "strength_model")],
        "LatentUpscaleModelLoader" => vec![(0, "model_name")],
        "RandomNoise" => vec![(0, "noise_seed")],
        "KSamplerSelect" => vec![(0, "sampler_name")],
        "ManualSigmas" => vec![(0, "sigmas")],
        "CFGGuider" => vec![(0, "cfg")],
        "LTXVConditioning" => vec![(0, "frame_rate")],
        "EmptyLTXVLatentVideo" => vec![
            (0, "width"),
            (1, "height"),
            (2, "length"),
            (3, "batch_size"),
        ],
        "LTXVEmptyLatentAudio" => vec![(0, "frames_number"), (1, "frame_rate"), (2, "batch_size")],
        "EmptyImage" => vec![(0, "width"), (1, "height"), (2, "batch_size"), (3, "color")],
        "ResizeImagesByLongerEdge" => vec![(0, "longer_edge")],
        "ResizeImageMaskNode" => vec![
            (0, "resize_type"),
            (1, "resize_type.width"),
            (2, "resize_type.height"),
            (3, "crop"),
            (4, "interpolation"),
        ],
        "LTXVPreprocess" => vec![(0, "padding")],
        "LTXVImgToVideoInplace" => vec![(0, "strength"), (1, "replace_latent")],
        "ComfyMathExpression" => vec![(0, "expression")],
        "VAEDecodeTiled" => vec![
            (0, "tile_size"),
            (1, "overlap"),
            (2, "temporal_size"),
            (3, "temporal_overlap"),
        ],
        "CreateVideo" => vec![(0, "fps")],
        "SaveVideo" => vec![(0, "filename_prefix"), (1, "format"), (2, "codec")],
        "LTXVConcatAVLatent"
        | "SamplerCustomAdvanced"
        | "LTXVSeparateAVLatent"
        | "LTXVCropGuides"
        | "LTXVLatentUpsampler"
        | "LTXVAudioVAELoader"
        | "LTXVAudioVAEDecode"
        | "Reroute" => Vec::new(),
        other => {
            return Err(ComfyApiError::MissingWidgetMapping {
                node_type: other.to_string(),
            })
        }
    };
    Ok(mapping)
}

fn input_array(node: &Value) -> impl Iterator<Item = &Value> {
    node.get("inputs")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
}

fn widget_values(node: &Value) -> &[Value] {
    node.get("widgets_values")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

fn node_id(node: &Value) -> Result<i64, ComfyApiError> {
    node.get("id")
        .and_then(Value::as_i64)
        .ok_or(ComfyApiError::MissingNode(-1))
}

fn node_type(node: &Value) -> Option<&str> {
    node.get("type")
        .or_else(|| node.get("class_type"))
        .and_then(Value::as_str)
}
