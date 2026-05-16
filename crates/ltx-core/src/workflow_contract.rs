use serde::Deserialize;
use std::collections::BTreeSet;

pub const REQUIRED_NODE_TYPES: &[&str] = &[
    "CheckpointLoaderSimple",
    "LTXAVTextEncoderLoader",
    "CLIPTextEncode",
    "LTXVConditioning",
    "EmptyLTXVLatentVideo",
    "LTXVAudioVAELoader",
    "LTXVEmptyLatentAudio",
    "LTXVConcatAVLatent",
    "LoraLoaderModelOnly",
    "ManualSigmas",
    "KSamplerSelect",
    "CFGGuider",
    "SamplerCustomAdvanced",
    "LTXVSeparateAVLatent",
    "LTXVCropGuides",
    "LatentUpscaleModelLoader",
    "LTXVLatentUpsampler",
    "VAEDecodeTiled",
    "LTXVAudioVAEDecode",
    "CreateVideo",
    "SaveVideo",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateContractReport {
    pub subgraph_name: String,
    pub node_count: usize,
    pub required_node_types: Vec<String>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ContractError {
    #[error("workflow JSON is invalid: {0}")]
    InvalidJson(String),
    #[error("workflow does not contain definitions.subgraphs[0]")]
    MissingSubgraph,
    #[error("workflow subgraph is missing required node types: {0:?}")]
    MissingRequiredNodes(Vec<String>),
    #[error(
        "template-driven execution is required; handcrafted graph construction is not allowed"
    )]
    HandcraftedGraph,
}

#[derive(Debug, Deserialize)]
struct Workflow {
    definitions: Option<Definitions>,
}

#[derive(Debug, Deserialize)]
struct Definitions {
    subgraphs: Vec<Subgraph>,
}

#[derive(Debug, Deserialize)]
struct Subgraph {
    name: Option<String>,
    nodes: Vec<Node>,
}

#[derive(Debug, Deserialize)]
struct Node {
    #[serde(rename = "type")]
    node_type: Option<String>,
    class_type: Option<String>,
}

pub fn validate_ltx23_template(raw: &str) -> Result<TemplateContractReport, ContractError> {
    if raw.contains("_build_prompt(")
        || raw.contains("\"class_type\":") && raw.contains("prompt = {")
    {
        return Err(ContractError::HandcraftedGraph);
    }

    let workflow: Workflow =
        serde_json::from_str(raw).map_err(|err| ContractError::InvalidJson(err.to_string()))?;
    let subgraph = workflow
        .definitions
        .and_then(|definitions| definitions.subgraphs.into_iter().next())
        .ok_or(ContractError::MissingSubgraph)?;

    let node_types: BTreeSet<String> = subgraph
        .nodes
        .iter()
        .filter_map(|node| node.node_type.clone().or(node.class_type.clone()))
        .collect();

    let missing: Vec<String> = REQUIRED_NODE_TYPES
        .iter()
        .filter(|required| !node_types.contains(**required))
        .map(|required| (*required).to_string())
        .collect();

    if !missing.is_empty() {
        return Err(ContractError::MissingRequiredNodes(missing));
    }

    Ok(TemplateContractReport {
        subgraph_name: subgraph
            .name
            .unwrap_or_else(|| "Text to Video (LTX-2.3)".to_string()),
        node_count: subgraph.nodes.len(),
        required_node_types: REQUIRED_NODE_TYPES
            .iter()
            .map(|node_type| (*node_type).to_string())
            .collect(),
    })
}
