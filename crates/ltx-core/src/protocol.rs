use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRequest {
    pub id: String,
    pub cmd: String,
    #[serde(default)]
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    #[serde(default)]
    pub payload: serde_json::Value,
}
