use chrono::{DateTime, Utc};
use ltx_core::{HardwareInfo, ModelEntry, ResolvedRequest};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct JobMetadata {
    pub created_at: DateTime<Utc>,
    pub resolved_request: ResolvedRequest,
    pub model_entry: Option<ModelEntry>,
    pub hardware: Option<HardwareInfo>,
    pub status: String,
    pub error: Option<serde_json::Value>,
    pub mock: bool,
}
