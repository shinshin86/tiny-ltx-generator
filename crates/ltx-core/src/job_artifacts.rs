use serde::Deserialize;
use std::path::{Path, PathBuf};

pub const REQUIRED_SUCCESS_FILES: &[&str] = &[
    "output.mp4",
    "metadata.json",
    "resolved_request.json",
    "events.jsonl",
    "worker_stats.json",
    "visual_check.json",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobArtifactReport {
    pub job_dir: PathBuf,
    pub required_files: Vec<String>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ArtifactError {
    #[error("job artifact is missing: {0}")]
    Missing(String),
    #[error("job artifact is empty: {0}")]
    Empty(String),
    #[error("visual_check.json is invalid JSON: {0}")]
    InvalidVisualCheck(String),
    #[error("visual sanity check did not pass: {0}")]
    VisualCheckFailed(String),
}

#[derive(Debug, Deserialize)]
struct VisualCheck {
    status: String,
    reason: Option<String>,
}

pub fn validate_success_artifacts(job_dir: &Path) -> Result<JobArtifactReport, ArtifactError> {
    for file in REQUIRED_SUCCESS_FILES {
        let path = job_dir.join(file);
        if !path.exists() {
            return Err(ArtifactError::Missing((*file).to_string()));
        }
        let metadata =
            std::fs::metadata(&path).map_err(|_| ArtifactError::Missing((*file).to_string()))?;
        if metadata.len() == 0 {
            return Err(ArtifactError::Empty((*file).to_string()));
        }
    }

    let visual_path = job_dir.join("visual_check.json");
    let visual_raw = std::fs::read_to_string(&visual_path)
        .map_err(|err| ArtifactError::InvalidVisualCheck(err.to_string()))?;
    let visual: VisualCheck = serde_json::from_str(&visual_raw)
        .map_err(|err| ArtifactError::InvalidVisualCheck(err.to_string()))?;
    if visual.status != "pass" {
        return Err(ArtifactError::VisualCheckFailed(
            visual.reason.unwrap_or_else(|| "unknown".to_string()),
        ));
    }

    Ok(JobArtifactReport {
        job_dir: job_dir.to_path_buf(),
        required_files: REQUIRED_SUCCESS_FILES
            .iter()
            .map(|file| (*file).to_string())
            .collect(),
    })
}
