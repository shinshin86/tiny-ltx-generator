use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GenerationMode {
    TextToVideo,
    ImageToVideo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileId {
    Auto,
    NoGpu,
    ColabTiny,
    ColabEco,
    ColabBalanced,
    ColabQuality,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelId {
    Auto,
    Ltx2_3Full,
    Ltx2_3Fp8,
    Ltx2_3DistilledFp8,
    Ltx2_3Distilled,
    #[serde(rename = "sulphur_2_dev_bf16")]
    Sulphur2DevBf16,
    #[serde(rename = "sulphur_2_dev_fp8mixed")]
    Sulphur2DevFp8Mixed,
    #[serde(rename = "sulphur_2_distil_bf16")]
    Sulphur2DistilBf16,
    #[serde(rename = "ltxv_13b_distilled_fp8")]
    Ltxv13bDistilledFp8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRequest {
    pub mode: GenerationMode,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub input_image: Option<String>,
    pub profile: ProfileId,
    pub model: ModelId,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub frames: Option<u32>,
    pub fps: Option<u32>,
    pub steps: Option<u32>,
    pub seed: Option<u64>,
    pub guidance_scale: Option<f32>,
    pub output: Option<String>,
    pub out_dir: Option<String>,
    pub allow_auto_downgrade: bool,
    pub copy_to_drive: bool,
    pub mock: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DowngradeRecord {
    pub field: String,
    pub requested: serde_json::Value,
    pub actual: serde_json::Value,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedRequest {
    pub job_id: String,
    pub mode: GenerationMode,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub input_image: Option<String>,
    pub profile: ProfileId,
    pub model: ModelId,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
    pub steps: u32,
    pub seed: u64,
    pub guidance_scale: f32,
    pub output_path: String,
    pub job_dir: String,
    pub downgrades: Vec<DowngradeRecord>,
    pub mock: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub os: String,
    pub cwd: String,
    pub content_disk_available_gb: Option<f64>,
    pub drive_mounted: bool,
    pub python_version: Option<String>,
    pub uv_version: Option<String>,
    pub cargo_version: Option<String>,
    pub rustc_version: Option<String>,
    pub ffmpeg_version: Option<String>,
    pub nvidia_smi_available: bool,
    pub gpu_name: Option<String>,
    pub total_vram_mb: Option<u64>,
    pub free_vram_mb: Option<u64>,
    pub torch: Option<TorchInfo>,
    pub recommended_profile: ProfileId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TorchInfo {
    pub import_ok: bool,
    pub version: Option<String>,
    pub cuda_available: bool,
    pub cuda_version: Option<String>,
    pub device_count: u32,
    pub gpu_name: Option<String>,
    pub allocated_vram_mb: Option<u64>,
    pub reserved_vram_mb: Option<u64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub display_name: String,
    pub checkpoint_path: Option<String>,
    pub gemma_root: Option<String>,
    pub text_encoder_path: Option<String>,
    pub vae_path: Option<String>,
    pub spatial_upsampler_path: Option<String>,
    pub temporal_upsampler_path: Option<String>,
    pub config_path: Option<String>,
    pub supports_audio: bool,
    pub supports_t2v: bool,
    pub supports_i2v: bool,
    pub supports_fp8_cast: bool,
    pub supports_fp8_scaled_mm: bool,
    pub quantization: Option<String>,
    pub preferred_profiles: Vec<ProfileId>,
    pub notes: Option<String>,
    #[serde(default)]
    pub extra: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchJob {
    pub id: Option<String>,
    pub mode: Option<GenerationMode>,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub input_image: Option<String>,
    pub seed: Option<u64>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub frames: Option<u32>,
    pub fps: Option<u32>,
    pub steps: Option<u32>,
    pub guidance_scale: Option<f32>,
    pub profile: Option<ProfileId>,
    pub model: Option<ModelId>,
}
