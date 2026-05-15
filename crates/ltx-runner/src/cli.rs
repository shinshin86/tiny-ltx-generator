use clap::{Args, Parser, Subcommand, ValueEnum};
use ltx_core::{GenerationMode, ModelId, ProfileId};

#[derive(Debug, Parser)]
#[command(
    name = "ltx-runner",
    version,
    about = "Colab-first LTX video generator CLI"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    Check(CheckArgs),
    Init(InitArgs),
    Generate(GenerateArgs),
    Batch(BatchArgs),
    #[command(name = "worker-health")]
    WorkerHealth(JsonArgs),
    #[command(name = "list-profiles")]
    ListProfiles(JsonArgs),
    #[command(name = "list-models")]
    ListModels(ListModelsArgs),
    Clean(CleanArgs),
    #[command(name = "show-config")]
    ShowConfig(ShowConfigArgs),
}

#[derive(Debug, Args, Clone)]
pub struct JsonArgs {
    #[arg(long)]
    pub json: bool,
}

pub type CheckArgs = JsonArgs;

#[derive(Debug, Args, Clone)]
pub struct InitArgs {
    #[arg(long)]
    pub config: Option<String>,
    #[arg(long)]
    pub output_dir: Option<String>,
    #[arg(long)]
    pub tmp_dir: Option<String>,
    #[arg(long)]
    pub model_dir: Option<String>,
    #[arg(long)]
    pub download_models: bool,
    #[arg(long)]
    pub copy_models_to_content: bool,
    #[arg(long)]
    pub yes: bool,
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Args, Clone)]
pub struct GenerateArgs {
    #[arg(long, value_enum)]
    pub mode: ModeArg,
    #[arg(long)]
    pub prompt: String,
    #[arg(long)]
    pub negative_prompt: Option<String>,
    #[arg(long)]
    pub input_image: Option<String>,
    #[arg(long, value_enum, default_value = "auto")]
    pub profile: ProfileArg,
    #[arg(long, value_enum, default_value = "auto")]
    pub model: ModelArg,
    #[arg(long)]
    pub width: Option<u32>,
    #[arg(long)]
    pub height: Option<u32>,
    #[arg(long)]
    pub frames: Option<u32>,
    #[arg(long)]
    pub fps: Option<u32>,
    #[arg(long)]
    pub steps: Option<u32>,
    #[arg(long)]
    pub seed: Option<u64>,
    #[arg(long)]
    pub guidance_scale: Option<f32>,
    #[arg(long)]
    pub output: Option<String>,
    #[arg(long)]
    pub out_dir: Option<String>,
    #[arg(long)]
    pub config: Option<String>,
    #[arg(long)]
    pub model_registry: Option<String>,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub allow_auto_downgrade: bool,
    #[arg(long)]
    pub no_auto_downgrade: bool,
    #[arg(long)]
    pub json: bool,
    #[arg(long)]
    pub jsonl_events: bool,
    #[arg(long, default_value_t = false)]
    pub copy_to_drive: bool,
    #[arg(long)]
    pub mock: bool,
}

#[derive(Debug, Args, Clone)]
pub struct BatchArgs {
    #[arg(long)]
    pub jobs: String,
    #[arg(long, value_enum, default_value = "auto")]
    pub profile: ProfileArg,
    #[arg(long, value_enum, default_value = "auto")]
    pub model: ModelArg,
    #[arg(long)]
    pub out_dir: Option<String>,
    #[arg(long)]
    pub config: Option<String>,
    #[arg(long)]
    pub model_registry: Option<String>,
    #[arg(long)]
    pub json: bool,
    #[arg(long)]
    pub jsonl_events: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub continue_on_error: bool,
    #[arg(long)]
    pub stop_on_error: bool,
    #[arg(long)]
    pub max_jobs: Option<usize>,
    #[arg(long)]
    pub start_index: Option<usize>,
    #[arg(long, default_value_t = false)]
    pub copy_to_drive: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub keep_model_warm: bool,
    #[arg(long, default_value_t = false)]
    pub unload_between_jobs: bool,
    #[arg(long)]
    pub mock: bool,
}

#[derive(Debug, Args, Clone)]
pub struct ListModelsArgs {
    #[arg(long)]
    pub model_registry: Option<String>,
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Args, Clone)]
pub struct CleanArgs {
    #[arg(long)]
    pub out_dir: Option<String>,
    #[arg(long)]
    pub tmp_dir: Option<String>,
    #[arg(long)]
    pub yes: bool,
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Args, Clone)]
pub struct ShowConfigArgs {
    #[arg(long)]
    pub config: Option<String>,
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ModeArg {
    TextToVideo,
    ImageToVideo,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ProfileArg {
    #[value(name = "auto")]
    Auto,
    #[value(name = "no_gpu")]
    NoGpu,
    #[value(name = "colab_tiny")]
    ColabTiny,
    #[value(name = "colab_eco")]
    ColabEco,
    #[value(name = "colab_balanced")]
    ColabBalanced,
    #[value(name = "colab_quality")]
    ColabQuality,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ModelArg {
    #[value(name = "auto")]
    Auto,
    #[value(name = "ltx2_3_full")]
    Ltx2_3Full,
    #[value(name = "ltx2_3_fp8")]
    Ltx2_3Fp8,
    #[value(name = "ltx2_3_distilled_fp8")]
    Ltx2_3DistilledFp8,
    #[value(name = "ltx2_3_distilled")]
    Ltx2_3Distilled,
    #[value(name = "ltxv_13b_distilled_fp8")]
    Ltxv13bDistilledFp8,
}

impl From<ModeArg> for GenerationMode {
    fn from(value: ModeArg) -> Self {
        match value {
            ModeArg::TextToVideo => Self::TextToVideo,
            ModeArg::ImageToVideo => Self::ImageToVideo,
        }
    }
}

impl From<ProfileArg> for ProfileId {
    fn from(value: ProfileArg) -> Self {
        match value {
            ProfileArg::Auto => Self::Auto,
            ProfileArg::NoGpu => Self::NoGpu,
            ProfileArg::ColabTiny => Self::ColabTiny,
            ProfileArg::ColabEco => Self::ColabEco,
            ProfileArg::ColabBalanced => Self::ColabBalanced,
            ProfileArg::ColabQuality => Self::ColabQuality,
        }
    }
}

impl From<ModelArg> for ModelId {
    fn from(value: ModelArg) -> Self {
        match value {
            ModelArg::Auto => Self::Auto,
            ModelArg::Ltx2_3Full => Self::Ltx2_3Full,
            ModelArg::Ltx2_3Fp8 => Self::Ltx2_3Fp8,
            ModelArg::Ltx2_3DistilledFp8 => Self::Ltx2_3DistilledFp8,
            ModelArg::Ltx2_3Distilled => Self::Ltx2_3Distilled,
            ModelArg::Ltxv13bDistilledFp8 => Self::Ltxv13bDistilledFp8,
        }
    }
}
