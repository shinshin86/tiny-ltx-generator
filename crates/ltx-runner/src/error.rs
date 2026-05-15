use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("validation error: {0}")]
    Validation(String),
    #[error("missing model files: {0}")]
    MissingModel(String),
    #[error("no CUDA GPU is available")]
    NoGpu,
    #[error("CUDA out of memory: {0}")]
    CudaOom(String),
    #[error("worker crash: {0}")]
    WorkerCrash(String),
    #[error("ffmpeg error: {0}")]
    Ffmpeg(String),
    #[error("unsupported pipeline or model option: {0}")]
    Unsupported(String),
}

impl AppError {
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Validation(_) => 2,
            Self::MissingModel(_) => 3,
            Self::NoGpu => 4,
            Self::CudaOom(_) => 5,
            Self::WorkerCrash(_) => 6,
            Self::Ffmpeg(_) => 7,
            Self::Unsupported(_) => 8,
        }
    }
}
