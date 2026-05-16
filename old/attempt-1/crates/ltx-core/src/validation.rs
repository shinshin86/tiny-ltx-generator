use crate::{GenerationMode, ResolvedRequest};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("prompt must not be empty")]
    EmptyPrompt,
    #[error("width and height must be divisible by 32")]
    BadResolution,
    #[error("frames must be one of 33, 49, 65, 81, 97, 121, 161")]
    BadFrames,
    #[error("fps must be positive and within 1..=24")]
    BadFps,
    #[error("image-to-video requires --input-image")]
    MissingInputImage,
    #[error("input image extension must be png, jpg, jpeg, or webp")]
    BadInputImageExtension,
}

pub fn validate_resolved(req: &ResolvedRequest) -> Result<(), ValidationError> {
    if req.prompt.trim().is_empty() {
        return Err(ValidationError::EmptyPrompt);
    }
    if req.width % 32 != 0 || req.height % 32 != 0 {
        return Err(ValidationError::BadResolution);
    }
    if !matches!(req.frames, 33 | 49 | 65 | 81 | 97 | 121 | 161) {
        return Err(ValidationError::BadFrames);
    }
    if req.fps == 0 || req.fps > 24 {
        return Err(ValidationError::BadFps);
    }
    if req.mode == GenerationMode::ImageToVideo {
        let Some(path) = req.input_image.as_deref() else {
            return Err(ValidationError::MissingInputImage);
        };
        let lower = path.to_ascii_lowercase();
        if !(lower.ends_with(".png")
            || lower.ends_with(".jpg")
            || lower.ends_with(".jpeg")
            || lower.ends_with(".webp"))
        {
            return Err(ValidationError::BadInputImageExtension);
        }
    }
    Ok(())
}
