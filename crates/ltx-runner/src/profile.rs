use anyhow::Result;
use ltx_core::{apply_profile_caps, defaults, resolve_profile, GenerationRequest, ResolvedRequest};
use uuid::Uuid;

use crate::{error::AppError, hardware, model_registry::ModelRegistry};

pub async fn resolve_request(
    req: GenerationRequest,
    worker_path: &str,
    registry: &ModelRegistry,
) -> Result<ResolvedRequest> {
    let hardware = hardware::inspect(false, worker_path).await?;
    let (profile, mut downgrades) =
        resolve_profile(req.profile, &hardware, req.allow_auto_downgrade).map_err(|msg| {
            if msg.contains("CUDA GPU") {
                AppError::NoGpu
            } else {
                AppError::Validation(msg)
            }
        })?;
    if profile == ltx_core::ProfileId::NoGpu && !req.mock {
        return Err(AppError::NoGpu.into());
    }
    let selected_model = registry
        .select(req.model, profile)
        .map(|(id, _)| id)
        .unwrap_or(req.model);
    let defaults = defaults(profile);
    let job_id = Uuid::new_v4().to_string();
    let out_dir = req
        .out_dir
        .clone()
        .unwrap_or_else(|| "/content/outputs".to_string());
    let job_dir = format!("{out_dir}/jobs/{job_id}");
    let output_path = req
        .output
        .clone()
        .unwrap_or_else(|| format!("{job_dir}/output.mp4"));
    let mut resolved = ResolvedRequest {
        job_id,
        mode: req.mode,
        prompt: req.prompt,
        negative_prompt: req.negative_prompt,
        input_image: req.input_image,
        profile,
        model: selected_model,
        width: req.width.unwrap_or(defaults.width),
        height: req.height.unwrap_or(defaults.height),
        frames: req.frames.unwrap_or(defaults.frames),
        fps: req.fps.unwrap_or(defaults.fps),
        steps: req.steps.unwrap_or(defaults.steps),
        seed: req.seed.unwrap_or_else(|| {
            let raw = Uuid::new_v4();
            u64::from_le_bytes(raw.as_bytes()[..8].try_into().unwrap())
        }),
        guidance_scale: req.guidance_scale.unwrap_or(defaults.guidance_scale),
        output_path,
        job_dir,
        downgrades: {
            downgrades.shrink_to_fit();
            downgrades
        },
        mock: req.mock,
    };
    if req.allow_auto_downgrade {
        apply_profile_caps(&mut resolved, profile);
    }
    Ok(resolved)
}
