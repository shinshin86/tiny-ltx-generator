use crate::{defaults, DowngradeRecord, ModelId, ProfileId, ResolvedRequest};
use serde_json::json;

pub fn apply_profile_caps(req: &mut ResolvedRequest, profile: ProfileId) {
    let caps = defaults(profile);
    cap_u32(&mut req.width, caps.width, "width", &mut req.downgrades);
    cap_u32(&mut req.height, caps.height, "height", &mut req.downgrades);
    cap_u32(&mut req.frames, caps.frames, "frames", &mut req.downgrades);
    cap_u32(&mut req.fps, caps.fps, "fps", &mut req.downgrades);
    cap_u32(&mut req.steps, caps.steps, "steps", &mut req.downgrades);
}

fn cap_u32(value: &mut u32, max: u32, field: &str, downgrades: &mut Vec<DowngradeRecord>) {
    if *value > max {
        let requested = *value;
        *value = max;
        downgrades.push(DowngradeRecord {
            field: field.to_string(),
            requested: json!(requested),
            actual: json!(*value),
            reason: "profile memory cap".to_string(),
        });
    }
}

pub fn apply_cuda_oom_retry_downgrade(req: &mut ResolvedRequest, fallback_model: ModelId) {
    set_profile(req, ProfileId::ColabTiny, "retry after CUDA OOM");
    set_model(req, fallback_model, "retry after CUDA OOM");
    cap_u32_reason(
        &mut req.width,
        512,
        "width",
        "retry after CUDA OOM",
        &mut req.downgrades,
    );
    cap_u32_reason(
        &mut req.height,
        512,
        "height",
        "retry after CUDA OOM",
        &mut req.downgrades,
    );
    cap_u32_reason(
        &mut req.frames,
        33,
        "frames",
        "retry after CUDA OOM",
        &mut req.downgrades,
    );
    cap_u32_reason(
        &mut req.fps,
        8,
        "fps",
        "retry after CUDA OOM",
        &mut req.downgrades,
    );
    cap_u32_reason(
        &mut req.steps,
        8,
        "steps",
        "retry after CUDA OOM",
        &mut req.downgrades,
    );
}

fn set_profile(req: &mut ResolvedRequest, actual: ProfileId, reason: &str) {
    if req.profile != actual {
        let requested = req.profile;
        req.profile = actual;
        req.downgrades.push(DowngradeRecord {
            field: "profile".to_string(),
            requested: json!(requested),
            actual: json!(actual),
            reason: reason.to_string(),
        });
    }
}

fn set_model(req: &mut ResolvedRequest, actual: ModelId, reason: &str) {
    if req.model != actual {
        let requested = req.model;
        req.model = actual;
        req.downgrades.push(DowngradeRecord {
            field: "model".to_string(),
            requested: json!(requested),
            actual: json!(actual),
            reason: reason.to_string(),
        });
    }
}

fn cap_u32_reason(
    value: &mut u32,
    max: u32,
    field: &str,
    reason: &str,
    downgrades: &mut Vec<DowngradeRecord>,
) {
    if *value > max {
        let requested = *value;
        *value = max;
        downgrades.push(DowngradeRecord {
            field: field.to_string(),
            requested: json!(requested),
            actual: json!(*value),
            reason: reason.to_string(),
        });
    }
}
