use crate::{defaults, DowngradeRecord, ProfileId, ResolvedRequest};
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
