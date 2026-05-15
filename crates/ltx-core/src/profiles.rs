use crate::{DowngradeRecord, HardwareInfo, ProfileId};
use serde_json::json;

#[derive(Debug, Clone, Copy)]
pub struct ProfileDefaults {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
    pub steps: u32,
    pub guidance_scale: f32,
}

pub fn defaults(profile: ProfileId) -> ProfileDefaults {
    match profile {
        ProfileId::NoGpu | ProfileId::Auto => ProfileDefaults {
            width: 512,
            height: 512,
            frames: 33,
            fps: 8,
            steps: 8,
            guidance_scale: 1.0,
        },
        ProfileId::ColabTiny => ProfileDefaults {
            width: 512,
            height: 512,
            frames: 33,
            fps: 8,
            steps: 8,
            guidance_scale: 1.0,
        },
        ProfileId::ColabEco => ProfileDefaults {
            width: 512,
            height: 512,
            frames: 49,
            fps: 12,
            steps: 8,
            guidance_scale: 1.0,
        },
        ProfileId::ColabBalanced => ProfileDefaults {
            width: 768,
            height: 512,
            frames: 65,
            fps: 12,
            steps: 20,
            guidance_scale: 3.0,
        },
        ProfileId::ColabQuality => ProfileDefaults {
            width: 1280,
            height: 720,
            frames: 97,
            fps: 16,
            steps: 30,
            guidance_scale: 3.0,
        },
    }
}

pub fn resolve_profile(
    requested: ProfileId,
    hardware: &HardwareInfo,
    allow_downgrade: bool,
) -> Result<(ProfileId, Vec<DowngradeRecord>), String> {
    let recommended = recommend_profile(hardware.free_vram_mb.or(hardware.total_vram_mb));
    if requested == ProfileId::Auto {
        return Ok((recommended, Vec::new()));
    }
    if requested == ProfileId::NoGpu {
        return Ok((ProfileId::NoGpu, Vec::new()));
    }
    if recommended == ProfileId::NoGpu {
        return Err("CUDA GPU is not available".to_string());
    }
    if profile_rank(recommended) >= profile_rank(requested) {
        return Ok((requested, Vec::new()));
    }
    if !allow_downgrade {
        return Err(format!(
            "requested profile {:?} exceeds detected hardware; recommended {:?}",
            requested, recommended
        ));
    }
    Ok((
        recommended,
        vec![DowngradeRecord {
            field: "profile".to_string(),
            requested: json!(requested),
            actual: json!(recommended),
            reason: "free VRAM is below requested profile threshold".to_string(),
        }],
    ))
}

pub fn recommend_profile(vram_mb: Option<u64>) -> ProfileId {
    match vram_mb.unwrap_or(0) {
        0..=1 => ProfileId::NoGpu,
        2..=9_999 => ProfileId::ColabTiny,
        10_000..=15_999 => ProfileId::ColabEco,
        16_000..=31_999 => ProfileId::ColabBalanced,
        _ => ProfileId::ColabQuality,
    }
}

fn profile_rank(profile: ProfileId) -> u8 {
    match profile {
        ProfileId::NoGpu => 0,
        ProfileId::ColabTiny => 1,
        ProfileId::ColabEco => 2,
        ProfileId::ColabBalanced => 3,
        ProfileId::ColabQuality => 4,
        ProfileId::Auto => 5,
    }
}
