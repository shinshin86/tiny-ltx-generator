use anyhow::Result;
use ltx_core::{recommend_profile, HardwareInfo, TorchInfo};
use regex::Regex;
use std::{env, path::Path, process::Command};

use crate::worker_process::WorkerProcess;

pub async fn inspect(include_worker: bool, worker_path: &str) -> Result<HardwareInfo> {
    let cwd = env::current_dir()?.display().to_string();
    let python_version = version("python3", &["--version"]);
    let uv_version = version("uv", &["--version"]);
    let cargo_version = version("cargo", &["--version"]);
    let rustc_version = version("rustc", &["--version"]);
    let ffmpeg_version = version("ffmpeg", &["-version"]);
    let (gpu_name, total_vram_mb, free_vram_mb, nvidia_smi_available) = nvidia_smi();
    let torch = if include_worker {
        match WorkerProcess::start(worker_path).await {
            Ok(mut worker) => {
                let response = worker.request("health", serde_json::json!({})).await.ok();
                let _ = worker.shutdown().await;
                response.and_then(|r| {
                    serde_json::from_value::<TorchInfo>(r.payload.get("torch").cloned()?).ok()
                })
            }
            Err(err) => Some(TorchInfo {
                import_ok: false,
                version: None,
                cuda_available: false,
                cuda_version: None,
                device_count: 0,
                gpu_name: None,
                allocated_vram_mb: None,
                reserved_vram_mb: None,
                error: Some(err.to_string()),
            }),
        }
    } else {
        None
    };
    let recommended_profile = if let Some(torch) = &torch {
        if torch.cuda_available {
            recommend_profile(free_vram_mb.or(total_vram_mb))
        } else {
            ltx_core::ProfileId::NoGpu
        }
    } else {
        recommend_profile(free_vram_mb.or(total_vram_mb))
    };
    Ok(HardwareInfo {
        os: format!("{} {}", env::consts::OS, env::consts::ARCH),
        cwd,
        content_disk_available_gb: disk_available_gb("/content"),
        drive_mounted: Path::new("/content/drive").exists(),
        python_version,
        uv_version,
        cargo_version,
        rustc_version,
        ffmpeg_version,
        nvidia_smi_available,
        gpu_name,
        total_vram_mb,
        free_vram_mb,
        torch,
        recommended_profile,
    })
}

fn version(bin: &str, args: &[&str]) -> Option<String> {
    Command::new(bin)
        .args(args)
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| {
            let text = if out.stdout.is_empty() {
                out.stderr
            } else {
                out.stdout
            };
            String::from_utf8_lossy(&text)
                .lines()
                .next()
                .unwrap_or("")
                .to_string()
        })
}

fn nvidia_smi() -> (Option<String>, Option<u64>, Option<u64>, bool) {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let Ok(out) = output else {
        return (None, None, None, false);
    };
    if !out.status.success() {
        return (None, None, None, true);
    }
    let line = String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .to_string();
    let parts: Vec<_> = line.split(',').map(|s| s.trim()).collect();
    if parts.len() < 3 {
        return (None, None, None, true);
    }
    (
        Some(parts[0].to_string()),
        parts[1].parse().ok(),
        parts[2].parse().ok(),
        true,
    )
}

fn disk_available_gb(path: &str) -> Option<f64> {
    let output = Command::new("df").args(["-k", path]).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let re = Regex::new(r"\s(\d+)\s+\d+%\s").ok()?;
    let caps = re.captures_iter(&text).last()?;
    let kb: f64 = caps.get(1)?.as_str().parse().ok()?;
    Some(kb / 1024.0 / 1024.0)
}
