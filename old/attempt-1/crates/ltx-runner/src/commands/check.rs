use anyhow::Result;

use crate::{cli::CheckArgs, config::EngineConfig, hardware, json_output};

pub async fn run(args: CheckArgs) -> Result<()> {
    let config = EngineConfig::load(None)?;
    let info = hardware::inspect(true, &config.python_worker).await?;
    if args.json {
        json_output::print_json(&info)?;
    } else {
        eprintln!("OS: {}", info.os);
        eprintln!("cwd: {}", info.cwd);
        eprintln!("/content free GB: {:?}", info.content_disk_available_gb);
        eprintln!("Drive mounted: {}", info.drive_mounted);
        eprintln!("Python: {:?}", info.python_version);
        eprintln!("uv: {:?}", info.uv_version);
        eprintln!("cargo: {:?}", info.cargo_version);
        eprintln!("rustc: {:?}", info.rustc_version);
        eprintln!("ffmpeg: {:?}", info.ffmpeg_version);
        eprintln!("nvidia-smi: {}", info.nvidia_smi_available);
        eprintln!("GPU: {:?}", info.gpu_name);
        eprintln!(
            "VRAM total/free MB: {:?}/{:?}",
            info.total_vram_mb, info.free_vram_mb
        );
        eprintln!("torch: {:?}", info.torch);
        eprintln!("recommended profile: {:?}", info.recommended_profile);
    }
    Ok(())
}
