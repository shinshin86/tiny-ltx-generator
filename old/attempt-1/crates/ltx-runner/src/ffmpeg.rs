use crate::error::AppError;
use anyhow::Result;
use std::process::Command;

pub fn require_ffmpeg() -> Result<()> {
    if Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
    {
        return Ok(());
    }
    Err(AppError::Ffmpeg(
        "ffmpeg is missing; in Colab run: apt-get update && apt-get install -y ffmpeg".to_string(),
    )
    .into())
}

pub fn write_mock_mp4(path: &str, width: u32, height: u32, fps: u32) -> Result<()> {
    require_ffmpeg()?;
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            &format!("color=c=black:s={}x{}:r={}:d=1", width, height, fps),
            "-vf",
            "drawtext=text='tiny-ltx-generator mock':fontcolor=white:fontsize=24:x=20:y=20",
            "-pix_fmt",
            "yuv420p",
            path,
        ])
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(AppError::Ffmpeg("mock mp4 generation failed".to_string()).into())
    }
}
