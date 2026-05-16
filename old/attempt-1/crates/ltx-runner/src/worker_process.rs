use anyhow::{Context, Result};
use ltx_core::{WorkerRequest, WorkerResponse};
use std::{env, process::Stdio};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
};
use uuid::Uuid;

use crate::error::AppError;

pub struct WorkerProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl WorkerProcess {
    pub async fn start(worker_path: &str) -> Result<Self> {
        let python = env::var("LTX_PYTHON").unwrap_or_else(|_| {
            if std::path::Path::new("py-worker/.venv/bin/python").exists() {
                "py-worker/.venv/bin/python".to_string()
            } else {
                "python3".to_string()
            }
        });
        let mut child = Command::new(python)
            .arg(worker_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .env("PYTHONPATH", "py-worker")
            .spawn()
            .with_context(|| format!("start Python worker {worker_path}"))?;
        let stdin = child.stdin.take().context("worker stdin unavailable")?;
        let stdout = child.stdout.take().context("worker stdout unavailable")?;
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })
    }

    pub async fn request(
        &mut self,
        cmd: &str,
        payload: serde_json::Value,
    ) -> Result<WorkerResponse> {
        let id = Uuid::new_v4().to_string();
        let req = WorkerRequest {
            id: id.clone(),
            cmd: cmd.to_string(),
            payload,
        };
        self.stdin
            .write_all(format!("{}\n", serde_json::to_string(&req)?).as_bytes())
            .await?;
        self.stdin.flush().await?;
        loop {
            let mut line = String::new();
            let bytes = self.stdout.read_line(&mut line).await?;
            if bytes == 0 {
                return Err(AppError::WorkerCrash("worker stdout closed".to_string()).into());
            }
            let response: WorkerResponse = serde_json::from_str(line.trim())
                .with_context(|| format!("parse worker response: {line}"))?;
            if response.id == id
                && response.response_type != "progress"
                && response.response_type != "log"
            {
                if response.response_type == "error" {
                    return Err(map_worker_error(response));
                }
                return Ok(response);
            }
        }
    }

    pub async fn request_stream<F>(
        &mut self,
        cmd: &str,
        payload: serde_json::Value,
        mut on_event: F,
    ) -> Result<WorkerResponse>
    where
        F: FnMut(&WorkerResponse) -> Result<()>,
    {
        let id = Uuid::new_v4().to_string();
        let req = WorkerRequest {
            id: id.clone(),
            cmd: cmd.to_string(),
            payload,
        };
        self.stdin
            .write_all(format!("{}\n", serde_json::to_string(&req)?).as_bytes())
            .await?;
        self.stdin.flush().await?;
        loop {
            let mut line = String::new();
            let bytes = self.stdout.read_line(&mut line).await?;
            if bytes == 0 {
                return Err(AppError::WorkerCrash("worker stdout closed".to_string()).into());
            }
            let response: WorkerResponse = serde_json::from_str(line.trim())
                .with_context(|| format!("parse worker response: {line}"))?;
            if response.id == id {
                on_event(&response)?;
                match response.response_type.as_str() {
                    "result" | "stats" => return Ok(response),
                    "error" => return Err(map_worker_error(response)),
                    _ => {}
                }
            }
        }
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        let _ = self.request("shutdown", serde_json::json!({})).await;
        let _ = self.child.wait().await;
        Ok(())
    }
}

fn map_worker_error(response: WorkerResponse) -> anyhow::Error {
    let code = response
        .payload
        .get("code")
        .and_then(|v| v.as_str())
        .unwrap_or("worker_error");
    let message = response
        .payload
        .get("message")
        .and_then(|v| v.as_str())
        .unwrap_or("worker error")
        .to_string();
    match code {
        "cuda_oom" => AppError::CudaOom(message).into(),
        "missing_model_files" => AppError::MissingModel(message).into(),
        "unsupported_pipeline" | "unsupported_option" => AppError::Unsupported(message).into(),
        _ => AppError::WorkerCrash(message).into(),
    }
}
