use anyhow::Result;

use crate::{cli::JsonArgs, config::EngineConfig, json_output, worker_process::WorkerProcess};

pub async fn run(args: JsonArgs) -> Result<()> {
    let config = EngineConfig::load(None)?;
    let mut worker = WorkerProcess::start(&config.python_worker).await?;
    let response = worker.request("health", serde_json::json!({})).await?;
    let _ = worker.shutdown().await;
    if args.json {
        json_output::print_json(&response.payload)?;
    } else {
        eprintln!("{}", serde_json::to_string_pretty(&response.payload)?);
    }
    Ok(())
}
