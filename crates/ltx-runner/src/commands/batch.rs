use anyhow::{Context, Result};
use chrono::Utc;
use ltx_core::{
    apply_cuda_oom_retry_downgrade, validate_resolved, GenerationRequest, ModelId, ResolvedRequest,
    WorkerResponse,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    fs,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use crate::{
    cli::BatchArgs,
    config::EngineConfig,
    error::AppError,
    ffmpeg,
    job::JobMetadata,
    json_output,
    model_registry::{missing_paths, model_key, preferred_models, ModelRegistry},
    protocol, storage,
    worker_process::WorkerProcess,
};

#[derive(Debug, Serialize, Deserialize)]
struct BatchSummary {
    total: usize,
    succeeded: usize,
    failed: usize,
    failures: Vec<serde_json::Value>,
}

pub async fn run(args: BatchArgs) -> Result<()> {
    if args.json && args.jsonl_events {
        return Err(AppError::Validation(
            "--json and --jsonl-events cannot be combined".to_string(),
        )
        .into());
    }
    let config = EngineConfig::load(args.config.as_deref())?;
    let registry_path = args
        .model_registry
        .clone()
        .unwrap_or_else(|| config.model_registry.clone());
    let registry = ModelRegistry::load(&registry_path)?;
    let out_dir = args.out_dir.clone().unwrap_or(config.output_dir.clone());
    storage::ensure_dirs(&[&out_dir])?;
    let file = File::open(&args.jobs).with_context(|| format!("open batch jobs {}", args.jobs))?;
    let reader = BufReader::new(file);
    let mut summary = BatchSummary {
        total: 0,
        succeeded: 0,
        failed: 0,
        failures: Vec::new(),
    };
    let mut worker = if args.mock {
        None
    } else {
        let mut worker = WorkerProcess::start(&config.python_worker).await?;
        let _ = worker.request("health", json!({})).await?;
        Some(worker)
    };
    let start = args.start_index.unwrap_or(0);
    for (idx, line) in reader.lines().enumerate().skip(start) {
        if let Some(max) = args.max_jobs {
            if summary.total >= max {
                break;
            }
        }
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        summary.total += 1;
        let job: ltx_core::BatchJob = match serde_json::from_str(&line) {
            Ok(job) => job,
            Err(err) => {
                summary.failed += 1;
                summary
                    .failures
                    .push(json!({"index": idx, "error": err.to_string()}));
                if args.stop_on_error || !args.continue_on_error {
                    break;
                }
                continue;
            }
        };
        if args.jsonl_events {
            json_output::print_jsonl(
                &json!({"type": "batch_job_start", "index": idx, "id": job.id}),
            )?;
        }
        let req = GenerationRequest {
            mode: job.mode.unwrap_or(ltx_core::GenerationMode::TextToVideo),
            prompt: job.prompt,
            negative_prompt: job.negative_prompt,
            input_image: job.input_image,
            profile: if matches!(args.profile, crate::cli::ProfileArg::Auto) {
                job.profile.unwrap_or_else(|| args.profile.into())
            } else {
                args.profile.into()
            },
            model: if matches!(args.model, crate::cli::ModelArg::Auto) {
                job.model.unwrap_or_else(|| args.model.into())
            } else {
                args.model.into()
            },
            width: job.width,
            height: job.height,
            frames: job.frames,
            fps: job.fps,
            steps: job.steps,
            seed: job.seed,
            guidance_scale: job.guidance_scale,
            output: None,
            out_dir: Some(out_dir.clone()),
            allow_auto_downgrade: true,
            copy_to_drive: args.copy_to_drive,
            mock: args.mock,
        };
        match run_batch_job(
            idx,
            req,
            &registry,
            &config,
            &config.python_worker,
            worker.as_mut(),
            args.jsonl_events,
            args.unload_between_jobs || !args.keep_model_warm,
        )
        .await
        {
            Ok(()) => summary.succeeded += 1,
            Err(err) => {
                summary.failed += 1;
                summary
                    .failures
                    .push(json!({"index": idx, "error": err.to_string()}));
                if args.stop_on_error || !args.continue_on_error {
                    break;
                }
            }
        }
    }
    if let Some(worker) = worker.as_mut() {
        let _ = worker.shutdown().await;
    }
    let summary_path = Path::new(&out_dir).join("batch_summary.json");
    storage::write_json(summary_path, &summary)?;
    if args.json {
        json_output::print_json(&summary)?;
    } else {
        eprintln!(
            "batch complete: {} succeeded, {} failed",
            summary.succeeded, summary.failed
        );
    }
    Ok(())
}

async fn run_batch_job(
    index: usize,
    req: GenerationRequest,
    registry: &ModelRegistry,
    config: &EngineConfig,
    worker_path: &str,
    worker: Option<&mut WorkerProcess>,
    jsonl_events: bool,
    unload_between_jobs: bool,
) -> Result<()> {
    let copy_to_drive = req.copy_to_drive;
    let resolved = crate::profile::resolve_request(req, worker_path, registry).await?;
    validate_resolved(&resolved).map_err(|err| AppError::Validation(err.to_string()))?;
    if let Some(image) = &resolved.input_image {
        if !Path::new(image).exists() {
            return Err(
                AppError::Validation(format!("input image does not exist: {image}")).into(),
            );
        }
    }
    ffmpeg::require_ffmpeg()?;
    let (selected_model, model_entry) = registry
        .select(resolved.model, resolved.profile)
        .ok_or_else(|| {
            AppError::MissingModel(format!("model {:?} is not configured", resolved.model))
        })?;
    super::generate::validate_model_support(&resolved, &model_entry)?;
    if !resolved.mock {
        let missing = missing_paths(&model_entry);
        if !missing.is_empty() {
            return Err(AppError::MissingModel(missing.join(", ")).into());
        }
    }
    storage::ensure_dirs(&[&resolved.job_dir])?;
    fs::write(format!("{}/prompt.txt", resolved.job_dir), &resolved.prompt)?;
    storage::write_json(
        format!("{}/resolved_request.json", resolved.job_dir),
        &resolved,
    )?;
    let events_path = Path::new(&resolved.job_dir).join("events.jsonl");
    if resolved.mock {
        protocol::record_event(
            &events_path,
            &WorkerResponse {
                id: resolved.job_id.clone(),
                response_type: "log".to_string(),
                payload: json!({"stage": "mock_generation", "mock": true, "batch_index": index}),
            },
            jsonl_events,
        )?;
        ffmpeg::write_mock_mp4(
            &resolved.output_path,
            resolved.width,
            resolved.height,
            resolved.fps,
        )?;
        write_metadata("success", None, resolved.clone(), Some(model_entry), true).await?;
        storage::write_json(
            format!("{}/worker_stats.json", resolved.job_dir),
            &json!({"mock": true}),
        )?;
        copy_to_drive_if_requested(copy_to_drive, config, &resolved)?;
        return Ok(());
    }
    let Some(worker) = worker else {
        return Err(AppError::WorkerCrash("batch worker is not available".to_string()).into());
    };
    let mut model_payload = serde_json::to_value(model_entry.clone())?;
    if let Some(obj) = model_payload.as_object_mut() {
        obj.insert("id".to_string(), json!(model_key(selected_model)));
    }
    match send_generation_on_worker(
        worker,
        index,
        model_payload,
        &resolved,
        &events_path,
        jsonl_events,
    )
    .await
    {
        Ok(result) => {
            storage::write_json(
                format!("{}/worker_stats.json", resolved.job_dir),
                &result.payload,
            )?;
            write_metadata("success", None, resolved.clone(), Some(model_entry), false).await?;
            copy_to_drive_if_requested(copy_to_drive, config, &resolved)?;
            if unload_between_jobs {
                let _ = worker.request("unload_model", json!({})).await;
            }
            Ok(())
        }
        Err(err) => {
            if is_cuda_oom(&err) {
                if let Some((retry_model, retry_entry)) = choose_cuda_oom_retry_model(registry) {
                    let _ = worker.request("unload_model", json!({})).await;
                    let mut retry_resolved = resolved.clone();
                    apply_cuda_oom_retry_downgrade(&mut retry_resolved, retry_model);
                    storage::write_json(
                        format!("{}/resolved_request.json", retry_resolved.job_dir),
                        &retry_resolved,
                    )?;
                    protocol::record_event(
                        &events_path,
                        &WorkerResponse {
                            id: retry_resolved.job_id.clone(),
                            response_type: "log".to_string(),
                            payload: json!({
                                "stage": "retry_after_cuda_oom",
                                "batch_index": index,
                                "profile": retry_resolved.profile,
                                "model": retry_resolved.model,
                                "downgrades": retry_resolved.downgrades.clone(),
                            }),
                        },
                        jsonl_events,
                    )?;
                    super::generate::validate_model_support(&retry_resolved, &retry_entry)?;
                    let missing = missing_paths(&retry_entry);
                    if !missing.is_empty() {
                        return Err(AppError::MissingModel(missing.join(", ")).into());
                    }
                    let mut retry_model_payload = serde_json::to_value(retry_entry.clone())?;
                    if let Some(obj) = retry_model_payload.as_object_mut() {
                        obj.insert("id".to_string(), json!(model_key(retry_model)));
                    }
                    match send_generation_on_worker(
                        worker,
                        index,
                        retry_model_payload,
                        &retry_resolved,
                        &events_path,
                        jsonl_events,
                    )
                    .await
                    {
                        Ok(result) => {
                            storage::write_json(
                                format!("{}/worker_stats.json", retry_resolved.job_dir),
                                &result.payload,
                            )?;
                            write_metadata(
                                "success",
                                None,
                                retry_resolved.clone(),
                                Some(retry_entry),
                                false,
                            )
                            .await?;
                            copy_to_drive_if_requested(copy_to_drive, config, &retry_resolved)?;
                            if unload_between_jobs {
                                let _ = worker.request("unload_model", json!({})).await;
                            }
                            return Ok(());
                        }
                        Err(retry_err) => {
                            write_metadata(
                                "error",
                                Some(json!({"message": retry_err.to_string(), "previous_error": err.to_string()})),
                                retry_resolved,
                                Some(retry_entry),
                                false,
                            )
                            .await?;
                            return Err(retry_err);
                        }
                    }
                }
            }
            write_metadata(
                "error",
                Some(json!({"message": err.to_string()})),
                resolved,
                Some(model_entry),
                false,
            )
            .await?;
            Err(err)
        }
    }
}

async fn send_generation_on_worker(
    worker: &mut WorkerProcess,
    index: usize,
    model_payload: serde_json::Value,
    resolved: &ResolvedRequest,
    events_path: &Path,
    jsonl_events: bool,
) -> Result<WorkerResponse> {
    worker
        .request_stream(
            "generate",
            json!({"request": resolved, "model": model_payload, "batch_index": index}),
            |event| protocol::record_event(events_path, event, jsonl_events),
        )
        .await
}

fn choose_cuda_oom_retry_model(
    registry: &ModelRegistry,
) -> Option<(ModelId, ltx_core::ModelEntry)> {
    preferred_models(ltx_core::ProfileId::ColabTiny)
        .into_iter()
        .filter_map(|model| registry.select(model, ltx_core::ProfileId::ColabTiny))
        .find(|(_, entry)| missing_paths(entry).is_empty())
}

fn is_cuda_oom(err: &anyhow::Error) -> bool {
    err.downcast_ref::<AppError>()
        .map(|app| matches!(app, AppError::CudaOom(_)))
        .unwrap_or(false)
}

fn copy_to_drive_if_requested(
    enabled: bool,
    config: &EngineConfig,
    resolved: &ResolvedRequest,
) -> Result<()> {
    if !enabled {
        return Ok(());
    }
    if !Path::new("/content/drive").exists() {
        return Err(AppError::Validation(
            "--copy-to-drive was set, but /content/drive is not mounted".to_string(),
        )
        .into());
    }
    let drive_outputs = format!("{}/outputs/jobs/{}", config.drive_root, resolved.job_id);
    storage::copy_dir_recursive(&resolved.job_dir, drive_outputs)
}

async fn write_metadata(
    status: &str,
    error: Option<serde_json::Value>,
    resolved: ResolvedRequest,
    model_entry: Option<ltx_core::ModelEntry>,
    mock: bool,
) -> Result<()> {
    let meta = JobMetadata {
        created_at: Utc::now(),
        resolved_request: resolved.clone(),
        model_entry,
        hardware: None,
        status: status.to_string(),
        error,
        mock,
    };
    storage::write_json(format!("{}/metadata.json", resolved.job_dir), &meta)
}
