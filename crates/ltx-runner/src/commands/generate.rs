use anyhow::{Context, Result};
use chrono::Utc;
use ltx_core::{
    apply_cuda_oom_retry_downgrade, validate_resolved, GenerationRequest, ModelId, ResolvedRequest,
    WorkerResponse,
};
use serde_json::json;
use std::{fs, path::Path};

use crate::{
    cli::GenerateArgs,
    config::EngineConfig,
    error::AppError,
    ffmpeg, hardware,
    job::JobMetadata,
    json_output,
    model_registry::{missing_paths, model_key, preferred_models, ModelRegistry},
    protocol, storage,
    worker_process::WorkerProcess,
};

pub async fn run(args: GenerateArgs) -> Result<()> {
    let config = EngineConfig::load(args.config.as_deref())?;
    let registry_path = args
        .model_registry
        .clone()
        .unwrap_or_else(|| config.model_registry.clone());
    let registry = ModelRegistry::load(&registry_path)?;
    let req = GenerationRequest {
        mode: args.mode.into(),
        prompt: args.prompt,
        negative_prompt: args.negative_prompt,
        input_image: args.input_image,
        profile: args.profile.into(),
        model: args.model.into(),
        width: args.width,
        height: args.height,
        frames: args.frames,
        fps: args.fps,
        steps: args.steps,
        seed: args.seed,
        guidance_scale: args.guidance_scale,
        output: args.output,
        out_dir: args.out_dir.or(Some(config.output_dir.clone())),
        allow_auto_downgrade: args.allow_auto_downgrade && !args.no_auto_downgrade,
        copy_to_drive: args.copy_to_drive,
        mock: args.mock,
    };
    let resolved = crate::profile::resolve_request(req, &config.python_worker, &registry).await?;
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
    validate_model_support(&resolved, &model_entry)?;
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
        ffmpeg::write_mock_mp4(
            &resolved.output_path,
            resolved.width,
            resolved.height,
            resolved.fps,
        )?;
        let meta = metadata(
            "success",
            None,
            resolved.clone(),
            Some(model_entry),
            args.mock,
        )
        .await?;
        storage::write_json(format!("{}/metadata.json", resolved.job_dir), &meta)?;
        storage::write_json(
            format!("{}/worker_stats.json", resolved.job_dir),
            &json!({"mock": true}),
        )?;
        copy_to_drive_if_requested(args.copy_to_drive, &config, &resolved)?;
        if args.json {
            json_output::print_json(
                &json!({"status": "success", "mock": true, "output": resolved.output_path, "job_dir": resolved.job_dir, "model": selected_model}),
            )?;
        } else {
            eprintln!("mock output written: {}", resolved.output_path);
        }
        return Ok(());
    }
    let result = match run_worker_generation(
        &config,
        selected_model,
        &model_entry,
        &resolved,
        &events_path,
        args.jsonl_events,
    )
    .await
    {
        Ok(result) => result,
        Err(err) => {
            if is_cuda_oom(&err) && args.allow_auto_downgrade && !args.no_auto_downgrade {
                if let Some((retry_model, retry_entry)) = choose_cuda_oom_retry_model(&registry) {
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
                                "profile": retry_resolved.profile,
                                "model": retry_resolved.model,
                                "downgrades": retry_resolved.downgrades.clone(),
                            }),
                        },
                        args.jsonl_events,
                    )?;
                    match run_worker_generation(
                        &config,
                        retry_model,
                        &retry_entry,
                        &retry_resolved,
                        &events_path,
                        args.jsonl_events,
                    )
                    .await
                    {
                        Ok(result) => {
                            storage::write_json(
                                format!("{}/worker_stats.json", retry_resolved.job_dir),
                                &result.payload,
                            )?;
                            let meta = metadata(
                                "success",
                                None,
                                retry_resolved.clone(),
                                Some(retry_entry),
                                false,
                            )
                            .await?;
                            storage::write_json(
                                format!("{}/metadata.json", retry_resolved.job_dir),
                                &meta,
                            )?;
                            copy_to_drive_if_requested(
                                args.copy_to_drive,
                                &config,
                                &retry_resolved,
                            )?;
                            if args.json {
                                json_output::print_json(
                                    &json!({"status": "success", "retried_after_cuda_oom": true, "output": retry_resolved.output_path, "job_dir": retry_resolved.job_dir}),
                                )?;
                            } else {
                                eprintln!(
                                    "output written after CUDA OOM retry: {}",
                                    retry_resolved.output_path
                                );
                            }
                            return Ok(());
                        }
                        Err(retry_err) => {
                            let meta = metadata(
                                "error",
                                Some(json!({"message": retry_err.to_string(), "previous_error": err.to_string()})),
                                retry_resolved.clone(),
                                Some(retry_entry),
                                false,
                            )
                            .await?;
                            storage::write_json(
                                format!("{}/metadata.json", retry_resolved.job_dir),
                                &meta,
                            )?;
                            return Err(retry_err);
                        }
                    }
                }
            }
            let meta = metadata(
                "error",
                Some(json!({"message": err.to_string()})),
                resolved.clone(),
                Some(model_entry.clone()),
                false,
            )
            .await?;
            storage::write_json(format!("{}/metadata.json", resolved.job_dir), &meta)?;
            return Err(err);
        }
    };
    storage::write_json(
        format!("{}/worker_stats.json", resolved.job_dir),
        &result.payload,
    )?;
    let meta = metadata("success", None, resolved.clone(), Some(model_entry), false).await?;
    storage::write_json(format!("{}/metadata.json", resolved.job_dir), &meta)?;
    copy_to_drive_if_requested(args.copy_to_drive, &config, &resolved)?;
    if args.json {
        json_output::print_json(
            &json!({"status": "success", "output": resolved.output_path, "job_dir": resolved.job_dir}),
        )?;
    } else {
        eprintln!("output written: {}", resolved.output_path);
    }
    Ok(())
}

async fn run_worker_generation(
    config: &EngineConfig,
    selected_model: ModelId,
    model_entry: &ltx_core::ModelEntry,
    resolved: &ResolvedRequest,
    events_path: &Path,
    jsonl_events: bool,
) -> Result<WorkerResponse> {
    validate_model_support(resolved, model_entry)?;
    let missing = missing_paths(model_entry);
    if !missing.is_empty() {
        return Err(AppError::MissingModel(missing.join(", ")).into());
    }
    let mut worker = WorkerProcess::start(&config.python_worker).await?;
    let _health = worker.request("health", json!({})).await?;
    let mut model_payload = serde_json::to_value(model_entry.clone())?;
    if let Some(obj) = model_payload.as_object_mut() {
        obj.insert("id".to_string(), json!(model_key(selected_model)));
    }
    let result = worker
        .request_stream(
            "generate",
            json!({
                "request": resolved,
                "model": model_payload,
            }),
            |event| protocol::record_event(events_path, event, jsonl_events),
        )
        .await;
    let _ = worker.shutdown().await;
    result
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
    resolved: &ltx_core::ResolvedRequest,
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

pub(crate) fn validate_model_support(
    resolved: &ltx_core::ResolvedRequest,
    model_entry: &ltx_core::ModelEntry,
) -> Result<()> {
    match resolved.mode {
        ltx_core::GenerationMode::TextToVideo if !model_entry.supports_t2v => {
            Err(AppError::Unsupported(format!(
                "{} does not support text-to-video",
                model_entry.display_name
            ))
            .into())
        }
        ltx_core::GenerationMode::ImageToVideo if !model_entry.supports_i2v => {
            Err(AppError::Unsupported(format!(
                "{} does not support image-to-video",
                model_entry.display_name
            ))
            .into())
        }
        _ => validate_quantization_support(model_entry),
    }
}

fn validate_quantization_support(model_entry: &ltx_core::ModelEntry) -> Result<()> {
    match model_entry.quantization.as_deref().unwrap_or("none") {
        "none" => Ok(()),
        "fp8-cast" if model_entry.supports_fp8_cast => Ok(()),
        "fp8-scaled-mm" if model_entry.supports_fp8_scaled_mm => Ok(()),
        mode => Err(AppError::Unsupported(format!(
            "{} does not support quantization mode {mode}",
            model_entry.display_name
        ))
        .into()),
    }
}

async fn metadata(
    status: &str,
    error: Option<serde_json::Value>,
    resolved: ltx_core::ResolvedRequest,
    model_entry: Option<ltx_core::ModelEntry>,
    mock: bool,
) -> Result<JobMetadata> {
    let hardware = hardware::inspect(false, "py-worker/worker.py").await.ok();
    Ok(JobMetadata {
        created_at: Utc::now(),
        resolved_request: resolved,
        model_entry,
        hardware,
        status: status.to_string(),
        error,
        mock,
    })
}
