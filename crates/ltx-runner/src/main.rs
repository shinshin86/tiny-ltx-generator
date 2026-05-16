use clap::{Parser, Subcommand};
use ltx_core::job_artifacts::REQUIRED_SUCCESS_FILES;
use ltx_core::reset_plan::restart_plan;
use ltx_core::template_patcher::{apply_template_patch, TemplatePatchRequest};
use ltx_core::workflow_contract::validate_ltx23_template;
use serde_json::json;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "ltx-runner")]
#[command(about = "Colab-first LTX-2.3 template runner")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    Plan,
    ValidateTemplate {
        #[arg(long)]
        workflow: PathBuf,
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        json: bool,
    },
    PatchTemplate(PatchTemplateArgs),
    PrepareRun(PrepareRunArgs),
}

#[derive(Debug, Parser)]
struct PatchTemplateArgs {
    #[arg(long)]
    workflow: PathBuf,
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value = "")]
    negative_prompt: String,
    #[arg(long)]
    width: u64,
    #[arg(long)]
    height: u64,
    #[arg(long)]
    duration_seconds: u64,
    #[arg(long)]
    fps: u64,
    #[arg(long)]
    seed: u64,
    #[arg(long)]
    checkpoint: String,
    #[arg(long)]
    text_encoder: String,
    #[arg(long)]
    distilled_lora: String,
    #[arg(long, default_value_t = 0.5)]
    lora_strength: f64,
    #[arg(long)]
    spatial_upscaler: String,
}

#[derive(Debug, Parser)]
struct PrepareRunArgs {
    #[command(flatten)]
    patch: PatchTemplateArgs,
    #[arg(long)]
    out_dir: PathBuf,
    #[arg(long)]
    job_id: String,
    #[arg(long, default_value = "/content/ComfyUI")]
    comfy_root: PathBuf,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command.unwrap_or(Command::Plan) {
        Command::Plan => {
            let plan = restart_plan();
            println!("{}", plan.primary_goal);
        }
        Command::ValidateTemplate {
            workflow,
            manifest,
            json: as_json,
        } => {
            let workflow_raw = fs::read_to_string(workflow)?;
            let manifest_raw = fs::read_to_string(manifest)?;
            let report = validate_ltx23_template(&workflow_raw)?;
            serde_json::from_str::<serde_json::Value>(&manifest_raw)?;
            if as_json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "valid": true,
                        "subgraph_name": report.subgraph_name,
                        "node_count": report.node_count,
                        "required_node_types": report.required_node_types,
                    }))?
                );
            } else {
                println!("template valid: {}", report.subgraph_name);
            }
        }
        Command::PatchTemplate(args) => {
            let workflow_raw = fs::read_to_string(args.workflow)?;
            let manifest_raw = fs::read_to_string(args.manifest)?;
            let patched = apply_template_patch(
                &workflow_raw,
                &manifest_raw,
                &TemplatePatchRequest {
                    prompt: args.prompt,
                    negative_prompt: args.negative_prompt,
                    width: args.width,
                    height: args.height,
                    duration_seconds: args.duration_seconds,
                    fps: args.fps,
                    seed: args.seed,
                    checkpoint: args.checkpoint,
                    text_encoder: args.text_encoder,
                    distilled_lora: args.distilled_lora,
                    lora_strength: args.lora_strength,
                    spatial_upscaler: args.spatial_upscaler,
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&patched.workflow)?);
        }
        Command::PrepareRun(args) => {
            let workflow_raw = fs::read_to_string(&args.patch.workflow)?;
            let manifest_raw = fs::read_to_string(&args.patch.manifest)?;
            let request = TemplatePatchRequest {
                prompt: args.patch.prompt,
                negative_prompt: args.patch.negative_prompt,
                width: args.patch.width,
                height: args.patch.height,
                duration_seconds: args.patch.duration_seconds,
                fps: args.patch.fps,
                seed: args.patch.seed,
                checkpoint: args.patch.checkpoint,
                text_encoder: args.patch.text_encoder,
                distilled_lora: args.patch.distilled_lora,
                lora_strength: args.patch.lora_strength,
                spatial_upscaler: args.patch.spatial_upscaler,
            };
            let patched = apply_template_patch(&workflow_raw, &manifest_raw, &request)?;
            let job_dir = args.out_dir.join("jobs").join(&args.job_id);
            fs::create_dir_all(&job_dir)?;

            let patched_workflow_path = job_dir.join("patched_workflow.json");
            let output_path = job_dir.join("output.mp4");
            fs::write(
                &patched_workflow_path,
                serde_json::to_string_pretty(&patched.workflow)?,
            )?;
            fs::write(job_dir.join("prompt.txt"), format!("{}\n", request.prompt))?;
            fs::write(
                job_dir.join("resolved_request.json"),
                serde_json::to_string_pretty(&json!({
                    "job_id": args.job_id,
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "width": request.width,
                    "height": request.height,
                    "duration_seconds": request.duration_seconds,
                    "fps": request.fps,
                    "seed": request.seed,
                    "checkpoint": request.checkpoint,
                    "text_encoder": request.text_encoder,
                    "distilled_lora": request.distilled_lora,
                    "lora_strength": request.lora_strength,
                    "spatial_upscaler": request.spatial_upscaler,
                }))?,
            )?;
            fs::write(
                job_dir.join("metadata.json"),
                serde_json::to_string_pretty(&json!({
                    "job_id": args.job_id,
                    "status": "prepared",
                    "workflow_source": args.patch.workflow,
                    "manifest_source": args.patch.manifest,
                    "changed_controls": patched.changed_controls,
                }))?,
            )?;
            fs::write(
                job_dir.join("adapter_request.json"),
                serde_json::to_string_pretty(&json!({
                    "schema_version": 1,
                    "backend": "comfyui_headless",
                    "comfy_root": args.comfy_root,
                    "job_dir": job_dir,
                    "workflow_path": patched_workflow_path,
                    "output_path": output_path,
                    "required_success_files": REQUIRED_SUCCESS_FILES,
                }))?,
            )?;
            fs::write(
                job_dir.join("events.jsonl"),
                format!(
                    "{}\n",
                    serde_json::to_string(&json!({
                        "type": "prepared",
                        "job_id": args.job_id,
                    }))?
                ),
            )?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "status": "prepared",
                    "job_id": args.job_id,
                    "job_dir": job_dir,
                    "adapter_request": job_dir.join("adapter_request.json"),
                }))?
            );
        }
    }
    Ok(())
}
