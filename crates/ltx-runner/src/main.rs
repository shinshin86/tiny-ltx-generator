use clap::{Parser, Subcommand};
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
    }
    Ok(())
}
