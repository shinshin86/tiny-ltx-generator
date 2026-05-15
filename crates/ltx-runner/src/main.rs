mod cli;
mod commands;
mod config;
mod error;
mod ffmpeg;
mod hardware;
mod job;
mod json_output;
mod model_registry;
mod profile;
mod protocol;
mod storage;
mod worker_process;

use crate::cli::{Cli, Commands};
use crate::error::AppError;
use clap::Parser;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Check(args) => commands::check::run(args).await,
        Commands::Init(args) => commands::init::run(args).await,
        Commands::Generate(args) => commands::generate::run(args).await,
        Commands::Batch(args) => commands::batch::run(args).await,
        Commands::WorkerHealth(args) => commands::worker_health::run(args).await,
        Commands::ListProfiles(args) => commands::list_profiles::run(args).await,
        Commands::ListModels(args) => commands::list_models::run(args).await,
        Commands::Clean(args) => commands::clean::run(args).await,
        Commands::ShowConfig(args) => commands::show_config::run(args).await,
    };

    if let Err(err) = result {
        eprintln!("error: {err}");
        std::process::exit(match err.downcast_ref::<AppError>() {
            Some(app) => app.exit_code(),
            None => 1,
        });
    }
}
