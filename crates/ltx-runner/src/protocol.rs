use anyhow::Result;
use ltx_core::WorkerResponse;
use std::path::Path;

use crate::{json_output, storage};

pub fn record_event(
    events_path: &Path,
    response: &WorkerResponse,
    jsonl_events: bool,
) -> Result<()> {
    storage::append_jsonl(events_path, response)?;
    if jsonl_events {
        json_output::print_jsonl(response)?;
    }
    Ok(())
}
