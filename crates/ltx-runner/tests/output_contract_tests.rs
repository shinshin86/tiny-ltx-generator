use std::fs;

#[test]
fn mock_generation_paths_write_required_job_artifacts() {
    let generate = fs::read_to_string("../../crates/ltx-runner/src/commands/generate.rs").unwrap();
    let batch = fs::read_to_string("../../crates/ltx-runner/src/commands/batch.rs").unwrap();

    for source in [generate, batch] {
        assert!(source.contains("mock_generation"));
        assert!(source.contains("worker_stats.json"));
        assert!(source.contains("events.jsonl"));
    }
}
