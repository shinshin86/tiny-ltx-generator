use std::fs;

#[test]
fn json_and_jsonl_events_are_rejected_together() {
    let generate = fs::read_to_string("../../crates/ltx-runner/src/commands/generate.rs").unwrap();
    let batch = fs::read_to_string("../../crates/ltx-runner/src/commands/batch.rs").unwrap();

    for source in [generate, batch] {
        assert!(source.contains("args.json && args.jsonl_events"));
        assert!(source.contains("--json and --jsonl-events cannot be combined"));
    }
}
