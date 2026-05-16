use ltx_core::job_artifacts::{validate_success_artifacts, ArtifactError};
use std::fs;
use tempfile::tempdir;

#[test]
fn success_requires_all_generation_artifacts_and_visual_pass() {
    let dir = tempdir().unwrap();
    write_success_files(dir.path(), r#"{"status":"pass","reason":"ok"}"#);

    let report = validate_success_artifacts(dir.path()).expect("complete job should pass");

    assert!(report.required_files.contains(&"output.mp4".to_string()));
    assert!(report
        .required_files
        .contains(&"worker_stats.json".to_string()));
    assert!(report
        .required_files
        .contains(&"visual_check.json".to_string()));
}

#[test]
fn success_is_rejected_when_worker_stats_are_missing() {
    let dir = tempdir().unwrap();
    write_success_files(dir.path(), r#"{"status":"pass","reason":"ok"}"#);
    fs::remove_file(dir.path().join("worker_stats.json")).unwrap();

    let err = validate_success_artifacts(dir.path()).expect_err("missing stats must fail");

    assert_eq!(err, ArtifactError::Missing("worker_stats.json".to_string()));
}

#[test]
fn success_is_rejected_when_visual_check_fails() {
    let dir = tempdir().unwrap();
    write_success_files(
        dir.path(),
        r#"{"status":"fail","reason":"flat_or_noise_like_output"}"#,
    );

    let err = validate_success_artifacts(dir.path()).expect_err("bad visual check must fail");

    assert_eq!(
        err,
        ArtifactError::VisualCheckFailed("flat_or_noise_like_output".to_string())
    );
}

fn write_success_files(dir: &std::path::Path, visual_check: &str) {
    fs::write(dir.join("output.mp4"), "not empty").unwrap();
    fs::write(dir.join("metadata.json"), "{}").unwrap();
    fs::write(dir.join("resolved_request.json"), "{}").unwrap();
    fs::write(dir.join("events.jsonl"), "{}\n").unwrap();
    fs::write(dir.join("worker_stats.json"), "{}").unwrap();
    fs::write(dir.join("visual_check.json"), visual_check).unwrap();
}
