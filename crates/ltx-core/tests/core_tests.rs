use ltx_core::{
    apply_cuda_oom_retry_downgrade, apply_profile_caps, recommend_profile, validate_resolved,
    BatchJob, GenerationMode, ModelId, ProfileId, ResolvedRequest, WorkerRequest, WorkerResponse,
};

fn base_request() -> ResolvedRequest {
    ResolvedRequest {
        job_id: "job".to_string(),
        mode: GenerationMode::TextToVideo,
        prompt: "hello".to_string(),
        negative_prompt: None,
        input_image: None,
        profile: ProfileId::ColabTiny,
        model: ModelId::Ltx2_3DistilledFp8,
        width: 512,
        height: 512,
        frames: 33,
        fps: 8,
        steps: 8,
        seed: 1,
        guidance_scale: 1.0,
        output_path: "/tmp/output.mp4".to_string(),
        job_dir: "/tmp/job".to_string(),
        downgrades: vec![],
        mock: true,
    }
}

#[test]
fn profile_resolver_uses_vram_thresholds() {
    assert_eq!(recommend_profile(Some(0)), ProfileId::NoGpu);
    assert_eq!(recommend_profile(Some(8_000)), ProfileId::ColabTiny);
    assert_eq!(recommend_profile(Some(12_000)), ProfileId::ColabEco);
    assert_eq!(recommend_profile(Some(24_000)), ProfileId::ColabBalanced);
    assert_eq!(recommend_profile(Some(40_000)), ProfileId::ColabQuality);
}

#[test]
fn validation_accepts_safe_request() {
    validate_resolved(&base_request()).unwrap();
}

#[test]
fn validation_accepts_ltx_landscape_smoke_request() {
    let mut req = base_request();
    req.profile = ProfileId::ColabBalanced;
    req.width = 768;
    req.height = 512;
    req.frames = 49;
    req.fps = 24;

    validate_resolved(&req).unwrap();
}

#[test]
fn validation_rejects_bad_resolution() {
    let mut req = base_request();
    req.width = 513;
    assert!(validate_resolved(&req).is_err());
}

#[test]
fn validation_rejects_bad_frames() {
    let mut req = base_request();
    req.frames = 40;
    assert!(validate_resolved(&req).is_err());
}

#[test]
fn validation_rejects_bad_image_extension() {
    let mut req = base_request();
    req.mode = GenerationMode::ImageToVideo;
    req.input_image = Some("/tmp/input.gif".to_string());
    assert!(validate_resolved(&req).is_err());
}

#[test]
fn downgrade_caps_record_each_changed_field() {
    let mut req = base_request();
    req.width = 1280;
    req.height = 720;
    req.frames = 97;
    req.fps = 24;
    req.steps = 30;
    apply_profile_caps(&mut req, ProfileId::ColabTiny);
    assert_eq!(req.width, 512);
    assert_eq!(req.height, 512);
    assert_eq!(req.frames, 33);
    assert_eq!(req.fps, 8);
    assert_eq!(req.steps, 8);
    assert_eq!(req.downgrades.len(), 5);
}

#[test]
fn cuda_oom_retry_downgrade_records_profile_model_and_caps() {
    let mut req = base_request();
    req.profile = ProfileId::ColabQuality;
    req.model = ModelId::Ltx2_3Full;
    req.width = 1280;
    req.height = 720;
    req.frames = 97;
    req.fps = 24;
    req.steps = 30;
    apply_cuda_oom_retry_downgrade(&mut req, ModelId::Ltx2_3Distilled);
    assert_eq!(req.profile, ProfileId::ColabTiny);
    assert_eq!(req.model, ModelId::Ltx2_3Distilled);
    assert_eq!(req.width, 512);
    assert_eq!(req.height, 512);
    assert_eq!(req.frames, 33);
    assert_eq!(req.fps, 8);
    assert_eq!(req.steps, 8);
    assert!(req
        .downgrades
        .iter()
        .all(|record| record.reason == "retry after CUDA OOM"));
}

#[test]
fn batch_job_schema_accepts_prompt_example_values() {
    let raw = r#"{"id":"dog-tokyo-001","mode":"text-to-video","prompt":"hello","negative_prompt":"blur","seed":12345,"width":768,"height":512,"frames":49,"fps":24,"profile":"colab_balanced","model":"ltxv_13b_distilled_fp8"}"#;
    let job: BatchJob = serde_json::from_str(raw).unwrap();
    assert_eq!(job.mode, Some(GenerationMode::TextToVideo));
    assert_eq!(job.model, Some(ModelId::Ltxv13bDistilledFp8));
}

#[test]
fn worker_protocol_jsonl_round_trip_shape() {
    let request = WorkerRequest {
        id: "abc".to_string(),
        cmd: "health".to_string(),
        payload: serde_json::json!({}),
    };
    let raw = serde_json::to_string(&request).unwrap();
    let parsed: WorkerRequest = serde_json::from_str(&raw).unwrap();
    assert_eq!(parsed.cmd, "health");

    let response: WorkerResponse =
        serde_json::from_str(r#"{"id":"abc","type":"progress","payload":{"stage":"x"}}"#).unwrap();
    assert_eq!(response.response_type, "progress");
}
