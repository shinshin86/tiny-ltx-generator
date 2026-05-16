use ltx_core::reset_plan::restart_plan;
use ltx_core::workflow_contract::{validate_ltx23_template, ContractError};

#[test]
fn restart_plan_keeps_the_actual_goal_visible() {
    let plan = restart_plan();

    assert!(plan
        .primary_goal
        .contains("official ComfyUI LTX-2.3 lightweight workflow"));
    assert!(plan.primary_goal.contains("Rust CLI"));
    assert!(plan
        .non_goals
        .contains(&"No handcrafted replacement for the LTX sampler graph in Phase 1."));
    assert_eq!(plan.phases[0].name, "contract-first");
}

#[test]
fn ltx23_template_fixture_satisfies_required_node_contract() {
    let raw = include_str!("../../../fixtures/ltx23_comfy_template_minimal.json");

    let report = validate_ltx23_template(raw).expect("template fixture must satisfy contract");

    assert_eq!(report.subgraph_name, "Text to Video (LTX-2.3)");
    assert!(report.node_count >= 20);
    assert!(report
        .required_node_types
        .contains(&"LTXAVTextEncoderLoader".to_string()));
    assert!(report
        .required_node_types
        .contains(&"LTXVLatentUpsampler".to_string()));
}

#[test]
fn missing_core_template_nodes_are_rejected_before_generation() {
    let raw = r#"{"definitions":{"subgraphs":[{"name":"bad","nodes":[{"type":"CheckpointLoaderSimple"}]}]}}"#;

    let err = validate_ltx23_template(raw).expect_err("incomplete template must be rejected");

    assert!(matches!(err, ContractError::MissingRequiredNodes(_)));
    assert!(err.to_string().contains("LTXAVTextEncoderLoader"));
}

#[test]
fn handcrafted_python_graphs_are_rejected_as_a_design_regression() {
    let raw = r#"def _build_prompt(payload):
    prompt = {"1": {"class_type": "CheckpointLoaderSimple"}}
    return prompt
"#;

    let err = validate_ltx23_template(raw).expect_err("handcrafted graph must be rejected");

    assert_eq!(err, ContractError::HandcraftedGraph);
}
