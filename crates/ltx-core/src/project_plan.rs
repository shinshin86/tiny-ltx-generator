#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phase {
    pub name: &'static str,
    pub exit_criteria: &'static [&'static str],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectPlan {
    pub primary_goal: &'static str,
    pub non_goals: &'static [&'static str],
    pub phases: &'static [Phase],
}

pub fn project_plan() -> ProjectPlan {
    ProjectPlan {
        primary_goal: "Drive lightweight LTX video generation workflows from a Rust CLI on Colab with deterministic artifacts.",
        non_goals: &[
            "No handwritten replacement for the LTX sampler graph in Phase 1.",
            "No web UI, public HTTP API, Docker requirement, or project-specific remote-control protocol.",
            "No local real generation tests on the developer machine.",
        ],
        phases: &[
            Phase {
                name: "contract-first",
                exit_criteria: &[
                    "A sanitized LTX template or API prompt fixture is parsed by tests.",
                    "Required node types, model artifacts, and runtime knobs are validated before generation.",
                    "Ad hoc prompt graph construction is rejected by tests.",
                ],
            },
            Phase {
                name: "colab-template-runner",
                exit_criteria: &[
                    "Colab can clone the repo, download required models, and run the template headlessly.",
                    "The CLI exits 0 only when the MP4 exists, metadata is complete, and a visual health check passes.",
                    "Failure logs preserve the ComfyUI node id, node type, exception, and job id.",
                ],
            },
            Phase {
                name: "rust-memory-control",
                exit_criteria: &[
                    "Rust owns job planning, artifact validation, output contracts, and memory policy.",
                    "Python is limited to a thin ComfyUI execution adapter.",
                    "Peak VRAM, output size, and visual sanity checks are recorded for every run.",
                ],
            },
        ],
    }
}
