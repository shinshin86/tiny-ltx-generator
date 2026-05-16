#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phase {
    pub name: &'static str,
    pub exit_criteria: &'static [&'static str],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartPlan {
    pub primary_goal: &'static str,
    pub non_goals: &'static [&'static str],
    pub phases: &'static [Phase],
}

pub fn restart_plan() -> RestartPlan {
    RestartPlan {
        primary_goal: "Drive the official ComfyUI LTX-2.3 lightweight workflow from a Rust CLI on Colab with less memory overhead and deterministic artifacts.",
        non_goals: &[
            "No handcrafted replacement for the LTX sampler graph in Phase 1.",
            "No web UI, HTTP server, Docker, ComfyUI workflow editor, or MCP integration.",
            "No local real generation tests on the developer machine.",
        ],
        phases: &[
            Phase {
                name: "contract-first",
                exit_criteria: &[
                    "A sanitized ComfyUI LTX-2.3 template fixture is parsed by tests.",
                    "Required node types, model artifacts, and runtime knobs are validated before generation.",
                    "Manual Python dict graph construction is rejected by tests.",
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
