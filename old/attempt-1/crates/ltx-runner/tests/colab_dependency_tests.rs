use std::fs;

#[test]
fn python_worker_pins_transformers_to_ltx_compatible_major_version() {
    let pyproject = fs::read_to_string("../../py-worker/pyproject.toml").unwrap();

    assert!(
        pyproject.contains("\"transformers>=4.52,<5\""),
        "ltx-core 1.0 expects the Transformers 4.x SigLIP object layout"
    );
}
