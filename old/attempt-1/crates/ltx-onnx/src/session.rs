use thiserror::Error;

#[derive(Debug, Error)]
pub enum OnnxSessionError {
    #[error("ONNX support is disabled; build with --features onnx")]
    Disabled,
}

#[cfg(not(feature = "onnx"))]
pub fn onnx_enabled() -> bool {
    false
}

#[cfg(feature = "onnx")]
pub fn onnx_enabled() -> bool {
    true
}
