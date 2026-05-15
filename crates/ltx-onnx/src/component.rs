#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Component {
    VaeDecoder,
    VaeEncoder,
    TextEncoder,
    SpatialUpscaler,
    TemporalUpscaler,
    Transformer,
}
