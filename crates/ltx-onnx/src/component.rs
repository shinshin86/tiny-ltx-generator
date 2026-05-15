#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Component {
    VaeDecoder,
    VaeEncoder,
    TextEncoder,
    SpatialUpscaler,
    TemporalUpscaler,
    Transformer,
}

pub fn migration_order() -> Vec<Component> {
    vec![
        Component::VaeDecoder,
        Component::VaeEncoder,
        Component::TextEncoder,
        Component::SpatialUpscaler,
        Component::TemporalUpscaler,
        Component::Transformer,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_order_keeps_transformer_last() {
        let order = migration_order();
        assert_eq!(order.first(), Some(&Component::VaeDecoder));
        assert_eq!(order.last(), Some(&Component::Transformer));
    }
}
