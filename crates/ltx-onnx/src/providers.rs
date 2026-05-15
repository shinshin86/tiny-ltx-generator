#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Cpu,
    Cuda,
    TensorRt,
}

pub fn fallback_order(requested: Provider) -> Vec<Provider> {
    match requested {
        Provider::TensorRt => vec![Provider::TensorRt, Provider::Cuda, Provider::Cpu],
        Provider::Cuda => vec![Provider::Cuda, Provider::Cpu],
        Provider::Cpu => vec![Provider::Cpu],
    }
}
