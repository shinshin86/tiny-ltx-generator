use std::{fmt, str::FromStr};

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

impl fmt::Display for Provider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Provider::Cpu => "cpu",
            Provider::Cuda => "cuda",
            Provider::TensorRt => "tensorrt",
        })
    }
}

impl FromStr for Provider {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "cpu" => Ok(Provider::Cpu),
            "cuda" => Ok(Provider::Cuda),
            "tensorrt" | "tensor_rt" | "tensor-rt" => Ok(Provider::TensorRt),
            other => Err(format!("unsupported ONNX provider: {other}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensorrt_falls_back_to_cuda_then_cpu() {
        assert_eq!(
            fallback_order(Provider::TensorRt),
            vec![Provider::TensorRt, Provider::Cuda, Provider::Cpu]
        );
    }

    #[test]
    fn provider_round_trips_display_names() {
        for provider in [Provider::Cpu, Provider::Cuda, Provider::TensorRt] {
            assert_eq!(provider.to_string().parse::<Provider>().unwrap(), provider);
        }
    }

    #[test]
    fn provider_accepts_tensor_rt_aliases() {
        assert_eq!("tensor_rt".parse::<Provider>().unwrap(), Provider::TensorRt);
        assert_eq!("tensor-rt".parse::<Provider>().unwrap(), Provider::TensorRt);
    }
}
