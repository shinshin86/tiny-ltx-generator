use anyhow::Result;
use serde_json::json;

use crate::{cli::JsonArgs, json_output};

pub async fn run(args: JsonArgs) -> Result<()> {
    let profiles = json!([
        {"id": "no_gpu", "generation": false},
        {"id": "colab_tiny", "width": 512, "height": 512, "frames": 33, "fps": 8},
        {"id": "colab_eco", "width": 512, "height": 512, "frames": 49, "fps": 12},
        {"id": "colab_balanced", "width": 768, "height": 512, "frames": 65, "fps": 12},
        {"id": "colab_quality", "width": 1280, "height": 720, "frames": 97, "fps": 16}
    ]);
    if args.json {
        json_output::print_json(&profiles)?;
    } else {
        eprintln!("no_gpu");
        eprintln!("colab_tiny");
        eprintln!("colab_eco");
        eprintln!("colab_balanced");
        eprintln!("colab_quality");
    }
    Ok(())
}
