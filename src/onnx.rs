#![deny(missing_docs)]

use ndarray::*;
use ort::{CUDAExecutionProvider, GraphOptimizationLevel, Session};
use tokenizers::Tokenizer;

use log::*;

/// ONNX Embeddings generator
pub fn generate_embeddings(
    model_path: &String,
    data: &Vec<String>,
) -> Result<Array2<f32>, ort::Error> {
    info!("creating model from {}", model_path);
    // Create the ONNX Runtime environment, enabling CPU/GPU execution providers for all sessions created in this process.
    ort::init()
        .with_name("valentinus")
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()
        .unwrap();
    // Load our model
    let session = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_intra_threads(1)?
        .commit_from_file(format!("{}/model.onnx", model_path))?;
    let tokenizer = Tokenizer::from_file(format!("{}/tokenizer.json", model_path))?;

    // Encode our input strings. `encode_batch` will pad each input to be the same length.
    let encodings = tokenizer.encode_batch(data.clone(), false)?;

    // Get the padded length of each encoding.
    let padded_token_length = encodings[0].len();

    // Get our token IDs & mask as a flattened array.
    let ids: Vec<i64> = encodings
        .iter()
        .flat_map(|e| e.get_ids().iter().map(|i| *i as i64))
        .collect();
    let mask: Vec<i64> = encodings
        .iter()
        .flat_map(|e| e.get_attention_mask().iter().map(|i| *i as i64))
        .collect();

    // Convert our flattened arrays into 2-dimensional tensors of shape [N, L].
    let a_ids = Array2::from_shape_vec([data.len(), padded_token_length], ids).unwrap();
    let a_mask = Array2::from_shape_vec([data.len(), padded_token_length], mask).unwrap();

    // Run the model.
    let outputs = session.run(ort::inputs![a_ids, a_mask]?)?;

    // Extract our embeddings tensor and convert it to a strongly-typed 2-dimensional array.
    let embeddings = outputs[1]
        .try_extract_tensor::<f32>()?
        .into_dimensionality::<Ix2>()
        .unwrap();

    Ok(embeddings.into_owned())
}
