#![deny(missing_docs)]

//! ort is a Rust binding for ONNX Runtime. For information on how to get started with ort, see https://ort.pyke.io/introduction.

use ndarray::*;
use ort::{
    execution_providers::CUDAExecutionProvider,
    session::builder::GraphOptimizationLevel,
    session::Session,
    value::TensorRef
};
use tokenizers::Tokenizer;

use log::*;

/// Used for controlling the amount of data being encoded
pub const BATCH_SIZE: usize = 100;

/// Default dimensions for the all-mini-lm-l6 model
const DEFUALT_DIMENSIONS: usize = 384;

/// Custom dimensions when setting custom models
const VALENTINUS_CUSTOM_DIM: &str = "VALENTINUS_CUSTOM_DIM";

/// Environment variable for parallel execution threads count
const ONNX_PARALLEL_THREADS: &str = "ONNX_PARALLEL_THREADS";

#[derive(Debug)]
pub enum OnnxError {
    OrtError(ort::Error),
    ShapeError(ShapeError),
}

/// ONNX Embeddings generator
fn generate_embeddings(model_path: &String, data: &[String]) -> Result<Array2<f32>, OnnxError> {
    let threads: usize = match std::env::var(ONNX_PARALLEL_THREADS) {
        Err(_) => 1,
        Ok(t) => t.parse::<usize>().unwrap_or(1),
    };
    info!(
        "generating encodings from {} with {} threads",
        model_path, threads
    );
    // Create the ONNX Runtime environment, enabling CPU/GPU execution providers for all sessions created in this process.
    ort::init()
        .with_name("valentinus")
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()
        .map_err(OnnxError::OrtError)?;
    // Load our model
    let mut session = Session::builder()
        .map_err(OnnxError::OrtError)?
        .with_optimization_level(GraphOptimizationLevel::Level1)
        .map_err(OnnxError::OrtError)?
        .with_parallel_execution(threads > 1)
        .map_err(OnnxError::OrtError)?
        .with_intra_threads(threads)
        .map_err(OnnxError::OrtError)?
        .commit_from_file(format!("{}/model.onnx", model_path))
        .map_err(OnnxError::OrtError)?;
    let tokenizer = Tokenizer::from_file(format!("{}/tokenizer.json", model_path))
        .map_err(|e| OnnxError::OrtError(ort::Error::new(e.to_string())))?;
    // Encode our input strings. `encode_batch` will pad each input to be the same length.
    let encodings = tokenizer
        .encode_batch(data.to_vec(), false)
        .map_err(|e| OnnxError::OrtError(ort::Error::new(e.to_string())))?;
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
    let a_ids = TensorRef::from_array_view(([data.len(), padded_token_length], &*ids))
        .map_err(|e| OnnxError::OrtError(ort::Error::new(e.to_string())))?;
    let a_mask = TensorRef::from_array_view(([data.len(), padded_token_length], &*mask))
        .map_err(|e| OnnxError::OrtError(ort::Error::new(e.to_string())))?;
    //let a_ids = Array2::from_shape_vec([data.len(), padded_token_length], ids)
    //    .map_err(OnnxError::ShapeError)?;
    //let a_mask = Array2::from_shape_vec([data.len(), padded_token_length], mask)
    //    .map_err(OnnxError::ShapeError)?;
    // Run the model.
    let outputs = session
        .run(ort::inputs![a_ids, a_mask])
        .map_err(OnnxError::OrtError)?;
    // Extract our embeddings tensor and convert it to a strongly-typed 2-dimensional array.
    let embeddings = outputs[1]
        .try_extract_tensor::<f32>()
        .map_err(OnnxError::OrtError)?
        .into_dimensionality::<Ix2>()
        .map_err(|e| OnnxError::OrtError(ort::Error::new(e.to_string())))?;
    Ok(embeddings.into_owned())
}

/// Batch embeddings with a batch size of 100 elements.
pub fn batch_embeddings(model_path: &String, data: &[String]) -> Result<Array2<f32>, OnnxError> {
    info!("batching length {} from {}", data.len(), model_path);
    let dimensions: usize = match std::env::var(VALENTINUS_CUSTOM_DIM) {
        Err(_) => DEFUALT_DIMENSIONS,
        Ok(t) => t.parse::<usize>().unwrap_or(DEFUALT_DIMENSIONS),
    };
    let mut data_array: ndarray::ArrayBase<OwnedRepr<f32>, ndarray::Dim<[usize; 2]>> =
        ndarray::Array::zeros((data.len(), dimensions));
    let mut begin: usize = 0;
    let mut multiplier: usize = 1;
    let length = data.len();
    while length - begin > BATCH_SIZE {
        info!("{} encodings remaining", length - begin);
        let end = (BATCH_SIZE * multiplier) - 1;
        let embeddings = generate_embeddings(model_path, &data[begin..end])?;
        for index1 in begin..end {
            for index2 in 0..dimensions {
                data_array[[index1, index2]] = embeddings[[index1 - begin, index2]];
            }
        }
        begin += BATCH_SIZE;
        multiplier += 1;
    }
    info!("{} encodings remaining", length - begin);
    let embeddings = generate_embeddings(model_path, &data[begin..length])?;
    for index1 in 0..length - begin {
        for index2 in 0..dimensions {
            data_array[[(index1 + begin), index2]] = embeddings[[index1, index2]];
        }
    }
    Ok(data_array)
}
