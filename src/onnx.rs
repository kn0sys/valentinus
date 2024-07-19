
use ndarray::*;
use ort::CPUExecutionProvider;
use ort::GraphOptimizationLevel;
use ort::Session;
use tokenizers::Tokenizer;

use log::*;
use crate::ml::*;

/// Use to store embedding in the database with their
/// 
/// mutations and their normalization.
#[derive(Debug, Default)]
pub struct GeneratedEmbeddings {
   pub v_f32: Vec<Vec<f32>>,
   pub norm: Vec<f32>,
}

/// ONNX Embeddings generator
pub fn generate_embeddings(model_path: &String, data: &Vec<String>) -> Result<GeneratedEmbeddings, ort::Error> {
    info!("creating model from {}", model_path);
    let mut embeddings_result: Vec<Vec<f32>> = Vec::new();
    let mut norm_result: Vec<f32> = Vec::new();
    // Create the ONNX Runtime environment, enabling CPU execution providers for all sessions created in this process.
	ort::init()
		.with_name("valentinus")
		.with_execution_providers([CPUExecutionProvider::default().build()])
		.commit().unwrap();
	// Load our model
	let session = Session::builder().unwrap()
		.with_optimization_level(GraphOptimizationLevel::Level1)?
		.with_intra_threads(1)?
		.commit_from_file(format!("{}/model.onnx", model_path))?;
    let tokenizer = Tokenizer::from_file(format!("{}/tokenizer.json", model_path))?;
    for v in data {
        let tokens = tokenizer.encode(String::from(v), false)?;
        let mask = tokens.get_attention_mask().iter().map(|i| *i as i64).collect::<Vec<i64>>();
        let ids = tokens.get_ids().iter().map(|i| *i as i64).collect::<Vec<i64>>();
        let a_ids = Array1::from_vec(ids);
        let a_mask = Array1::from_vec(mask);
        let input_ids = a_ids.view().insert_axis(Axis(0));
        let input_mask = a_mask.view().insert_axis(Axis(0));
        let outputs = session.run(ort::inputs![input_ids, input_mask]?)?;
        let tensor = outputs[1].try_extract_tensor::<f32>();
        if tensor.is_err() {
            error!("failed to extract tensor");
            return Ok(Default::default())
        }
        let u_tensor: Vec<f32> = tensor.unwrap().iter().copied().collect::<Vec<f32>>();
        let mut norm_vec: Vec<f32> = Vec::new();
        for v in &u_tensor {
            norm_vec.push(*v);
        }
        embeddings_result.push(u_tensor);
        let norm: f32 = normalize(Array::from_vec(norm_vec).view());
        norm_result.push(norm);
    }
    let result: GeneratedEmbeddings  = GeneratedEmbeddings { v_f32: embeddings_result, norm: norm_result };
    Ok(result)

}