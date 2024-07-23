#![deny(missing_docs)]

//! Machine learning algorithms

use log::info;
use ndarray::*;

use crate::embeddings::CosineThreshold;

/// Compute cosine similarity for two vectors.
/// 
/// Let `qv` be the query vector and `cv` be the collection of embeddings
/// 
/// Reference: https://github.com/pykeio/ort/commit/c5538f26d0414e69cb7bbb92451f012a9e2bfa37
pub fn compute_cosine_similarity(qv: Array2<f32>, cv: Array2<f32>, docs: Vec<String>, ct: CosineThreshold) -> Vec<String> {
    info!("calculating cosine similarity");
    let mut results: Vec<String> = Vec::new();
    let query = qv.index_axis(Axis(0), 0);
	for (cv, sentence) in cv.axis_iter(Axis(0)).zip(docs.iter()).skip(1) {
		// Calculate cosine similarity against the 'query' sentence.
		let dot_product: f32 = query.iter().zip(cv.iter()).map(|(a, b)| a * b).sum();
        if ct == CosineThreshold::Related && dot_product > 0.0 {
            results.push(String::from(sentence));
        }
        if ct == CosineThreshold::NotRelated && dot_product < 0.0 {
            results.push(String::from(sentence));
        }
	}
    results
}

#[cfg(test)]
mod tests {
    
    use crate::onnx::generate_embeddings;

    use super::*;

    #[test]
    fn cosine_similarity_test() {
        let qv = ["I like pizza"];
        let model_path = String::from("all-Mini-LM-L6-v2_onnx");
        let data_sentences = [
            "The canine barked loudly.",
            "The dog made a noisy bark.",
            "He ate a lot of pizza.",
            "He devoured a large quantity of pizza pie.",
            "We went on a picnic.",
            "Water is made of two hyroden atoms and one oxygen atom.",
            "A triangle has 3 sides",
        ];
        let query: Vec<String> = qv.iter().map(|s| String::from(*s)).collect();
        let data: Vec<String> = data_sentences.iter().map(|s| String::from(*s)).collect();
        let query_embeddings = generate_embeddings(&model_path, &query).unwrap_or_default();
        let data_embeddings = generate_embeddings(&model_path, &data).unwrap_or_default();
        let result = compute_cosine_similarity(query_embeddings, data_embeddings, data, CosineThreshold::Related);
        assert!(result.len() > 0);
    }
}
