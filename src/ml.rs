
//! Machine learning algorithms

use distance::L2Dist;
use linfa_nn::*;
use log::info;
use ndarray::*;

/// Compute the nearest embedding
pub fn compute_nearest(data: Vec<Vec<f32>>, qv: Vec<f32>) -> usize {
    info!("computing nearest embedding");
    // convert nested embeddings vector to Array
    let dimensions = qv.len();
    let embed_len = data.len();
    let mut data_array: ndarray::ArrayBase<OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>
        = ndarray::Array::zeros((embed_len, dimensions));
    for index in 0..dimensions {
        for index1 in 0..embed_len {
            data_array[[index1,index]] = data[index1][index]
        }
    }
    // Query data
    let a_qv = ndarray::Array::from_vec(qv);
    // Kdtree using Euclidean distance
    let nn = CommonNearestNeighbour::KdTree.from_batch(&data_array, L2Dist).unwrap();
    // Compute the nearest point to the query vector
    let nearest = nn.k_nearest(a_qv.view(), 1).unwrap();
    let mut v_nearest: Vec<f32> = Vec::new();
    for v in nearest[0].0.to_vec() {
        v_nearest.push(v);
    }
    let location = data.iter().rposition(|x| x == &v_nearest);
    location.unwrap_or(0)
}

#[cfg(test)]
mod tests {

    use rust_bert::pipelines::sentence_embeddings::{SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType};

    use super::*;

    #[test]
    fn compute_nearest_test() {
        let qv = "I like pizza";
        let model = SentenceEmbeddingsBuilder::remote(
            SentenceEmbeddingsModelType::AllMiniLmL12V2
        ).create_model().expect("model");
    
        let qv_sentence = [qv];
        let data_sentences = [
            "The canine barked loudly.",
            "The dog made a noisy bark.",
            "He ate a lot of pizza.",
            "He devoured a large quantity of pizza pie.",
            "We went on a picnic.",
            "Water is made of two hyroden atoms and one oxygen atom.",
            "A triangle has 3 sides",
        ];
    
        let data_output = model.encode(&data_sentences).expect("embeddings");
        let qv_output = &model.encode(&qv_sentence).expect("embeddings")[0];
        let i_nearest = compute_nearest(data_output, qv_output.to_vec());

        assert_eq!(data_sentences[i_nearest], data_sentences[2]);
    }
}
