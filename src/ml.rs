
//! Machine learning algorithms

use distance::L2Dist;
use linfa_nn::*;
use log::{error, info};
use ndarray::*;

/// Compute the nearest embedding
pub fn compute_nearest(data: Vec<Vec<f32>>, qv: Vec<f32>) -> usize {
    if data.is_empty() || qv.is_empty() {
        error!("can't compute empty vectors");
        return 0;
    }
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

pub fn normalize(x: ArrayView1<f32>) -> f32 {
    x.dot(&x).sqrt()
}

/// Compute cosine similarity for two vectors. Let `qv` be the unprocessed query vector
/// 
/// and `cv` be the pre-processed collection value stored in the database with its norms.
/// 
/// Reference: https://rust-lang-nursery.github.io/rust-cookbook/science/mathematics/linear_algebra.html
pub fn compute_cosine_similarity(qv: ArrayView1<f32>, cv: ArrayView1<f32>, cn: f32) -> f32 {
    let dot_product = qv.dot(&cv);
    let qv_norm = normalize(qv.view());
    dot_product / (qv_norm * cn)
}

#[cfg(test)]
mod tests {

    use crate::onnx::generate_embeddings;

    use super::*;

    #[test]
    fn compute_nearest_test() {
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
        let i_nearest = compute_nearest(data_embeddings.v_f32, query_embeddings.v_f32[0].to_vec());
        assert_eq!(data_sentences[i_nearest], data_sentences[2]);
    }

    #[test]
    fn cosine_similarity_test() {
        let model_path = String::from("all-Mini-LM-L6-v2_onnx");
        let threshold: f32 = 0.5;
        let dog = ["dog"].iter().map(|s| String::from(*s)).collect::<Vec<String>>();
        let cat = ["cat"].iter().map(|s| String::from(*s)).collect::<Vec<String>>();
        let car = ["fast cars"].iter().map(|s| String::from(*s)).collect::<Vec<String>>();
        let mut e_dog = generate_embeddings(&model_path, &dog).unwrap_or_default();
        let mut e_cat = generate_embeddings(&model_path, &cat).unwrap_or_default();
        let mut e_car = generate_embeddings(&model_path, &car).unwrap_or_default();
        let a_dog = Array::from(e_dog.v_f32.remove(0));
        let a_cat = Array1::from_vec(e_cat.v_f32.remove(0));
        let a_car = Array1::from_vec(e_car.v_f32.remove(0));
        let dog_cat = compute_cosine_similarity(a_dog.view(), a_cat.view(), e_cat.norm[0]);
        let dog_car = compute_cosine_similarity(a_dog.view(), a_car.view(), e_car.norm[0]);
        assert!(dog_cat > threshold);
        assert!(dog_car < threshold);
    }
}
