use ndarray::array;
use ndarray_linalg::{normalize, NormalizeAxis};
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder,
    SentenceEmbeddingsModelType
};

use distance::L2Dist;
use linfa_nn::*;
use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use valentinus::embeddings::compute_nearest;

fn main() {

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
    println!("nearest embedding {:?}", data_sentences[i_nearest]);
}
