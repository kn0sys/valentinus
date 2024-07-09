///! Library for handling embeddings

// TODO: Write embeddings to lmdb

// TODO: Read embeddings to lmdb

// TODO: Indexer for faster queries

use core::f32;

use distance::L2Dist;
use linfa_nn::*;
use ndarray::{Array, Array2, ArrayBase, Axis, Dim, OwnedRepr, ShapeBuilder, Slice};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

/// Compute the nearest embedding
pub fn compute_nearest(data: Vec<Vec<f32>>, qv: Vec<f32>) -> usize {
    // convert nested embeddings vector to Array
    let dimensions = qv.len();
    let embed_len = data.len();
    let mut data_array: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> = Array::zeros((embed_len, dimensions));
    for index in 0..dimensions {
        for index1 in 0..embed_len {
            data_array[[index1,index]] = data[index1][index]
        }
    }
    // Query data
    let a_qv = Array::from_vec(qv);
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
