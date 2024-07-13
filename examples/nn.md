```rust
use distance::L2Dist;
use linfa_nn::*;
use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

fn main() {
    let mut rng = rand::thread_rng();
    let num_samples = 10000;
    let dimensions = 5;
    let dist = Uniform::new(0., 1.);
    // Multi-dimensional array of data
    let data = Array::random_using((num_samples, dimensions), dist, &mut rng);
    // Query data
    let query_vector = Array::random_using(dimensions, dist, &mut rng);
    println!("query vector: {} ", query_vector);
    // Kdtree using Euclidean distance
    let nn = CommonNearestNeighbour::KdTree.from_batch(&data, L2Dist).unwrap();
    // Compute the nearest points to the query vector
    let nearest = nn.k_nearest(query_vector.view(), 1).unwrap();
    println!("Nearest: {:?}", nearest[0].0)
}
````
