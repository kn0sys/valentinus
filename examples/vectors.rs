use ndarray::array;
use ndarray_linalg::*;

fn main() {
    let zero: f64 = 0 as f64;
    let one: f64 = 1 as f64;
    let two: f64 = 2 as f64;
    let v1 = array!([one, zero]);
    let v2 = array!([zero, one]);
    let v3 = array!([two.sqrt(), two.sqrt()]);
    println!("Dimension of 1: {:?}", v1.ndim());
    println!("Dimension of 2: {:?}", v2.ndim());
    let v1_norm = normalize(v1.clone(), NormalizeAxis::Row);
    let v3_norm = normalize(v3, NormalizeAxis::Row);
    println!("Magnitude of v1: {:?}", v1_norm.1);
    println!("Magnitude of v3: {:?}", v3_norm.1);
    let dot_product = (v1 * v2).sum();
    println!("Dot product of v1 * v2: {}", dot_product);
}
