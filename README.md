[![.github/workflows/rust.yml](https://github.com/kn0sys/valentinus/actions/workflows/rust.yml/badge.svg)](https://github.com/kn0sys/valentinus/actions/workflows/rust.yml) [![Crates.io Version](https://img.shields.io/crates/v/valentinus)](https://crates.io/crates/valentinus)
 [![Crates.io Downloads (latest version)](https://img.shields.io/crates/dv/valentinus)](https://crates.io/crates/valentinus) [![docs.rs](https://img.shields.io/docsrs/valentinus)](https://docs.rs/valentinus) [![GitHub commit activity](https://img.shields.io/github/commit-activity/m/kn0sys/valentinus)](https://github.com/kn0sys/valentinus/commits/main/) [![Matrix](https://img.shields.io/matrix/valentinus%3Amatrix.org)](https://app.element.io/#/room/#valentinus:matrix.org)


![alt text](logo.png) 

# valentinus 

proo-of-concept vector db built with lmdb bindings

### dependencies

* lmdb-rs        - database bindings
* linfa          - machine learning
* ndarray        - numpy equivalent
* rust-bert      - embeddings encoding
* libtorch 2.1.1 - required by rust-bert
* see [rust-bert/tch-rs](https://github.com/guillaume-be/rust-bert)

### getting started

```bash
git clone https://github.com/kn0sys/valentinus
```

```bash
export LIBTORCH="/home/user/libtorch"
export LD_LIBRARY_PATH="$LIBTORCH/lib"
export LIBTORCH_BYPASS_VERSION_CHECK=1
```

### examples

see [examples](./examples/embeddings.md)

### reference

[inspired by this chromadb python tutorial](https://realpython.com/chromadb-vector-database/#what-is-a-vector-database)
