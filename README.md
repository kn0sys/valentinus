[![.github/workflows/rust.yml](https://github.com/kn0sys/valentinus/actions/workflows/rust.yml/badge.svg)](https://github.com/kn0sys/valentinus/actions/workflows/rust.yml)

# valentinus
vector db built on top of lmdb

# dependencies

* lmdb-rs        - database bindings
* linfa          - machine learning
* ndarray        - numpy equivalent
* rust-bert      - embeddings encoding
* libtorch 2.1.1 - required by rust-bert
* see [rust-bert/tch-rs](https://github.com/guillaume-be/rust-bert)

# getting started

```bash
git clone --recursive https://github.com/kn0sys/valentinus
```

```bash
git submodule --init --update
```

```bash
export LIBTORCH="/home/user/libtorch"
export LD_LIBRARY_PATH="$LIBTORCH/lib"
export LIBTORCH_BYPASS_VERSION_CHECK=1
```

# examples

see [examples](./examples/embeddings.md)

# reference

[inspired by this chromadb python tutorial](https://realpython.com/chromadb-vector-database/#what-is-a-vector-database)