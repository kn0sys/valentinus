[![.github/workflows/rust.yml](https://github.com/kn0sys/valentinus/actions/workflows/rust.yml/badge.svg)](https://github.com/kn0sys/valentinus/actions/workflows/rust.yml) [![Crates.io Version](https://img.shields.io/crates/v/valentinus)](https://crates.io/crates/valentinus)
 [![Crates.io Downloads (latest version)](https://img.shields.io/crates/dv/valentinus)](https://crates.io/crates/valentinus) [![docs.rs](https://img.shields.io/docsrs/valentinus)](https://docs.rs/valentinus) [![GitHub commit activity](https://img.shields.io/github/commit-activity/m/kn0sys/valentinus)](https://github.com/kn0sys/valentinus/commits/main/) [![Matrix](https://img.shields.io/matrix/valentinus%3Amatrix.org)](https://app.element.io/#/room/#valentinus:matrix.org)


![alt text](logo.png) 

# valentinus 

next generation vector db built with lmdb bindings

### dependencies

* bincode/serde  - serialize/deserialize
* lmdb-rs        - database bindings
* linfa          - machine learning
* ndarray        - numpy equivalent
* ort/onnx       - embeddings

### getting started

```bash
git clone https://github.com/kn0sys/valentinus && cd valentinus
```

# tests

* Note: all tests currently require the `all-Mini-LM-L6-v2_onnx` directory
* Get the model.onnx and tokenizer.json from huggingface or [build them](https://huggingface.co/docs/optimum/en/exporters/onnx/usage_guides/export_a_model)

```bash
mkdir all-Mini-LM-L6-v2_onnx
cd all-Mini-LM-L6-v2_onnx && wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json
wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx
wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/special_tokens_map.json
wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json
wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json
wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt
```

### examples

see [examples](./examples/embeddings.md)

### gpu 

set the environment variable `VALENTINUS_GPU_MODE=1` to enable the 
`CUDAExecutionProvider`

### reference

[inspired by this chromadb python tutorial](https://realpython.com/chromadb-vector-database/#what-is-a-vector-database)
