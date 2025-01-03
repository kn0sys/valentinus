[![.github/workflows/rust.yml](https://github.com/kn0sys/valentinus/actions/workflows/rust.yml/badge.svg)](https://github.com/kn0sys/valentinus/actions/workflows/rust.yml) [![test](https://github.com/kn0sys/valentinus/actions/workflows/test.yml/badge.svg)](https://github.com/kn0sys/valentinus/actions/workflows/test.yml) [![Crates.io Version](https://img.shields.io/crates/v/valentinus)](https://crates.io/crates/valentinus)
 ![Crates.io Total Downloads](https://img.shields.io/crates/d/valentinus) [![docs.rs](https://img.shields.io/docsrs/valentinus)](https://docs.rs/valentinus) [![GitHub commit activity](https://img.shields.io/github/commit-activity/m/kn0sys/valentinus)](https://github.com/kn0sys/valentinus/commits/main/) [![Matrix](https://img.shields.io/matrix/valentinus%3Amatrix.org)](https://app.element.io/#/room/#valentinus:matrix.org)


![alt text](logo.png) 

# valentinus 

next generation vector db built with lmdb bindings

### dependencies

* bincode/serde  - serialize/deserialize
* lmdb-rs        - database bindings
* ndarray        - numpy equivalent
* ort/onnx       - embeddings

### getting started

```bash
git clone https://github.com/kn0sys/valentinus && cd valentinus
```

### optional environment variables

| var| usage | default |
|----|-------| --------|
|`LMDB_USER` | working directory of the user for database | $USER|
|`LMDB_MAP_SIZE` | Sets max environment size, i.e. size in memory/disk of all data  | 20% of available memory |
|`ONNX_PARALLEL_THREADS` | parallel execution mode for this session | 1 |
|`VALENTINUS_CUSTOM_DIM` | embeddings dimensions for custom models | all-mini-lm-6 -> 384 |
|`VALENTINUS_LMDB_ENV`| environment for the database (i.e. test, prod) | test |


# tests

* Note: all tests currently require the `all-MiniLM-L6-v2_onnx` directory
* Get the model.onnx and tokenizer.json from huggingface or [build them](https://huggingface.co/docs/optimum/en/exporters/onnx/usage_guides/export_a_model)

```bash
mkdir all-MiniLM-L6-v2_onnx
cd all-MiniLM-L6-v2_onnx && wget https://huggingface.co/nigel-christian/all-MiniLM-L6-v2_onnx/resolve/main/config.json
wget https://huggingface.co/nigel-christian/all-MiniLM-L6-v2_onnx/resolve/main/model.onnx
wget https://huggingface.co/nigel-christian/all-MiniLM-L6-v2_onnx/resolve/main/special_tokens_map.json
wget https://huggingface.co/nigel-christian/all-MiniLM-L6-v2_onnx/resolve/main/tokenizer_config.json
wget https://huggingface.co/nigel-christian/all-MiniLM-L6-v2_onnx/resolve/main/tokenizer.json
wget https://huggingface.co/nigel-christian/all-MiniLM-L6-v2_onnx/resolve/main/vocab.txt
```

`RUST_TEST_THREADS=1 cargo test`

### examples

see [examples](https://github.com/kn0sys/valentinus/tree/main/examples)

### donations

[Monero](https://getmonero.org) donations accepted via open alias

```bash
donate.hiahatf.org
``` 

or subaddress 

```bash
859uSbW547HNhH4TzaThbdXCeoPas97y9EpqYCScRYDbciAGBuow2R6YW4DzARixD5J4v9zEPv2r3AWLrDNGkS6Z9X6yAoD
```

### reference

[inspired by this chromadb python tutorial](https://realpython.com/chromadb-vector-database/#what-is-a-vector-database)
