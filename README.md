# valentinus
vector db built on top of lmdb

# dependencies

* lmdb
* libtorch 2.1.1
* rust-bert/tch-rs

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
