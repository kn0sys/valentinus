#![deny(missing_docs)]

//! Logic for interfacing with LMDB.

extern crate lmdb_rs as lmdb;

use lmdb::{DbFlags, DbHandle, EnvBuilder, Environment};
use lmdb_rs::Database;
use log::{error, info};
use sysinfo::System;

/// Write collections to a separate database
pub const COLLECTIONS: &str = "collections";
/// Keys indexer constant for writing all collections keys
pub const VALENTINUS_KEYS: &str = "keys";
/// Views indexer constant for writing all collections view names
pub const VALENTINUS_VIEWS: &str = "views";
/// Key lookup
pub const VALENTINUS_KEY: &str = "key";
/// View lookup
pub const VALENTINUS_VIEW: &str = "view";
/// Ratio of map size to available memory is 20 percent
const MAP_SIZE_MEMORY_RATIO: f32 = 0.2;
/// Ratio of chunk size to available memory is 0.2 percent
const CHUNK_SIZE_MEMORY_RATIO: f32 = MAP_SIZE_MEMORY_RATIO * 0.01;

/// The database environment for handling primary database operations.
///
/// By default the database will be written to /home/user/.valentinus/{ENV}/lmdb
pub struct DatabaseEnvironment {
    pub env: Environment,
    pub handle: DbHandle,
}

impl DatabaseEnvironment {
    /// Opens environment in specified path. The map size defaults to 20 percent
    ///
    /// of available memory and can be set via the `LMDB_MAP_SIZE` environment variable.
    ///
    /// The path of the user can be set with `LMDB_USER`.
    pub fn open(env: &str) -> Self {
        let s = System::new_all();
        let default_map_size: u64 =
            (s.available_memory() as f32 * MAP_SIZE_MEMORY_RATIO).floor() as u64;
        let env_map_size: u64 = match std::env::var("LMDB_MAP_SIZE") {
            Err(_) => default_map_size,
            Ok(size) => size.parse::<u64>().unwrap_or(default_map_size),
        };
        info!("setting lmdb map size to: {}", env_map_size);
        let user: String = match std::env::var("LMDB_USER") {
            Err(_) => std::env::var("USER").unwrap_or(String::from("user")),
            Ok(user) => user,
        };
        info!("$LMDB_USER={}", user);
        info!("excecuting lmdb open");
        let file_path: String = format!("/home/{}/.{}/", user, "valentinus");
        let env: Environment = EnvBuilder::new()
            .map_size(env_map_size)
            .open(format!("{}/{}", file_path, env), 0o777)
            .unwrap_or_else(|_| panic!("could not open LMDB at {}", file_path));
        let default: Result<DbHandle, lmdb_rs::MdbError> = env.get_default_db(DbFlags::empty());
        if default.is_err() {
            panic!("could not set db handle")
        }
        let handle: DbHandle = default.unwrap();
        DatabaseEnvironment { env, handle }
    }
    /// Write a key/value pair to the database. It is not possible to
    ///
    /// write with empty keys.
    fn write(e: &Environment, h: &DbHandle, k: &Vec<u8>, v: &Vec<u8>) {
        info!("excecuting lmdb write");
        if k.is_empty() {
            error!("can't write empty key");
            return;
        }
        let new_txn: Result<lmdb_rs::Transaction, lmdb_rs::MdbError> = e.new_transaction();
        if new_txn.is_err() {
            error!("failed write txn on key: {:?}", k);
            return;
        }
        let txn = new_txn.unwrap();
        {
            let db: Database = txn.bind(h);
            let pair: Vec<(&Vec<u8>, &Vec<u8>)> = vec![(k, v)];
            for &(key, value) in pair.iter() {
                db.set(key, value)
                    .unwrap_or_else(|_| error!("failed to set key: {:?}", k));
            }
        }
        txn.commit().unwrap()
    }
    /// Read key from the database. If it doesn't exist then
    ///
    /// an empty vector will be returned. Treat all empty vectors
    ///
    /// from database operations as failures.
    pub fn read(e: &Environment, h: &DbHandle, k: &Vec<u8>) -> Vec<u8> {
        info!("excecuting lmdb read");
        // don't try and read empty keys
        if k.is_empty() {
            error!("can't read empty key");
            return Vec::new();
        }
        let get_reader = e.get_reader();
        if get_reader.is_err() {
            error!("failed to read key {:?} from db", k);
            return Vec::new();
        }
        let reader: lmdb_rs::ReadonlyTransaction = get_reader.unwrap();
        let db: Database = reader.bind(h);
        let mut result: Vec<u8> = Vec::new();
        for num_writes in 0..usize::MAX {
            let mut new_key: Vec<u8> = k.to_vec();
            let mut key_count: Vec<u8> = (num_writes).to_be_bytes().to_vec();
            new_key.append(&mut key_count);
            let mut r = db.get::<Vec<u8>>(&new_key).unwrap_or_default();
            if r.is_empty() {
                break;
            }
            result.append(&mut r);
        }
        {
            if result.is_empty() {
                error!("failed to read key {:?} from db", k);
            }
        }
        result
    }
    /// Deletes a key/value pair from the database
    pub fn delete(e: &Environment, h: &DbHandle, k: &Vec<u8>) {
        info!("excecuting lmdb delete");
        if k.is_empty() {
            error!("can't delete empty key");
            return;
        }
        let new_txn = e.new_transaction();
        if new_txn.is_err() {
            error!("failed txn deleting key: {:?}", k);
            return;
        }
        let txn = new_txn.unwrap();
        let get_reader = e.get_reader();
        if get_reader.is_err() {
            error!("failed to read key {:?} from db", k);
        }
        let reader: lmdb_rs::ReadonlyTransaction = get_reader.unwrap();
        let db_reader: Database = reader.bind(h);
        {
            let db = txn.bind(h);

            for num_writes in 0..usize::MAX {
                let mut new_key: Vec<u8> = k.to_vec();
                let mut key_count: Vec<u8> = num_writes.to_be_bytes().to_vec();
                new_key.append(&mut key_count);
                let r = db_reader.get::<Vec<u8>>(&new_key).unwrap_or_default();
                if r.is_empty() {
                    break;
                }
                db.del(&new_key)
                    .unwrap_or_else(|_| error!("failed to delete"));
            }
        }
        txn.commit().unwrap()
    }
}

/// Write chunks to the database. This function uses ten percent
///
/// of the map size . Setting the map_size to a low value
///
/// will cause degraded performance.
pub fn write_chunks(e: &Environment, h: &DbHandle, k: &[u8], v: &[u8]) {
    let s = System::new_all();
    let chunk_size = s.available_memory() as f32 * CHUNK_SIZE_MEMORY_RATIO;
    let mut writes: usize = 0;
    let mut index: usize = 0;
    let mut key_counter: usize = 0;
    let mut chunk: Vec<u8> = Vec::new();
    loop {
        while writes < chunk_size as usize {
            chunk.push(v[index]);
            if index == v.len() - 1 {
                break;
            }
            index += 1;
            writes += 1;
        }
        writes = 0; // reset chunks
        let mut old_key: Vec<u8> = k.to_vec();
        let mut append: Vec<u8> = (key_counter).to_be_bytes().to_vec();
        old_key.append(&mut append);
        DatabaseEnvironment::write(e, h, &old_key, &chunk);
        key_counter += 1;
        chunk = Default::default(); // empty chunk container for next write
        if index == v.len() - 1 {
            break;
        }
    }
}

// Tests
//-------------------------------------------------------------------------------
#[cfg(test)]
mod tests {

    use super::*;

    use rand::RngCore;

    #[test]
    fn environment_test() {
        const MB: usize = 1000000;
        let db = DatabaseEnvironment::open("10-mb-test");
        const DATA_SIZE_10MB: usize = 10000000;
        log::debug!("creating {} MB of data", DATA_SIZE_10MB/MB);
        let mut data = vec![0u8; DATA_SIZE_10MB];
        rand::thread_rng().fill_bytes(&mut data);
        log::debug!("finished creating {} MB of data", DATA_SIZE_10MB/MB);
        let k = "test-key".as_bytes();
        let expected = &data.to_vec();
        log::debug!("writing {} MB of data", DATA_SIZE_10MB/MB);
        write_chunks(&db.env, &db.handle, &Vec::from(k), &Vec::from(data));
        log::debug!("finished writing {} MB of data", DATA_SIZE_10MB/MB);
        let actual = DatabaseEnvironment::read(&db.env, &db.handle, &Vec::from(k));
        assert_eq!(expected.to_vec(), actual);
        DatabaseEnvironment::delete(&db.env, &db.handle, &Vec::from(k));
        let db = DatabaseEnvironment::open("100-mb-test");
        const DATA_SIZE_100MB: usize = 100000000;
        log::debug!("creating {} MB of data", DATA_SIZE_100MB/MB);
        let mut data = vec![0u8; DATA_SIZE_100MB];
        rand::thread_rng().fill_bytes(&mut data);
        log::debug!("finished creating {} MB of data", DATA_SIZE_100MB/MB);
        let k = "test-key".as_bytes();
        let expected = &data.to_vec();
        log::debug!("writing {} MB of data", DATA_SIZE_100MB/MB);
        write_chunks(&db.env, &db.handle, &Vec::from(k), &Vec::from(data));
        log::debug!("finished writing {} MB of data", DATA_SIZE_100MB/MB);
        let actual = DatabaseEnvironment::read(&db.env, &db.handle, &Vec::from(k));
        assert_eq!(expected.to_vec(), actual);
        DatabaseEnvironment::delete(&db.env, &db.handle, &Vec::from(k));
    }
}
