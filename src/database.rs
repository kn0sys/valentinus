//! Primary LMDB interface for read, write, delete etc.

extern crate lmdb_rs as lmdb;

use lmdb::{
    DbFlags,
    DbHandle,
    EnvBuilder,
    Environment,
};
use lmdb_rs::Database;
use log::{
    error, info
};
use sysinfo::System;

/// Test environment string for differentiating between dev and prod
pub const TEST: &str = "test";
/// Keys indexer constant for writing all collections keys
pub const VALENTINUS_KEYS: &str = "keys";
/// Views indexer constant for writing all collections view names
pub const VALENTINUS_VIEWS: &str = "views";
/// Key lookup
pub const VALENTINUS_KEY: &str = "key";
/// View lookup
pub const VALENTINUS_VIEW: &str = "view";

/// The database environment for handling primary database operations.
/// 
/// By default the database will be written to /home/user/.valentinus/<ENV>/lmdb
pub struct DatabaseEnvironment {
    pub env: Environment,
    pub handle: DbHandle,
}

impl DatabaseEnvironment {
    /// Opens environment in specified path
    pub fn open(env: &str) -> Self {
        const MAP_SIZE: u64  = 1024 * 1024 * 1024;
        let mut user: String = match std::env::var("LMDB_USER") {
            Err(_) => String::new(),
            Ok(user) => user,
        };
        if user == String::new() {
            user = String::from("user");
            error!("LMDB_USER environment variable not set, defaulting to \"user\"")
        }
        info!("excecuting lmdb open");
        let file_path: String = format!("/home/{}/.{}/", user, "valentinus");
        let env: Environment = EnvBuilder::new()
            .map_size(MAP_SIZE)
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
    /// overwrite an existing key/value pair.
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
            let pair: Vec<(&Vec<u8>, &Vec<u8>)> = vec![(&k, v)];
            for &(key, value) in pair.iter() {
                db.set(key, value).unwrap_or_else(|_| error!("failed to set key: {:?}", k));
            }
        }
        txn.commit().unwrap()
    }
    /// Read key from the database. If it doesn't exist then
    /// 
    /// an empty vector will be returned. Treat all empty vectors
    /// 
    /// from database operations failures.
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
            if r.is_empty() { break; }
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
                if r.is_empty() { break; }
                db.del(&new_key).unwrap_or_else(|_| error!("failed to delete"));
            }
        }
        txn.commit().unwrap()
    }
}

/// Write chunks to the database. This function uses available memory
/// 
/// reduced three orders of magnitude.
pub fn write_chunks(e: &Environment, h: &DbHandle, k: &[u8], v: &Vec<u8>) {
    let s = System::new_all();
    let chunk_size = s.available_memory() / 1000;
    let mut writes: usize = 0;
    let mut index: usize = 0;
    let mut key_counter: usize = 0;
    let mut chunk: Vec<u8> = Vec::new();
    loop {
        while writes < chunk_size as usize {
            chunk.push(v[index]);
            if index == v.len() - 1 { break; }
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
        if index == v.len() - 1 { break; }
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
        let db = DatabaseEnvironment::open(TEST);
        const DATA_SIZE: usize = 1000000000;
        let mut data = vec![0u8; DATA_SIZE];
        rand::thread_rng().fill_bytes(&mut data);
        let k = "test-key".as_bytes();
        let expected = &data.to_vec();
        write_chunks(&db.env, &db.handle, &Vec::from(k), &Vec::from(data));
        let actual = DatabaseEnvironment::read(&db.env, &db.handle, &Vec::from(k));
        assert_eq!(expected.to_vec(), actual);
        DatabaseEnvironment::delete(&db.env, &db.handle, &Vec::from(k));   
    }
}
