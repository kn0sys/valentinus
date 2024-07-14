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
    error,
    info,
};


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
    pub fn write(e: &Environment, h: &DbHandle, k: &Vec<u8>, v: &Vec<u8>) {
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
        let value: Vec<u8> = db.get::<Vec<u8>>(k).unwrap_or_default();
        {
            if value.is_empty() {
                error!("failed to read key {:?} from db", k)
            }
        }
        value
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
        {
            let db = txn.bind(h);
            db.del(k).unwrap_or_else(|_| error!("failed to delete"));
        }
        txn.commit().unwrap()
    }
}

// Tests
//-------------------------------------------------------------------------------
#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn environment_test() {
        let db = DatabaseEnvironment::open(TEST);
        let k = "test-key".as_bytes();
        let v = "test-value".as_bytes();
        DatabaseEnvironment::write(&db.env, &db.handle, &Vec::from(k), &Vec::from(v));
        let expected = Vec::from(v);
        let actual = DatabaseEnvironment::read(&db.env, &db.handle, &Vec::from(k));
        assert_eq!(expected, actual);
        DatabaseEnvironment::delete(&db.env, &db.handle, &Vec::from(k));   
    }
}
