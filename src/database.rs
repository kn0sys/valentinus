//! Primary LMDB interface for read, write, delete etc.

extern crate lmdb_rs as lmdb;

use lmdb::{
    DbFlags,
    DbHandle,
    EnvBuilder,
    Environment,
};
use log::{
    debug,
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
    /// Instantiation of ```Environment``` and ```DbHandle```
    pub fn open(env: &str) -> Self {
        const MAP_SIZE: u64  = 1 * 1024 * 1024 * 1024;
        info!("excecuting lmdb open");
        let file_path = format!(
            "/home/{}/.{}/",
            std::env::var("USER").unwrap_or(String::from("user")),
            "valentinus",
        );
        let env = EnvBuilder::new()
            // increase map size for writing the multisig txset
            .map_size(MAP_SIZE)
            .open(format!("{}/{}", file_path, env), 0o777)
            .expect(&format!("could not open LMDB at {}", file_path));
        let handle = env.get_default_db(DbFlags::empty()).unwrap();
        DatabaseEnvironment { env, handle }
    }
    /// Write a key/value pair to the database. It is not possible to
    /// 
    /// overwrite an existing key/value pair.
    pub fn write(e: &Environment, h: &DbHandle, k: &Vec<u8>, v: &Vec<u8>) {
        info!("excecuting lmdb write");
        // don't try and write empty keys
        if k.is_empty() {
            error!("can't write empty key");
            return;
        }
        let txn = e.new_transaction().unwrap();
        {
            // get a database bound to this transaction
            let db = txn.bind(&h);
            let pair = vec![(k, v)];
            for &(key, value) in pair.iter() {
                db.set(key, value).unwrap();
            }
        }
        match txn.commit() {
            Err(_) => error!("failed to commit!"),
            Ok(_) => (),
        }
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
        let reader = e.get_reader().unwrap();
        let db = reader.bind(&h);
        let value = db.get::<Vec<u8>>(k).unwrap_or(Vec::new());
        {
            if value.is_empty() {
                debug!("Failed to read from db.")
            }
        }
        value
    }
    /// Deletes a key/value pair from the database
    pub fn delete(e: &Environment, h: &DbHandle, k: &Vec<u8>) {
        info!("excecuting lmdb delete");
        // don't try and delete empty keys
        if k.is_empty() {
            error!("can't delete empty key");
            return;
        }
        let txn = e.new_transaction().unwrap();
        {
            // get a database bound to this transaction
            let db = txn.bind(&h);
            db.del(k).unwrap_or_else(|_| error!("failed to delete"));
        }
        match txn.commit() {
            Err(_) => error!("failed to commit!"),
            Ok(_) => (),
        }
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
