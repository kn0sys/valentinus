
//! LMDB core interface.

use std::{fs, path::Path};
use lmdb::*;
use log::*;

/// The name of this application
const VALENTINUS: &str = "valentinus";
/// Test environment string for differentiating between dev and prod
pub const TEST: &str = "test";
/// LMDB map size
const MAP_SIZE: usize = 1 * 1024 * 1024 * 1024 * 1024;
// File path string for current user
pub const USER: &str = "USER";
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
    pub env: Environment
}

impl DatabaseEnvironment {
    /// Open the database for read, write and delete functionality.
    pub fn open(dir: &str) -> Self {
        info!("opening lmdb");
        let user = std::env::var(USER)
            .unwrap_or(String::from(USER)
            .to_lowercase());
        let string_path = format!("/home/{}/.{}/{}/lmdb" ,user, VALENTINUS, dir);
        fs::create_dir_all(&string_path).expect("failed to create lmdb directory");
        let file_path = Path::new(&string_path);
        let eb = Environment::new().set_map_size(MAP_SIZE).open_with_permissions(file_path, 0o777);
        // panic because we can't do anything if the database fails to open
        if eb.is_err() {
            panic!("failed to open lmdb");
        }
        // if there was no panic then it's safe to unwrap here
        DatabaseEnvironment { env: eb.unwrap() }
    }
    /// Write a key/value pair to the database. It is not possible to
    /// 
    /// overwrite an existing key/value pair.
    pub fn write(e: &Environment, k: &Vec<u8>, v: &Vec<u8>) {
        info!("writing to lmdb");
        // handle any unwrapping issues here hopefully
        if DatabaseEnvironment::key_exists(e, k) {
            error!("key {:?} already exists", k);
            return;
        }
        let mut txn = e.begin_rw_txn().unwrap();
        {
            let db = txn.open_rw_cursor(e.open_db(None).unwrap());
            db.unwrap().put(k, v, WriteFlags::NO_OVERWRITE).unwrap();
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
    pub fn read(e: &Environment, k: &Vec<u8>) -> Vec<u8> {
        info!("reading from lmdb");
        let txn = e.begin_rw_txn().unwrap();
        let mut result: Vec<u8> = Vec::new();
        info!("initialized read result: {:?}", result);
        {
            let db = txn.open_ro_cursor(e.open_db(None).unwrap());
            let get = db.unwrap().get(Some(k), None, 0);
            if get.is_err() {
                return Vec::new();
            }
            result = Vec::from(get.unwrap().1);
            match txn.commit() {
                Err(_) => error!("failed to commit!"),
                Ok(_) => (),
            }
        }
        return result;
    }
    /// Deletes a key/value pair from the database
    pub fn delete(e: &Environment, k: &Vec<u8>) {
        info!("deleting from lmdb");
        // handle any unwrapping issues here hopefully
        if !DatabaseEnvironment::key_exists(e, k) {
            error!("key {:?} not found", k);
            return;
        }
        let mut txn = e.begin_rw_txn().unwrap();
        {
            txn.del(e.open_db(None).unwrap(), k, None).unwrap();
            match txn.commit() {
                Err(_) => error!("failed to commit!"),
                Ok(_) => (),
            }
        }
    }
    /// Perform key validation before writing and deleting
    pub fn key_exists(e: &Environment, k: &Vec<u8>) -> bool{
        info!("checking for the key: {:?}", k);
        let s: Vec<u8> = DatabaseEnvironment::read(e, k);
        !s.is_empty()
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn environment_test() {
        const TEST_VAL_I32: f32 = -1.63;
        let test_val_string: &str = &format!("{}", TEST_VAL_I32);
        let v_test_key: Vec<u8> = Vec::from("test-key".as_bytes());
        let v_test_val: Vec<u8> = Vec::from(test_val_string.as_bytes());
        let dbenv = DatabaseEnvironment::open("test").env;
        DatabaseEnvironment::write(&dbenv, &v_test_key, &v_test_val);
        let r = DatabaseEnvironment::read(&dbenv, &v_test_key);
        let r_parsed = std::str::from_utf8(&r).unwrap_or(Default::default());
        let float = r_parsed.parse::<f32>().unwrap_or(0.0);
        assert_eq!(r_parsed, test_val_string);
        assert_eq!(float, TEST_VAL_I32);
        // can't write existing key
        DatabaseEnvironment::write(&dbenv, &v_test_key, &v_test_val);
        DatabaseEnvironment::delete(&dbenv, &v_test_key);
        let r = DatabaseEnvironment::read(&dbenv, &v_test_key);
        let r_parsed = std::str::from_utf8(&r).unwrap_or(Default::default());
        assert_eq!(r_parsed, String::new());
        // can't delete non-existent key
        DatabaseEnvironment::delete(&dbenv, &v_test_key);    
    }
}
