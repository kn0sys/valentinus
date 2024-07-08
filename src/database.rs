
//! LMDB core interface.

use std::{fs, path::Path};

use lmdb::{
    Cursor, Environment, Transaction, WriteFlags
};
use log::*;



/// The database environment for handling primary database operations.
pub struct DatabaseEnvironment {
    pub env: Environment
}

impl DatabaseEnvironment {
    /// Open the database for read, write and delete functionality.
    pub fn open(dir: &str) -> Self {
        info!("opening lmdb");
        let user = std::env::var(crate::USER)
            .unwrap_or(String::from(crate::USER)
            .to_lowercase());
        let string_path = format!("/home/{}/.{}/{}" ,user, crate::VALENTINUS, dir);
        fs::create_dir_all(&string_path).expect("failed to create lmdb directory");
        let file_path = Path::new(&string_path);
        let eb = Environment::new().open_with_permissions(file_path, 0o777);
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
            db.unwrap().put(k, v, WriteFlags::NO_DUP_DATA).unwrap();
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
        let mut txn = e.begin_rw_txn().unwrap();
        let mut result: Vec<u8> = Vec::new();
        info!("initialized read result: {:?}", result);
        {
            let db = txn.open_rw_cursor(e.open_db(None).unwrap());
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
