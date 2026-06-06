#![deny(missing_docs)]

//! Logic for interfacing with LMDB.

extern crate kn0sys_lmdb_rs as lmdb;

use lmdb::*;
use log::{error, info};
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

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
    /// Opens environment in specified path.
    pub fn open(env: &str) -> Result<Self, MdbError> {
        let s = System::new_with_specifics(
            RefreshKind::nothing().with_memory(MemoryRefreshKind::nothing().with_ram()),
        );
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
        info!("executing lmdb open");
        let file_path: String = format!("/home/{}/.{}/", user, "valentinus");
        let env_path = format!("{}/{}", file_path, env);
        let env: Environment = EnvBuilder::new()
            .map_size(env_map_size)
            .open(&env_path, 0o777)
            .map_err(|e| {
                error!("could not open LMDB at {}: {:?}", env_path, e);
                e
            })?;

        let handle: DbHandle = env.get_default_db(DbFlags::empty())?;
        Ok(DatabaseEnvironment { env, handle })
    }
}

/// Writes a value as a series of chunks within a given transaction.
pub fn write_chunks_in_txn(
    txn: &Transaction,
    h: &DbHandle,
    k: &[u8],
    v: &[u8],
) -> Result<(), MdbError> {
    let s = System::new_with_specifics(
        RefreshKind::nothing().with_memory(MemoryRefreshKind::nothing().with_ram()),
    );
    let chunk_size = (s.available_memory() as f32 * CHUNK_SIZE_MEMORY_RATIO) as usize;

    let chunks: Vec<_> = if v.is_empty() || chunk_size == 0 {
        Vec::new()
    } else {
        v.chunks(chunk_size).collect()
    };
    let num_chunks = chunks.len();

    let db = txn.bind(h);

    // 1. Write the metadata (number of chunks) at the main key.
    db.set(&k.to_vec(), &num_chunks.to_be_bytes().to_vec())?;

    // 2. Write all the data chunks with indexed keys.
    for (i, chunk) in chunks.iter().enumerate() {
        let mut chunk_key = k.to_vec();
        chunk_key.extend_from_slice(&i.to_be_bytes());
        db.set(&chunk_key, &chunk.to_vec())?;
    }
    Ok(())
}

/// Reads a potentially chunked value from the database.
pub fn read(e: &Environment, h: &DbHandle, k: &Vec<u8>) -> Result<Option<Vec<u8>>, MdbError> {
    info!("executing lmdb read for key: {:?}", k);
    if k.is_empty() {
        error!("can't read empty key");
        return Ok(None);
    }

    let reader = e.get_reader()?;
    let db = reader.bind(h);

    let num_chunks_bytes = match db.get::<Vec<u8>>(&k.to_vec()) {
        Ok(bytes) => bytes,
        Err(MdbError::NotFound) => return Ok(None),
        Err(e) => return Err(e),
    };

    if num_chunks_bytes.len() != std::mem::size_of::<usize>() {
        error!("invalid chunk metadata for key {:?}", k);
        return Err(MdbError::Corrupted);
    }
    let num_chunks = usize::from_be_bytes(num_chunks_bytes.try_into().unwrap());

    if num_chunks == 0 {
        return Ok(Some(Vec::new()));
    }

    let s = System::new_with_specifics(
        RefreshKind::nothing().with_memory(MemoryRefreshKind::nothing().with_ram()),
    );
    let chunk_size = (s.available_memory() as f32 * CHUNK_SIZE_MEMORY_RATIO) as usize;
    let mut result: Vec<u8> = Vec::with_capacity(num_chunks * chunk_size);

    for i in 0..num_chunks {
        let mut chunk_key = k.to_vec();
        chunk_key.extend_from_slice(&i.to_be_bytes());
        match db.get::<Vec<u8>>(&chunk_key) {
            Ok(mut chunk_data) => result.append(&mut chunk_data),
            Err(MdbError::NotFound) => {
                error!("data corruption: missing chunk {} for key {:?}", i, k);
                return Err(MdbError::Corrupted);
            }
            Err(e) => return Err(e),
        }
    }

    Ok(Some(result))
}

/// Deletes a chunked value from the database within a given transaction.
pub fn delete_in_txn(txn: &Transaction, h: &DbHandle, k: &[u8]) -> Result<(), MdbError> {
    info!("executing lmdb delete for key: {:?}", k);
    if k.is_empty() {
        error!("can't delete empty key");
        return Err(MdbError::NotFound);
    }

    let db = txn.bind(h);

    let num_chunks = match db.get::<Vec<u8>>(&k.to_vec()) {
        Ok(bytes) if bytes.len() == std::mem::size_of::<usize>() => {
            usize::from_be_bytes(bytes.try_into().unwrap())
        }
        Ok(_) => return Err(MdbError::Corrupted),
        Err(MdbError::NotFound) => 0,
        Err(e) => return Err(e),
    };

    for i in 0..num_chunks {
        let mut chunk_key = k.to_vec();
        chunk_key.extend_from_slice(&i.to_be_bytes());
        match db.del(&chunk_key) {
            Ok(_) | Err(MdbError::NotFound) => continue,
            Err(e) => return Err(e),
        }
    }

    match db.del(&k.to_vec()) {
        Ok(_) | Err(MdbError::NotFound) => (),
        Err(e) => return Err(e),
    }

    Ok(())
}

// Tests
//-------------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, rng};
    use std::{fs, path::Path};

    // Helper to set up a clean test environment for database tests
    fn setup_db_test_env(env_name: &str) -> DatabaseEnvironment {
        let user = std::env::var("USER").unwrap_or_else(|_| "user".to_string());
        let db_path = format!("/home/{}/.{}/{}", user, "valentinus", env_name);
        if Path::new(&db_path).exists() {
            fs::remove_dir_all(&db_path).unwrap();
        }
        DatabaseEnvironment::open(env_name).unwrap()
    }

    #[test]
    fn environment_test() -> Result<(), MdbError> {
        let db = setup_db_test_env("db_env_test");
        const DATA_SIZE_10MB: usize = 10_000_000;
        let mut data = vec![0u8; DATA_SIZE_10MB];
        rng().fill_bytes(&mut data);

        let k = "test-key".as_bytes();
        let expected = data.to_vec();

        // Create a transaction for writing
        let txn = db.env.new_transaction()?;
        write_chunks_in_txn(&txn, &db.handle, k, &data)?;
        txn.commit()?;

        // Read is still standalone
        let actual = read(&db.env, &db.handle, &k.to_vec())?
            .expect("read should return Some(data) for a key that was just written");

        assert_eq!(expected, actual);

        // Create a transaction for deleting
        let txn_del = db.env.new_transaction()?;
        delete_in_txn(&txn_del, &db.handle, k)?;
        txn_del.commit()?;

        // Verify deletion
        let after_delete = read(&db.env, &db.handle, &k.to_vec())?;
        assert!(
            after_delete.is_none(),
            "read should return None after deletion"
        );

        Ok(())
    }
}
