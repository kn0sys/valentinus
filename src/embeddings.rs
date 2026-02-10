#![deny(missing_docs)]
//! ## Example
//!
//! ```rust,no_run
//! use valentinus::embeddings::*;
//! use serde_json::Value;
//! use std::{fs::File, path::Path, sync::Arc};
//! use serde::Deserialize;
//!
//! /// Let's extract reviews and ratings
//! #[derive(Default, Deserialize)]
//! struct Review {
//!     review: Option<String>,
//!     rating: Option<String>,
//!     vehicle_title: Option<String>,
//! }
//!
//! fn foo() -> Result<(), ValentinusError> {
//!     // 1. Create a single, shared Valentinus instance.
//!     let valentinus = Arc::new(Valentinus::new("test_env")?);
//!
//!     // --- Data Loading ---
//!     let mut documents: Vec<String> = Vec::new();
//!     let mut metadata: Vec<Vec<String>> = Vec::new();
//!     let file_path = Path::new(env!("CARGO_MANIFEST_DIR"))
//!         .join("data")
//!         .join("Scraped_Car_Review_tesla.csv");
//!     let file = File::open(file_path).expect("csv file not found");
//!     let mut rdr = csv::Reader::from_reader(file);
//!     for result in rdr.deserialize() {
//!         let record: Review = result.unwrap_or_default();
//!         documents.push(record.review.unwrap_or_default());
//!         let rating: u64 = record.rating.unwrap_or_default().parse::<u64>().unwrap_or_default();
//!         let mut year: String = record.vehicle_title.unwrap_or_default();
//!         if !year.is_empty() {
//!             year = year[0..5].to_string();
//!         }
//!         metadata.push(vec![
//!             format!(r#"{{"Year": {}}}"#, year),
//!             format!(r#"{{"Rating": {}}}"#, rating),
//!         ]);
//!     }
//!     let mut ids: Vec<String> = Vec::new();
//!     for i in 0..documents.len() {
//!         ids.push(format!("id{}", i));
//!     }
//!
//!     // 2. Define collection parameters
//!     let model_path = String::from("all-MiniLM-L6-v2_onnx");
//!     let model_type = ModelType::AllMiniLmL6V2;
//!     let collection_name = String::from("test_collection");
//!
//!     // 3. Create the collection using the new API
//!     valentinus.create_collection(
//!         collection_name.clone(),
//!         documents,
//!         metadata,
//!         ids,
//!         model_type,
//!         model_path,
//!     )?;
//!
//!     // 4. Query the collection
//!     let query_string = String::from("Find the best reviews.");
//!     let result = valentinus.cosine_query(
//!         query_string.clone(),
//!         collection_name.clone(),
//!         10,
//!         Some(vec![
//!             String::from(r#"{ "Year": {"eq": 2017} }"#),
//!             String::from(r#"{ "Rating": {"gt": 3} }"#),
//!         ]),
//!     )?;
//!
//!     assert_eq!(result.get_docs().len(), 10);
//!
//!     // 5. Delete the collection
//!     valentinus.delete_collection(&collection_name)?;
//!
//!     Ok(())
//! }
//! ```

use crate::{database::*, md2f::filter_where, onnx::*};
use kn0sys_lmdb_rs as lmdb;
use kn0sys_lmdb_rs::MdbError;
use kn0sys_nn::distance::L2Dist;
use kn0sys_nn::*;
use log::*;
use ndarray::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};
use thiserror::Error;
use uuid::Uuid;
use wincode::{SchemaRead, SchemaWrite};

// --- Public Structs and Enums ---

/// The primary, thread-safe entry point for all database operations.
///
/// This struct manages the database environment and a thread-safe, in-memory cache
/// for collections to ensure high performance and concurrency safety. An instance
/// of `Valentinus` should be wrapped in an `Arc` and shared across your application.
pub struct Valentinus {
    db: DatabaseEnvironment,
    // A thread-safe, in-memory cache. Key is the collection's internal key (UUID-based).
    collections: Arc<RwLock<HashMap<String, Arc<EmbeddingCollection>>>>,
}

/// A data container for a single collection of embeddings.
///
/// This struct holds all the data related to a collection, including documents,
/// metadata, and the vector embeddings themselves. It is designed to be immutable
/// once created and cached in memory.
#[derive(Clone, Debug, Default, Deserialize, Serialize, SchemaWrite, SchemaRead)]
pub struct EmbeddingCollection {
    /// The original text documents.
    documents: Vec<String>,
    /// The vector embeddings generated from the documents.
    data: Vec<f32>,
    shape: (usize, usize),
    /// Metadata associated with each document, matched by index.
    metadata: Vec<Vec<String>>,
    /// Path to the ONNX model files used for this collection.
    model_path: String,
    /// The type of model used.
    model_type: ModelType,
    /// User-provided IDs for each document.
    ids: Vec<String>,
    /// The internal, unique key for the collection (e.g., "key-uuid").
    key: String,
    /// The user-facing, unique name for the collection (e.g., "view-my_collection").
    view: String,
}

/// Identifier for the model used with the collection.
#[derive(Clone, Debug, Default, Deserialize, Serialize, SchemaWrite, SchemaRead)]
pub enum ModelType {
    /// AllMiniLmL12V2 model.
    AllMiniLmL12V2,
    /// AllMiniLmL6V2 model.
    #[default]
    AllMiniLmL6V2,
    /// A custom model. Be sure to set `VALENTINUS_CUSTOM_DIM` environment
    /// variable to the number of dimensions for that model.
    Custom,
}

/// Container for the `cosine_query` results.
#[derive(Debug, Default, Deserialize, Serialize)]
pub struct CosineQueryResult {
    documents: Vec<String>,
    similarities: Vec<f32>,
    metadata: Vec<Vec<String>>,
}

/// Error handling enum for all operations.
#[derive(Debug, Error)]
pub enum ValentinusError {
    /// Cache read error
    #[error("Cache error: {0}")]
    CacheError(String),
    /// Wincode serialization/deserialization failure.
    #[error("Serialization/deserialization error: {0}")]
    WincodeError(String),
    /// A collection with the given name was not found.
    #[error("Collection '{0}' not found")]
    CollectionNotFound(String),
    /// Cosine query failure.
    #[error("Cosine query failure: {0}")]
    CosineError(String),
    /// LMDB database error.
    #[error("Database error: {0}")]
    DatabaseError(#[from] MdbError),
    /// The provided view name is invalid or already exists.
    #[error("Invalid view name: {0}")]
    InvalidViewName(String),
    /// Failure during metadata filtering.
    #[error("Metadata filter error")]
    Md2fsError,
    /// Failure in nearest neighbors query.
    #[error("Nearest neighbors query failure: {0}")]
    NearestError(String),
    /// Failure to generate embeddings in the ONNX module.
    #[error("ONNX error")]
    OnnxError(OnnxError),
    /// A required resource was not found.
    #[error("Not found: {0}")]
    NotFound(String),
    /// An error occurred during testing.
    #[error("Test failure")]
    TestError,
}

// --- Internal Serialization Structs (for backward compatibility) ---

#[derive(SchemaWrite, SchemaRead)]
struct PreCollection {
    serde: EmbeddingCollection,
}

#[derive(Debug, Default, Deserialize, Serialize, SchemaWrite, SchemaRead)]
struct KeyViewIndexer {
    values: Vec<String>,
}

#[derive(Default, SchemaWrite, SchemaRead)]
struct KVIndexer {
    serde: KeyViewIndexer,
}

// --- Static Constants ---

static VIEWS_NAMING_CHECK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new("^[a-zA-Z0-9_]+$").expect("regex should be valid"));
const VALENTINUS_KEYS: &str = "keys";
const VALENTINUS_VIEWS: &str = "views";
const VALENTINUS_KEY: &str = "key";
const VALENTINUS_VIEW: &str = "view";

// --- Valentinus Implementation ---

impl Valentinus {
    /// Creates a new `Valentinus` instance.
    ///
    /// This should be called once at application startup. The returned instance
    /// should be wrapped in an `Arc` to be shared across threads.
    ///
    /// # Arguments
    ///
    /// * `env` - A name for the database environment (e.g., "production", "test").
    pub fn new(env: &str) -> Result<Self, ValentinusError> {
        let db = DatabaseEnvironment::open(env)?;
        Ok(Valentinus {
            db,
            collections: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Creates a new collection, generates embeddings, and saves it to the database.
    pub fn create_collection(
        &self,
        name: String,
        documents: Vec<String>,
        metadata: Vec<Vec<String>>,
        ids: Vec<String>,
        model_type: ModelType,
        model_path: String,
    ) -> Result<(), ValentinusError> {
        // --- 1. Validate Input ---
        if !VIEWS_NAMING_CHECK.is_match(&name) {
            return Err(ValentinusError::InvalidViewName(format!(
                "Name '{}' must only contain alphanumerics and underscores.",
                name
            )));
        }

        // --- 2. Generate Embeddings ---
        info!("Generating embeddings for new collection '{}'", name);
        let array_embeddings: Array2<f32> =
            batch_embeddings(&model_path, &documents).map_err(ValentinusError::OnnxError)?;
        let shape = (array_embeddings.nrows(), array_embeddings.ncols());
        let data = array_embeddings.into_raw_vec_and_offset().0;
        // --- 3. Prepare Collection Struct ---
        let key = format!("{}-{}", VALENTINUS_KEY, Uuid::new_v4());
        let view = format!("{}-{}", VALENTINUS_VIEW, name);
        let collection = EmbeddingCollection {
            documents,
            data,
            shape,
            metadata,
            model_path,
            model_type,
            ids,
            key,
            view,
        };

        // --- 4. Atomic Database Write ---
        info!("Saving new collection '{}' to database.", name);
        let txn = self.db.env.new_transaction()?;
        {
            let db_handle = &self.db.handle;

            // Check for view name uniqueness within the transaction
            let mut views_indexer = Self::get_indexer_mut(&txn, db_handle, VALENTINUS_VIEWS)?;
            if views_indexer.serde.values.contains(&name) {
                return Err(ValentinusError::InvalidViewName(format!(
                    "View name '{}' already exists.",
                    name
                )));
            }

            // Add new view and key to indexers
            views_indexer.serde.values.push(name.clone());
            let mut keys_indexer = Self::get_indexer_mut(&txn, db_handle, VALENTINUS_KEYS)?;
            keys_indexer.serde.values.push(collection.key.clone());

            // Write the updated indexers
            Self::write_indexer(&txn, db_handle, VALENTINUS_VIEWS, &views_indexer)?;
            Self::write_indexer(&txn, db_handle, VALENTINUS_KEYS, &keys_indexer)?;

            // Write the view-to-key lookup using the full view name
            txn.bind(db_handle)
                .set(&collection.view.as_bytes(), &collection.key.as_bytes())?;

            // Write the main collection data
            let pre_collection = PreCollection {
                serde: collection.clone(),
            };
            let encoded_collection = wincode::serialize(&pre_collection)
                .map_err(|e| ValentinusError::WincodeError(e.to_string()))?;

            write_chunks_in_txn(
                &txn,
                db_handle,
                collection.key.as_bytes(),
                &encoded_collection,
            )?;
        }
        txn.commit()?;

        Ok(())
    }

    /// Retrieves a collection, loading it from the database and caching it if necessary.
    pub fn get_collection(
        &self,
        view_name: &str,
    ) -> Result<Arc<EmbeddingCollection>, ValentinusError> {
        // --- 1. Check cache with a read lock ---
        {
            let cache = self
                .collections
                .read()
                .map_err(|e| ValentinusError::CacheError(e.to_string()))?;
            if let Some(collection) = cache.values().find(|c| c.view.ends_with(view_name)) {
                info!("Cache hit for collection '{}'", view_name);
                return Ok(Arc::clone(collection));
            }
        } // Read lock is released here

        // --- 2. If not in cache, acquire a write lock to load it ---
        let mut cache = self.collections.write().unwrap();

        // Double-check if another thread loaded it while we were waiting for the write lock
        if let Some(collection) = cache.values().find(|c| c.view.ends_with(view_name)) {
            info!("Cache hit for collection '{}' (after lock)", view_name);
            return Ok(Arc::clone(collection));
        }

        // --- 3. Load from DB ---
        info!(
            "Cache miss. Loading collection '{}' from database.",
            view_name
        );
        let key = self.get_key_for_view(view_name)?;
        let collection_data = read(&self.db.env, &self.db.handle, &key.as_bytes().to_vec())?
            .ok_or_else(|| ValentinusError::CollectionNotFound(view_name.to_string()))?;

        let pre_collection: PreCollection = wincode::deserialize(&collection_data)
            .map_err(|e| ValentinusError::WincodeError(e.to_string()))?;

        let collection = Arc::new(pre_collection.serde);
        cache.insert(key, Arc::clone(&collection));

        Ok(collection)
    }

    /// Deletes a collection from the database and removes it from the cache.
    pub fn delete_collection(&self, view_name: &str) -> Result<(), ValentinusError> {
        info!("Deleting collection '{}'", view_name);

        // --- 1. Atomic Database Deletion ---
        let txn = self.db.env.new_transaction()?;
        let key_to_delete: String;
        let full_view_name = format!("{}-{}", VALENTINUS_VIEW, view_name);
        {
            let db_handle = &self.db.handle;

            // Get the internal key from the view-to-key lookup
            let key_bytes = txn
                .bind(db_handle)
                .get::<Vec<u8>>(&full_view_name.as_bytes())
                .map_err(|_| ValentinusError::CollectionNotFound(view_name.to_string()))?;
            key_to_delete = String::from_utf8(key_bytes).unwrap_or_default();

            if key_to_delete.is_empty() {
                return Err(ValentinusError::CollectionNotFound(view_name.to_string()));
            }

            // Update indexers
            let mut views_indexer = Self::get_indexer_mut(&txn, db_handle, VALENTINUS_VIEWS)?;
            views_indexer.serde.values.retain(|v| v != view_name);
            Self::write_indexer(&txn, db_handle, VALENTINUS_VIEWS, &views_indexer)?;

            let mut keys_indexer = Self::get_indexer_mut(&txn, db_handle, VALENTINUS_KEYS)?;
            keys_indexer.serde.values.retain(|k| k != &key_to_delete);
            Self::write_indexer(&txn, db_handle, VALENTINUS_KEYS, &keys_indexer)?;

            // Delete collection data and the view-to-key lookup
            delete_in_txn(&txn, db_handle, key_to_delete.as_bytes())?;
            txn.bind(db_handle).del(&full_view_name.as_bytes())?;
        }
        txn.commit()?;

        // --- 2. Remove from cache ---
        let mut cache = self.collections.write().unwrap();
        cache.remove(&key_to_delete);

        Ok(())
    }

    /// Performs a cosine similarity query against a collection.
    pub fn cosine_query(
        &self,
        query_string: String,
        view_name: String,
        num_results: usize,
        f_where: Option<Vec<String>>,
    ) -> Result<CosineQueryResult, ValentinusError> {
        info!("Starting cosine query on collection '{}'", view_name);
        let collection = self.get_collection(&view_name)?;
        let is_filtering = f_where.is_some();

        // Generate embedding for the query string
        let qv_string = vec![query_string];
        let qv = batch_embeddings(&collection.model_path, &qv_string)
            .map_err(ValentinusError::OnnxError)?;
        let query_embedding = qv.index_axis(Axis(0), 0);

        let mut results: Vec<(f32, String, Vec<String>)> = Vec::new();

        // Consume the flattened data back to Array2
        let collection_embeddings =
            Array2::from_shape_vec(collection.shape, collection.data.clone()).unwrap_or_default();
        // --- Iterate safely using enumerate to get a reliable index ---
        for (index, (cv, sentence)) in collection_embeddings
            .axis_iter(Axis(0))
            .zip(collection.documents.iter())
            .enumerate()
        {
            let metadata = &collection.metadata[index];
            let raw_f = f_where.as_deref().unwrap_or(&[]);

            if !is_filtering
                || filter_where(raw_f, metadata).map_err(|_| ValentinusError::Md2fsError)?
            {
                let dot_product: f32 = query_embedding
                    .iter()
                    .zip(cv.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                results.push((dot_product, sentence.clone(), metadata.clone()));
            }
        }

        // Sort by similarity score (descending)
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Truncate results if necessary
        if num_results > 0 && results.len() > num_results {
            results.truncate(num_results);
        }

        // Format final result
        let (similarities, documents, metadata) = results.into_iter().fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut sims, mut docs, mut metas), (sim, doc, meta)| {
                sims.push(sim);
                docs.push(doc);
                metas.push(meta);
                (sims, docs, metas)
            },
        );

        Ok(CosineQueryResult {
            documents,
            similarities,
            metadata,
        })
    }

    /// Finds the nearest document in a collection using Euclidean distance.
    pub fn nearest_query(
        &self,
        query_string: String,
        view_name: String,
    ) -> Result<String, ValentinusError> {
        info!("Starting nearest query on collection '{}'", view_name);
        let collection = self.get_collection(&view_name)?;

        let qv_string = vec![query_string];
        let qv = batch_embeddings(&collection.model_path, &qv_string)
            .map_err(ValentinusError::OnnxError)?;
        let query_embedding = qv.index_axis(Axis(0), 0);
        let collection_embeddings =
            Array2::from_shape_vec(collection.shape, collection.data.clone()).unwrap_or_default();
        let nn = CommonNearestNeighbour::KdTree
            .batch(&collection_embeddings, L2Dist)
            .map_err(|e| ValentinusError::NearestError(e.to_string()))?;

        let nearest = nn
            .k_nearest(query_embedding, 1)
            .map_err(|e| ValentinusError::NearestError(e.to_string()))?;

        if nearest.is_empty() {
            return Err(ValentinusError::NotFound(
                "No nearest neighbor found.".to_string(),
            ));
        }

        let nearest_embedding = nearest[0].0.to_vec();
        let position = collection_embeddings
            .axis_iter(Axis(0))
            .position(|x| x.to_vec() == nearest_embedding);

        match position {
            Some(idx) => Ok(collection.documents[idx].clone()),
            None => Err(ValentinusError::NotFound(
                "Could not map nearest embedding back to a document.".to_string(),
            )),
        }
    }

    // --- Private Helper Functions ---

    fn get_key_for_view(&self, view_name: &str) -> Result<String, ValentinusError> {
        let reader = self.db.env.get_reader()?;
        let db = reader.bind(&self.db.handle);
        // The lookup key IS the full view name.
        let full_view_name = format!("{}-{}", VALENTINUS_VIEW, view_name);
        let key_bytes = db
            .get::<Vec<u8>>(&full_view_name.as_bytes())
            .map_err(|_| ValentinusError::CollectionNotFound(view_name.to_string()))?;
        String::from_utf8(key_bytes)
            .map_err(|_| ValentinusError::CollectionNotFound("Invalid key format".to_string()))
    }

    fn get_indexer_mut(
        txn: &lmdb::Transaction,
        db_handle: &lmdb::DbHandle,
        indexer_name: &str,
    ) -> Result<KVIndexer, ValentinusError> {
        match txn.bind(db_handle).get::<Vec<u8>>(&indexer_name.as_bytes()) {
            Ok(bytes) => Ok(wincode::deserialize(&bytes)
                .map_err(|e| ValentinusError::WincodeError(e.to_string()))?),
            Err(MdbError::NotFound) => Ok(KVIndexer::default()), // Return empty if not found
            Err(e) => Err(ValentinusError::DatabaseError(e)),
        }
    }

    fn write_indexer(
        txn: &lmdb::Transaction,
        db_handle: &lmdb::DbHandle,
        indexer_name: &str,
        indexer: &KVIndexer,
    ) -> Result<(), ValentinusError> {
        let encoded = wincode::serialize(indexer)
            .map_err(|e| ValentinusError::WincodeError(e.to_string()))?;
        txn.bind(db_handle)
            .set(&indexer_name.as_bytes(), &encoded)?;
        Ok(())
    }
}

// --- Public Accessors for Result Structs ---

impl CosineQueryResult {
    /// Get documents from a query result.
    pub fn get_docs(&self) -> &Vec<String> {
        &self.documents
    }
    /// Get similarities from a query result.
    pub fn get_similarities(&self) -> &Vec<f32> {
        &self.similarities
    }
    /// Get metadata from a query result.
    pub fn get_metadata(&self) -> &Vec<Vec<String>> {
        &self.metadata
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::{fs, fs::File, path::Path};

    /// Test data structure for CSV parsing.
    #[derive(Default, Deserialize, SchemaWrite, SchemaRead)]
    struct Review {
        review: Option<String>,
        rating: Option<String>,
        vehicle_title: Option<String>,
    }

    // Helper to set up a clean test environment
    fn setup_test_env(env_name: &str) -> Arc<Valentinus> {
        let user = std::env::var("USER").unwrap_or_else(|_| "user".to_string());
        let db_path = format!("/home/{}/.{}/{}", user, "valentinus", env_name);
        // Clean up previous test runs
        if Path::new(&db_path).exists() {
            fs::remove_dir_all(&db_path).unwrap();
        }
        Arc::new(Valentinus::new(env_name).unwrap())
    }

    #[test]
    fn test_full_etl_and_query_workflow() -> Result<(), ValentinusError> {
        env_logger::init();
        let valentinus = setup_test_env("full_workflow_test");
        let collection_name = "tesla_reviews".to_string();

        // --- 1. Create Collection ---
        let (documents, metadata, ids) = load_test_csv_data();
        let expected_docs = documents.clone();
        valentinus.create_collection(
            collection_name.clone(),
            documents,
            metadata,
            ids,
            ModelType::AllMiniLmL6V2,
            "all-MiniLM-L6-v2_onnx".to_string(),
        )?;

        // --- 2. Verify creation by getting the collection ---
        let collection = valentinus.get_collection(&collection_name)?;
        assert_eq!(collection.documents, expected_docs);
        assert!(!collection.data.is_empty());

        // --- 3. Test Cosine Query with Filters ---
        let query_string = "Find the best reviews.".to_string();
        let result = valentinus.cosine_query(
            query_string.clone(),
            collection_name.clone(),
            10,
            Some(vec![
                r#"{ "Year": {"eq": 2017} }"#.to_string(),
                r#"{ "Rating": {"gt": 3} }"#.to_string(),
            ]),
        )?;

        assert_eq!(result.get_docs().len(), 10);
        let first_meta = &result.get_metadata()[0];
        let v_year: Value = serde_json::from_str(&first_meta[0]).unwrap();
        let v_rating: Value = serde_json::from_str(&first_meta[1]).unwrap();
        assert_eq!(v_year["Year"].as_u64().unwrap(), 2017);
        assert!(v_rating["Rating"].as_u64().unwrap() > 3);

        // --- 4. Test Cosine Query without Filters ---
        let no_filter_result =
            valentinus.cosine_query(query_string, collection_name.clone(), 5, None)?;
        assert_eq!(no_filter_result.get_docs().len(), 5);

        // --- 5. Test Nearest Query ---
        let nearest_query_str = "Find me some delicious pizza!".to_string();
        // We need a different collection for this test.
        let nearest_collection_name = "nearest_test_coll".to_string();
        let (docs, md, ids) = create_nearest_test_data();
        valentinus.create_collection(
            nearest_collection_name.clone(),
            docs.clone(),
            md,
            ids,
            ModelType::AllMiniLmL6V2,
            "all-MiniLM-L6-v2_onnx".to_string(),
        )?;
        let nearest_doc =
            valentinus.nearest_query(nearest_query_str, nearest_collection_name.clone())?;
        assert_eq!(nearest_doc, docs[3]);

        // --- 6. Delete Collections ---
        valentinus.delete_collection(&collection_name)?;
        valentinus.delete_collection(&nearest_collection_name)?;

        // --- 7. Verify Deletion ---
        let res = valentinus.get_collection(&collection_name);
        assert!(matches!(res, Err(ValentinusError::CollectionNotFound(_))));

        Ok(())
    }

    // Helper function to load test data from CSV
    fn load_test_csv_data() -> (Vec<String>, Vec<Vec<String>>, Vec<String>) {
        let mut documents = Vec::new();
        let mut metadata = Vec::new();
        let file_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join("Scraped_Car_Review_tesla.csv");
        let file = File::open(file_path).expect("csv file not found");
        let mut rdr = csv::Reader::from_reader(file);

        for result in rdr.deserialize() {
            let record: Review = result.unwrap_or_default();
            documents.push(record.review.unwrap_or_default());
            let rating = record
                .rating
                .unwrap_or_default()
                .parse::<u64>()
                .unwrap_or(0);
            let year_str = record.vehicle_title.unwrap_or_default();
            let year = if year_str.len() >= 4 {
                year_str[0..4].to_string()
            } else {
                "0".to_string()
            };
            metadata.push(vec![
                format!(r#"{{"Year": {}}}"#, year),
                format!(r#"{{"Rating": {}}}"#, rating),
            ]);
        }
        let ids = (0..documents.len()).map(|i| format!("id{}", i)).collect();
        (documents, metadata, ids)
    }

    // Helper function for nearest neighbor test data
    fn create_nearest_test_data() -> (Vec<String>, Vec<Vec<String>>, Vec<String>) {
        let docs = [
            "The latest iPhone model comes with impressive features and a powerful camera.",
            "Exploring the beautiful beaches and vibrant culture of Bali is a dream for many travelers.",
            "Einstein's theory of relativity revolutionized our understanding of space and time.",
            "Traditional Italian pizza is famous for its thin crust, fresh ingredients, and wood-fired ovens.",
            "The American Revolution had a profound impact on the birth of the United States as a nation.",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

        let ids = (0..docs.len()).map(|i| format!("id{}", i)).collect();
        let metadata = vec![vec![]; docs.len()]; // Empty metadata for this test
        (docs, metadata, ids)
    }
}
