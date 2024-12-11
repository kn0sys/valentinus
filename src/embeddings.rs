#![deny(missing_docs)]

//! ## Example
//!
//! ```rust
//! use valentinus::embeddings::*;
//! use serde_json::Value;
//! use std::{fs::File, path::Path};
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
//!     let mut documents: Vec<String> = Vec::new();
//!     let mut metadata: Vec<Vec<String>> = Vec::new();
//!     // https://www.kaggle.com/datasets/ankkur13/edmundsconsumer-car-ratings-and-reviews?resource=download&select=Scraped_Car_Review_tesla.csv
//!     let file_path = Path::new(env!("CARGO_MANIFEST_DIR"))
//!         .join("data")
//!         .join("Scraped_Car_Review_tesla.csv");
//!     let file = File::open(file_path).expect("csv file not found");
//!     let mut rdr = csv::Reader::from_reader(file);
//!     for result in rdr.deserialize() {
//!         let record: Review = result.unwrap_or_default();
//!         documents.push(record.review.unwrap_or_default());
//!         let rating: u64 = record
//!             .rating
//!             .unwrap_or_default()
//!             .parse::<u64>()
//!             .unwrap_or_default();
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
//!     let model_path = String::from("all-Mini-LM-L6-v2_onnx");
//!     let model_type = ModelType::AllMiniLmL6V2;
//!     let name = String::from("test_collection");
//!     let expected: Vec<String> = documents.clone();
//!     let mut ec: EmbeddingCollection =
//!         EmbeddingCollection::new(documents, metadata, ids, name, model_type, model_path)?;
//!     let created_docs: &Vec<String> = ec.get_documents();
//!     assert_eq!(expected, created_docs.to_vec());
//!     // save collection to db
//!     ec.save()?;
//!     // query the collection
//!     let query_string: &String = &String::from("Find the best reviews.");
//!     let result: CosineQueryResult = EmbeddingCollection::cosine_query(
//!         String::from(query_string),
//!         String::from(ec.get_view()),
//!         10,
//!         Some(vec![
//!             String::from(r#"{ "Year":   {"eq": 2017} }"#),
//!             String::from(r#"{ "Rating": {"gt": 3} }"#),
//!         ]),
//!     )?;
//!     assert_eq!(result.get_docs().len(), 10);
//!     let v_year: Result<Value, serde_json::Error> =
//!         serde_json::from_str(&result.get_metadata()[0][0]);
//!     let v_rating: Result<Value, serde_json::Error> =
//!         serde_json::from_str(&result.get_metadata()[0][1]);
//!     let rating_filter: u64 = 3;
//!     let year_filter: u64 = 2017;
//!     assert!(v_rating.map_err(|_| ValentinusError::TestError)?["Rating"].as_u64().unwrap_or(0) > rating_filter);
//!     assert_eq!(v_year.map_err(|_| ValentinusError::TestError)?["Year"].as_u64().unwrap_or(0), year_filter);
//!     let no_filter_result: CosineQueryResult = EmbeddingCollection::cosine_query(
//!         String::from(query_string),
//!         String::from(ec.get_view()),
//!         5,
//!         None,
//!     )?;
//!     assert_eq!(no_filter_result.get_docs().len(), 5);
//!     // remove collection from db
//!     EmbeddingCollection::delete(String::from(ec.get_view()))?;
//!     Ok(())
//! }
//! ```

use kn0sys_nn::distance::L2Dist;
use kn0sys_nn::*;
use kn0sys_lmdb_rs::MdbError;
use ndarray::*;
use regex::Regex;
use serde::Deserialize;
use serde::Serialize;
use std::sync::LazyLock;
use thiserror::Error;
use uuid::Uuid;

use crate::{database::*, md2f::filter_where, onnx::*};
use log::*;

/// Views naming restriction. Required to be alphanumeric/unederscore
static VIEWS_NAMING_CHECK: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new("^[a-zA-Z0-9_]+$").expect("regex should be valid")
});

/// Identifier for model used with the collection.
///
/// Be sure to set `VALENTINUS_CUSTOM_DIM` environment
///
/// variable to the number of dimensions for that model.
#[derive(Debug, Default, Deserialize, Serialize)]
pub enum ModelType {
    /// AllMiniLmL12V2 model
    AllMiniLmL12V2,
    /// AllMiniLmL6V2 model
    #[default]
    AllMiniLmL6V2,
    /// You can also use any model you like
    Custom,
}

/// Use to write the vector of keys and indexes
#[derive(Debug, Default, Deserialize, Serialize)]
pub struct KeyViewIndexer {
    values: Vec<String>,
}

impl KeyViewIndexer {
    /// Used to create a new indexer.
    fn new(v: &[String]) -> KeyViewIndexer {
        KeyViewIndexer { values: v.to_vec() }
    }
    /// Accessor for values of the indexer
    pub fn get_values(&self) -> &Vec<String> {
        &self.values
    }
}

/// Container for the `cosine_query` results
#[derive(Debug, Default, Deserialize, Serialize)]
pub struct CosineQueryResult {
    documents: Vec<String>,
    similarities: Vec<f32>,
    metadata: Vec<Vec<String>>,
}

impl CosineQueryResult {
    /// Used to create a result from `cosine_query`.
    pub fn create(
        documents: Vec<String>,
        similarities: Vec<f32>,
        metadata: Vec<Vec<String>>,
    ) -> CosineQueryResult {
        CosineQueryResult {
            documents,
            similarities,
            metadata,
        }
    }
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

/// Error handling enum for valentinus
#[derive(Debug, Error)]
pub enum ValentinusError {
    /// Bincode failure to serialize/desearilaize
    #[error("deserialization error")]
    BincodeError,
    /// Cosine query failure
    #[error("Cosine query failure")]
    CosineError,
    /// LMDB bindings error
    #[error("LMDB error: {0}")]
    DatabaseError(MdbError),
    /// View name must contain alphanumerics, underscores and be unique
    #[error("Invalid view name. View name must contain alphanumerics, underscores and be unique")]
    InvalidViewName,
    /// Failure to filter metadata
    #[error("Metadata filter error")]
    Md2fsError,
    /// Failure in nearest query
    #[error("Nearest neighbors query failure")]
    NearestError,
    /// Failure to generate embeddings in the onnx moduler
    #[error("ONNX error")]
    OnnxError(OnnxError),
    /// Failure to save new collection to the database
    #[error("Failed to save collection")]
    SaveError,
    /// Error in testing
    #[error("Test failure")]
    TestError,
}

/// Want to write a collection to the db?
///
/// Look no further. Use `EmbeddingCollection::new()`
///
/// to create a new EmbeddingCollection. Write it to the
///
/// database with `EmbeddingCollection::save()`.
#[derive(Debug, Default, Deserialize, Serialize)]
pub struct EmbeddingCollection {
    /// Ideally an array of &str slices mapped to a vector
    documents: Vec<String>,
    /// What separates us from the other dbs. Embeddings are set when saving
    embeddings: Array2<f32>,
    /// Genres mapped to their perspective document by index
    metadata: Vec<Vec<String>>,
    /// Path to model.onnx and tokenizer.json
    model_path: String,
    /// model type
    model_type: ModelType,
    /// Ids for each document
    ids: Vec<String>,
    /// Key for the collection itself. Keys are recorded as `keys` as a `Vec<String>`
    key: String,
    /// View name for convenice sake. Lookup is recorded in `views` as a `Vec<String>`
    view: String,
}

impl EmbeddingCollection {
    /// Create a new collection of unstructured data. Must be saved with the `save` method
    pub fn new(
        documents: Vec<String>,
        metadata: Vec<Vec<String>>,
        ids: Vec<String>,
        name: String,
        model_type: ModelType,
        model_path: String,
    ) -> Result<EmbeddingCollection, ValentinusError> {
        if !VIEWS_NAMING_CHECK.is_match(&name) {
            error!(
                "views name {} must only contain alphanumerics/underscores",
                &name
            );
            return Err(ValentinusError::InvalidViewName);
        }
        // check if  the views name is unique
        let db: &DatabaseEnvironment = &DATABASE_LOCK;
        let views_lookup: Vec<u8> = Vec::from(VALENTINUS_VIEWS.as_bytes());
        let views = DatabaseEnvironment::read(&db.env, &db.handle, &views_lookup)
            .map_err(ValentinusError::DatabaseError)?;
        if !views.is_empty() {
            let view_indexer: KeyViewIndexer =
                bincode::deserialize(&views[..]).map_err(|_| ValentinusError::BincodeError)?;
            if view_indexer.values.contains(&name) {
                error!("view name must be unique");
                return Err(ValentinusError::InvalidViewName);
            }
        }
        info!("creating new collection: {}", &name);
        let id: Uuid = Uuid::new_v4();
        let key: String = format!("{}-{}", VALENTINUS_KEY, id);
        let view: String = format!("{}-{}", VALENTINUS_VIEW, name);
        let ec = EmbeddingCollection {
            documents,
            metadata,
            ids,
            key,
            view,
            model_path,
            model_type,
            ..Default::default()
        };
        Ok(ec)
    }
    /// Save a collection to the database. Error if the key already exists.
    pub fn save(&mut self) -> Result<(), ValentinusError> {
        info!("saving new embedding collection: {}", self.view);
        self.set_key_indexes()?;
        self.set_kv_index()?;
        self.set_view_indexes()?;
        // set the embeddings
        let mut embeddings: Array2<f32> = Default::default();
        info!("initialized embeddings: {}", embeddings.len());
        embeddings = batch_embeddings(&self.model_path, &self.documents).unwrap_or_default();
        self.set_embeddings(embeddings);
        let collection: Vec<u8> = bincode::serialize(&self).unwrap_or_default();
        if collection.is_empty() {
            error!("failed to save collection: {}", &self.key);
            return Err(ValentinusError::SaveError);
        }
        let key = &self.key;
        let b_key = Vec::from(key.as_bytes());
        let db: &DatabaseEnvironment = &DATABASE_LOCK;
        write_chunks(&db.env, &db.handle, &b_key, &collection)
            .map_err(ValentinusError::DatabaseError)?;
        Ok(())
    }
    /// Fetch all known keys or views in the database.
    ///
    /// By default the database will return keys. Set the
    ///
    /// views argument to `true` to fetch all the views.
    pub fn fetch_collection_keys(views: bool) -> Result<KeyViewIndexer, ValentinusError> {
        let mut b_key: Vec<u8> = Vec::from(VALENTINUS_KEYS.as_bytes());
        if views {
            info!("setting search to views");
            b_key = Vec::from(VALENTINUS_VIEWS.as_bytes());
        }
        info!("fetching keys embedding collection");
        let db: &DatabaseEnvironment = &DATABASE_LOCK;
        let keys = DatabaseEnvironment::read(&db.env, &db.handle, &b_key)
            .map_err(ValentinusError::DatabaseError)?;
        let indexer: KeyViewIndexer = bincode::deserialize(&keys[..]).unwrap_or_default();
        Ok(indexer)
    }
    /// Send a cosine similarity query on a collection against a query string.
    ///
    /// Setting `num_results=0`, and metadata `None` will return all related results.
    ///
    /// Let `f_where` be a valid ```Vec<&str>``` of JSON strings to filter on. Valid
    ///
    /// filter operations are eq,gt,gte,lt,lte and in for string arrays. Configure
    ///
    /// parallel threads with `ONNX_PARALLEL_THREADS=X`
    pub fn cosine_query(
        query_string: String,
        view_name: String,
        num_results: usize,
        f_where: Option<Vec<String>>,
    ) -> Result<CosineQueryResult, ValentinusError> {
        let is_filtering = f_where.is_some();
        info!("querying {} embedding collection", view_name);
        let collection: EmbeddingCollection = find(None, Some(view_name))?;
        let qv_string = vec![query_string];
        let qv_output = batch_embeddings(&collection.model_path, &qv_string);
        if qv_output.is_err() {
            error!("failed to generate embeddings for query vector");
            return Err(ValentinusError::CosineError);
        }
        let qv = qv_output.unwrap_or_default();
        let cv = collection.embeddings;
        let docs = collection.documents;
        info!("calculating cosine similarity");
        let mut r_docs: Vec<String> = Vec::new();
        let mut r_sims: Vec<f32> = Vec::new();
        let mut r_meta: Vec<Vec<String>> = Vec::new();
        let query = qv.index_axis(Axis(0), 0);
        for (cv, sentence) in cv.axis_iter(Axis(0)).zip(docs.iter()) {
            let index: Option<usize> = docs.iter().rposition(|x| x == sentence);
            let raw_f: &Vec<String> = &f_where.clone().unwrap_or_default();
            let raw_m: &Vec<String> = &collection.metadata[index.unwrap_or_default()];
            let f_where: bool =
                filter_where(raw_f, raw_m).map_err(|_| ValentinusError::Md2fsError)?;
            if !is_filtering || f_where {
                // Calculate cosine similarity against the 'query' sentence.
                let dot_product: f32 = query.iter().zip(cv.iter()).map(|(a, b)| a * b).sum();
                if dot_product > 0.0 {
                    r_docs.push(String::from(sentence));
                    r_sims.push(dot_product);
                    r_meta.push(raw_m.to_vec());
                }
            }
        }
        if r_docs.len() < num_results || num_results == 0 {
            Ok(CosineQueryResult::create(r_docs, r_sims, r_meta))
        } else {
            Ok(CosineQueryResult::create(
                r_docs[0..num_results].to_vec(),
                r_sims[0..num_results].to_vec(),
                r_meta[0..num_results].to_vec(),
            ))
        }
    }
    /// Calculate the nearest vector using KdTree with eclidean distance.
    ///
    /// Returns `String` of the document matching the nearest embedding.
    pub fn nearest_query(
        query_string: String,
        view_name: String,
    ) -> Result<usize, ValentinusError> {
        info!("querying {} embedding collection for nearest", view_name);
        let collection: EmbeddingCollection = find(None, Some(view_name))?;
        let qv_string = vec![query_string];
        let qv_output = batch_embeddings(&collection.model_path, &qv_string);
        if qv_output.is_err() {
            error!("failed to generate embeddings for query vector");
            return Err(ValentinusError::NearestError);
        }
        let qv = qv_output.unwrap_or_default();
        let cv = collection.embeddings;
        info!("computing nearest embedding");
        // Kdtree using Euclidean distance
        let nn = CommonNearestNeighbour::KdTree
            .batch(&cv, L2Dist)
            .map_err(|_| ValentinusError::NearestError)?;
        // Compute the nearest point to the query vector
        let nearest = nn
            .k_nearest(qv.index_axis(Axis(0), 0), 1)
            .map_err(|_| ValentinusError::NearestError)?;
        let location = cv
            .axis_iter(Axis(0))
            .position(|x| x.to_vec() == nearest[0].0.to_vec());
        if location.is_none() {
            log::error!("could not compute nearest");
            return Err(ValentinusError::NearestError);
        }
        Ok(location.unwrap_or_default())
    }
    /// Delete a collection from the database
    pub fn delete(view_name: String) -> Result<(), ValentinusError> { 
        info!("deleting {} embedding collection", view_name);
        let collection: EmbeddingCollection = find(None, Some(String::from(&view_name)))?;
        let db: &DatabaseEnvironment = &DATABASE_LOCK;
        let s_key = String::from(&collection.key);
        let b_key: Vec<u8> = Vec::from(s_key.as_bytes());
        DatabaseEnvironment::delete(&db.env, &db.handle, &b_key)
            .map_err(ValentinusError::DatabaseError)?;
        // update collections keys
        let b_keys: Vec<u8> = Vec::from(VALENTINUS_KEYS.as_bytes());
        let v_keys: Vec<u8> = Vec::from(VALENTINUS_VIEWS.as_bytes());
        let db: &DatabaseEnvironment = &DATABASE_LOCK;
        let all_keys = DatabaseEnvironment::read(&db.env, &db.handle, &b_keys)
            .map_err(ValentinusError::DatabaseError)?;
        let all_views = DatabaseEnvironment::read(&db.env, &db.handle, &v_keys)
            .map_err(ValentinusError::DatabaseError)?;
        let mut keys_indexer: KeyViewIndexer = bincode::deserialize(&all_keys[..]).unwrap_or_default();
        let mut views_indexer: KeyViewIndexer = bincode::deserialize(&all_views[..]).unwrap_or_default();
        let key_del_index = keys_indexer.values.iter().position(|x| *x == String::from(&collection.key)).unwrap();
        keys_indexer.values.remove(key_del_index);
        let views_del_index = views_indexer.values.iter().position(|x| *x == String::from(&view_name)).unwrap();
        views_indexer.values.remove(views_del_index);
        // reset the indexers
        let b_keys_indexer: Vec<u8> =
            bincode::serialize(&keys_indexer).map_err(|_| ValentinusError::BincodeError)?;
        let b_views_indexer: Vec<u8> =
            bincode::serialize(&views_indexer).map_err(|_| ValentinusError::BincodeError)?;
        DatabaseEnvironment::delete(&db.env, &db.handle, &b_keys)
            .map_err(ValentinusError::DatabaseError)?;
        DatabaseEnvironment::delete(&db.env, &db.handle, &v_keys)
            .map_err(ValentinusError::DatabaseError)?;
        write_chunks(&db.env, &db.handle, &b_keys, &b_keys_indexer)
            .map_err(ValentinusError::DatabaseError)?;
        write_chunks(&db.env, &db.handle, &v_keys, &b_views_indexer)
            .map_err(ValentinusError::DatabaseError)?;
        Ok(())
    }
    /// Getter for documents
    pub fn get_documents(&self) -> &Vec<String> {
        &self.documents
    }
    /// Getter for metadata
    pub fn get_metadata(&self) -> &Vec<Vec<String>> {
        &self.metadata
    }
    /// Getter for ids
    pub fn get_ids(&self) -> &Vec<String> {
        &self.ids
    }
    /// Getter for key
    pub fn get_key(&self) -> &String {
        &self.key
    }
    /// Getter for view
    pub fn get_view(&self) -> &String {
        &self.view
    }
    /// Setter for embeddings
    fn set_embeddings(&mut self, embeddings: Array2<f32>) {
        self.embeddings = embeddings;
    }
    /// Sets the list of views in the database
    fn set_view_indexes(&self) -> Result<(), ValentinusError> {
        let db: &DatabaseEnvironment = &DATABASE_LOCK;
        let b_key: Vec<u8> = Vec::from(VALENTINUS_VIEWS.as_bytes());
        // get the current indexes
        let b_keys: Vec<u8> = DatabaseEnvironment::read(&db.env, &db.handle, &b_key)
            .map_err(ValentinusError::DatabaseError)?;
        let kv_index: KeyViewIndexer = bincode::deserialize(&b_keys[..]).unwrap_or_default();
        let mut current_keys: Vec<String> = Vec::new();
        if !kv_index.values.is_empty() {
            for i in kv_index.values {
                current_keys.push(i);
            }
        }
        // set the new index
        current_keys.push(String::from(&self.view));
        let v_indexer: KeyViewIndexer = KeyViewIndexer::new(&current_keys);
        let b_v_indexer: Vec<u8> =
            bincode::serialize(&v_indexer).map_err(|_| ValentinusError::BincodeError)?;
        DatabaseEnvironment::delete(&db.env, &db.handle, &b_key)
            .map_err(ValentinusError::DatabaseError)?;
        write_chunks(&db.env, &db.handle, &b_key, &b_v_indexer)
            .map_err(ValentinusError::DatabaseError)?;
        Ok(())
    }
    /// Sets the lists of keys in the database
    fn set_key_indexes(&self) -> Result<(), ValentinusError> {
        // set the keys indexer
        let db: &DatabaseEnvironment = &DATABASE_LOCK;
        let b_key: Vec<u8> = Vec::from(VALENTINUS_KEYS.as_bytes());
        // get the current indexes
        let b_keys: Vec<u8> = DatabaseEnvironment::read(&db.env, &db.handle, &b_key)
            .map_err(ValentinusError::DatabaseError)?;
        let kv_index: KeyViewIndexer = bincode::deserialize(&b_keys[..]).unwrap_or_default();
        let mut current_keys: Vec<String> = Vec::new();
        if !kv_index.values.is_empty() {
            for i in kv_index.values {
                current_keys.push(i);
            }
        }
        // set the new index
        current_keys.push(String::from(&self.key));
        let k_indexer: KeyViewIndexer = KeyViewIndexer::new(&current_keys);
        let b_k_indexer: Vec<u8> =
            bincode::serialize(&k_indexer).map_err(|_| ValentinusError::BincodeError)?;
        write_chunks(&db.env, &db.handle, &b_key, &b_k_indexer)
            .map_err(ValentinusError::DatabaseError)?;
        Ok(())
    }
    /// Sets key-to-view lookups
    fn set_kv_index(&self) -> Result<(), ValentinusError> {
        let db: &DatabaseEnvironment = &DATABASE_LOCK;
        let kv_lookup_key: String = format!("{}-{}", VALENTINUS_KEY, self.view);
        let b_kv_lookup_key: Vec<u8> = Vec::from(kv_lookup_key.as_bytes());
        let kv_lookup_value: String = String::from(&self.key);
        let b_v_indexer: Vec<u8> = Vec::from(kv_lookup_value.as_bytes());
        write_chunks(&db.env, &db.handle, &b_kv_lookup_key, &b_v_indexer)
            .map_err(ValentinusError::DatabaseError)?;
        Ok(())
    }
}

/// Look up a collection by key or view. If both key and view are passed,
///
/// then key lookup will override the latter.
pub fn find(key: Option<String>, view: Option<String>) -> Result<EmbeddingCollection, ValentinusError> {
    if key.is_some() {
        let db: &DatabaseEnvironment = &DATABASE_LOCK;
        let s_key = key.unwrap_or_default();
        let b_key: Vec<u8> = Vec::from(s_key.as_bytes());
        let collection: Vec<u8> = DatabaseEnvironment::read(&db.env, &db.handle, &b_key)
            .map_err(ValentinusError::DatabaseError)?;
        let result: EmbeddingCollection =
            bincode::deserialize(&collection[..]).map_err(|_| ValentinusError::BincodeError)?;
        Ok(result)
    } else {
        info!("performing key view lookup");
        let db: &DatabaseEnvironment = &DATABASE_LOCK;
        let s_view = view.unwrap_or_default();
        let kv_lookup: String = format!("{}-{}", VALENTINUS_KEY, s_view);
        let b_kv_lookup: Vec<u8> = Vec::from(kv_lookup.as_bytes());
        let key: Vec<u8> = DatabaseEnvironment::read(&db.env, &db.handle, &b_kv_lookup)
            .map_err(ValentinusError::DatabaseError)?;
        let collection: Vec<u8> = DatabaseEnvironment::read(&db.env, &db.handle, &key)
            .map_err(ValentinusError::DatabaseError)?;
        let result: EmbeddingCollection =
            bincode::deserialize(&collection[..]).map_err(|_| ValentinusError::BincodeError)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    use serde::Deserialize;
    use serde_json::Value;
    use std::{fs::File, path::Path};

    /// Let's extract reviews and ratings
    #[derive(Default, Deserialize)]
    struct Review {
        review: Option<String>,
        rating: Option<String>,
        vehicle_title: Option<String>,
    }

    #[test]
    fn cosine_etl_test() -> Result<(), ValentinusError> {
        let mut documents: Vec<String> = Vec::new();
        let mut metadata: Vec<Vec<String>> = Vec::new();
        // https://www.kaggle.com/datasets/ankkur13/edmundsconsumer-car-ratings-and-reviews?resource=download&select=Scraped_Car_Review_tesla.csv
        let file_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join("Scraped_Car_Review_tesla.csv");
        let file = File::open(file_path).expect("csv file not found");
        let mut rdr = csv::Reader::from_reader(file);
        for result in rdr.deserialize() {
            let record: Review = result.unwrap_or_default();
            documents.push(record.review.unwrap_or_default());
            let rating: u64 = record
                .rating
                .unwrap_or_default()
                .parse::<u64>()
                .unwrap_or_default();
            let mut year: String = record.vehicle_title.unwrap_or_default();
            if !year.is_empty() {
                year = year[0..5].to_string();
            }
            metadata.push(vec![
                format!(r#"{{"Year": {}}}"#, year),
                format!(r#"{{"Rating": {}}}"#, rating),
            ]);
        }
        let mut ids: Vec<String> = Vec::new();
        for i in 0..documents.len() {
            ids.push(format!("id{}", i));
        }
        let model_path = String::from("all-MiniLM-L6-v2_onnx");
        let model_type = ModelType::AllMiniLmL6V2;
        let name = String::from("test_collection");
        let expected: Vec<String> = documents.clone();
        let mut ec: EmbeddingCollection =
            EmbeddingCollection::new(documents, metadata, ids, name, model_type, model_path)?;
        let created_docs: &Vec<String> = ec.get_documents();
        assert_eq!(expected, created_docs.to_vec());
        // save collection to db
        ec.save()?;
        // query the collection
        let query_string: &String = &String::from("Find the best reviews.");
        let result: CosineQueryResult = EmbeddingCollection::cosine_query(
            String::from(query_string),
            String::from(ec.get_view()),
            10,
            Some(vec![
                String::from(r#"{ "Year":   {"eq": 2017} }"#),
                String::from(r#"{ "Rating": {"gt": 3} }"#),
            ]),
        )?;
        assert_eq!(result.get_docs().len(), 10);
        let v_year: Result<Value, serde_json::Error> =
            serde_json::from_str(&result.get_metadata()[0][0]);
        let v_rating: Result<Value, serde_json::Error> =
            serde_json::from_str(&result.get_metadata()[0][1]);
        let rating_filter: u64 = 3;
        let year_filter: u64 = 2017;
        assert!(
            v_rating.map_err(|_| ValentinusError::TestError)?["Rating"]
                .as_u64()
                .unwrap_or(0)
                > rating_filter
        );
        assert_eq!(
            v_year.map_err(|_| ValentinusError::TestError)?["Year"]
                .as_u64()
                .unwrap_or(0),
            year_filter
        );
        let no_filter_result: CosineQueryResult = EmbeddingCollection::cosine_query(
            String::from(query_string),
            String::from(ec.get_view()),
            5,
            None,
        )?;
        assert_eq!(no_filter_result.get_docs().len(), 5);
        // remove collection from db
        EmbeddingCollection::delete(String::from(ec.get_view()))?;
        Ok(())
    }

    #[test]
    fn nearest_test() -> Result<(), ValentinusError> {
        let slice_documents: [&str; 10] = [
        "The latest iPhone model comes with impressive features and a powerful camera.",
        "Exploring the beautiful beaches and vibrant culture of Bali is a dream for many travelers.",
        "Einstein's theory of relativity revolutionized our understanding of space and time.",
        "Traditional Italian pizza is famous for its thin crust, fresh ingredients, and wood-fired ovens.",
        "The American Revolution had a profound impact on the birth of the United States as a nation.",
        "Regular exercise and a balanced diet are essential for maintaining good physical health.",
        "Leonardo da Vinci's Mona Lisa is considered one of the most iconic paintings in art history.",
        "Climate change poses a significant threat to the planet's ecosystems and biodiversity.",
        "Startup companies often face challenges in securing funding and scaling their operations.",
        "Beethoven's Symphony No. 9 is celebrated for its powerful choral finale, 'Ode to Joy.'",
        ];
        let mut documents: Vec<String> = Vec::new();
        for slice in 0..slice_documents.len() {
            documents.push(String::from(slice_documents[slice]));
        }
        // no metadata for nearest query
        let metadata: Vec<String> = Vec::new();
        let mut ids: Vec<String> = Vec::new();
        for i in 0..documents.len() {
            ids.push(format!("id{}", i));
        }
        let name = String::from("test_collection");
        let expected: Vec<String> = documents.clone();
        let model_path = String::from("all-MiniLM-L6-v2_onnx");
        let model_type = ModelType::AllMiniLmL6V2;
        let mut ec: EmbeddingCollection = EmbeddingCollection::new(
            documents.clone(),
            vec![metadata],
            ids,
            name,
            model_type,
            model_path,
        )?;
        let created_docs: &Vec<String> = ec.get_documents();
        assert_eq!(expected, created_docs.to_vec());
        // save collection to db
        ec.save()?;
        // query the collection
        let query_string: String = String::from("Find me some delicious food!");
        let result: usize =
            EmbeddingCollection::nearest_query(query_string, String::from(ec.get_view()))?;
        assert_eq!(documents.clone()[result], documents[3]);
        // remove collection from db
        EmbeddingCollection::delete(String::from(ec.get_view()))?;
        Ok(())
    }
}
