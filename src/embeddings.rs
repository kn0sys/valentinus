#![deny(missing_docs)]

//! Library for handling embeddings.
//!
//! ## Example
//!
//! ```rust
//! use valentinus::embeddings::*;
//!
//! fn foo() {
//!     const SLICE_DOCUMENTS: [&str; 11] = [
//!             "The latest iPhone model comes with impressive features and a powerful camera.",
//!             "Exploring the beautiful beaches and vibrant culture of Bali is a dream for many travelers.",
//!             "Einstein's theory of relativity revolutionized our understanding of space and time.",
//!             "Traditional Italian pizza is famous for its thin crust, fresh ingredients, and wood-fired ovens.",
//!             "The American Revolution had a profound impact on the birth of the United States as a nation.",
//!             "Regular exercise and a balanced diet are essential for maintaining good physical health.",
//!             "Leonardo da Vinci's Mona Lisa is considered one of the most iconic paintings in art history.",
//!             "Climate change poses a significant threat to the planet's ecosystems and biodiversity.",
//!             "Startup companies often face challenges in securing funding and scaling their operations.",
//!             "Beethoven's Symphony No. 9 is celebrated for its powerful choral finale, 'Ode to Joy.'",
//!             "Soy sauce ramen is common, with typical toppings including sliced pork, nori, menma, and scallion.",
//!     ];
//!     const  SLICE_METADATA: [&str; 11] = [
//!             "technology",
//!             "travel",
//!             "science",
//!             "food",
//!             "history",
//!             "fitness",
//!             "art",
//!             "climate change",
//!             "business",
//!             "music",
//!             "food",
//!     ];
//! let documents: Vec<String> = SLICE_DOCUMENTS.iter().map(|s| String::from(*s)).collect();
//! let metadata: Vec<String> = SLICE_METADATA.iter().map(|s| String::from(*s)).collect();
//! let mut ids: Vec<String> = Vec::new();
//! for i in 0..documents.len() {
//!    ids.push(format!("id{}", i));
//! }
//! let model_path = String::from("all-Mini-LM-L6-v2_onnx");
//! let model_type = ModelType::AllMiniLmL6V2.value();
//! let name = String::from("test_collection");
//! let expected: Vec<String> = documents.clone();
//! let mut ec: EmbeddingCollection = EmbeddingCollection::new(documents, metadata, ids, name, model_type, model_path);
//! let created_docs: &Vec<String> = ec.get_documents();
//! assert_eq!(expected, created_docs.to_vec());
//! // save collection to db
//! ec.save();
//! // query the collection
//! let query_string: String = String::from("Find me some delicious food!");
//! let related: CosineQueryResult = EmbeddingCollection::cosine_query(
//!    query_string.clone(),
//!    String::from(ec.get_view()),
//!    3,
//!    Some(vec![String::from("food")]),
//! );
//! let not_related: CosineQueryResult = EmbeddingCollection::cosine_query(
//!    query_string.clone(),
//!    String::from(ec.get_view()),
//!    1,
//!    None,
//! );
//! let all: CosineQueryResult = EmbeddingCollection::cosine_query(
//!    query_string,
//!    String::from(ec.get_view()),
//!    0,
//!    None,
//! );
//! assert!(related.get_docs().len() == 2);
//! assert!(not_related.get_docs().len() == 1);
//! assert!(all.get_docs().len() == SLICE_DOCUMENTS.len());
//! }
//! ```

use lazy_static::lazy_static;
use ndarray::Array2;
use ndarray::Axis;
use regex::Regex;
use serde::Deserialize;
use serde::Serialize;
use uuid::Uuid;

use crate::{database::*, onnx::*};
use log::*;

lazy_static! {
    static ref VIEWS_NAMING_CHECK: Regex = Regex::new("^[a-zA-Z0-9_]+$").unwrap();
}

/// Identifier for model used with the collection.
#[derive(Debug, Deserialize, Serialize)]
pub enum ModelType {
    /// AllMiniLmL12V2 model
    AllMiniLmL12V2,
    /// AllMiniLmL6V2 model
    AllMiniLmL6V2,
    /// You can also use any model you like
    Custom,
}

impl ModelType {
    /// Return `ModelType` as a string value.
    pub fn value(&self) -> String {
        match *self {
            Self::AllMiniLmL12V2 => String::from("AllMiniLmL12V2"),
            Self::AllMiniLmL6V2 => String::from("AllMiniLmL6V2"),
            Self::Custom => String::from("Custom"),
        }
    }
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
}

/// Container for the `cosine_query` results
#[derive(Debug, Default, Deserialize, Serialize)]
pub struct CosineQueryResult {
    documents: Vec<String>,
    similarities: Vec<f32>,
}

impl CosineQueryResult {
    /// Used to create a result from `cosine_query`.
    pub fn create(documents: Vec<String>, similarities: Vec<f32>) -> CosineQueryResult {
        CosineQueryResult {
            documents,
            similarities,
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
    metadata: Vec<String>,
    /// Path to model.onnx and tokenizer.json
    model_path: String,
    /// model type
    model_type: String,
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
        metadata: Vec<String>,
        ids: Vec<String>,
        name: String,
        model_type: String,
        model_path: String,
    ) -> EmbeddingCollection {
        if !VIEWS_NAMING_CHECK.is_match(&name) {
            error!(
                "views name {} must only contain alphanumerics/underscores",
                &name
            );
            return Default::default();
        }
        // check if  the views name is unique
        let db: DatabaseEnvironment = DatabaseEnvironment::open(COLLECTIONS);
        let views_lookup: Vec<u8> = Vec::from(VALENTINUS_VIEWS.as_bytes());
        let views = DatabaseEnvironment::read(&db.env, &db.handle, &views_lookup);
        let view_indexer: KeyViewIndexer = bincode::deserialize(&views[..]).unwrap_or_default();
        if view_indexer.values.contains(&name) {
            error!("view name must be unique");
            return Default::default();
        }
        info!("creating new collection: {}", &name);
        let id: Uuid = Uuid::new_v4();
        let key: String = format!("{}-{}", VALENTINUS_KEY, id);
        let view: String = format!("{}-{}", VALENTINUS_VIEW, name);
        EmbeddingCollection {
            documents,
            embeddings: Default::default(),
            metadata,
            ids,
            key,
            view,
            model_path,
            model_type,
        }
    }
    /// Save a collection to the database. Error if the key already exists.
    pub fn save(&mut self) {
        info!("saving new embedding collection: {}", self.view);
        self.set_key_indexes();
        self.set_kv_index();
        self.set_view_indexes();
        // set the embeddings
        let mut embeddings: Array2<f32> = Default::default();
        info!("initialized embeddings: {}", embeddings.len());
        embeddings = batch_embeddings(&self.model_path, &self.documents).unwrap_or_default();
        self.set_embeddings(embeddings);
        let collection: Vec<u8> = bincode::serialize(&self).unwrap();
        if collection.is_empty() {
            error!("failed to save collection: {}", &self.key);
        }
        let key = &self.key;
        let b_key = Vec::from(key.as_bytes());
        let db: DatabaseEnvironment = DatabaseEnvironment::open(COLLECTIONS);
        write_chunks(&db.env, &db.handle, &b_key, &collection);
    }
    /// Fetch all known keys or views in the database.
    ///
    /// By default the database will return keys. Set the
    ///
    /// views argument to `true` to fetch all the views.
    pub fn fetch_collection_keys(views: bool) -> KeyViewIndexer {
        let mut b_key: Vec<u8> = Vec::from(VALENTINUS_KEYS.as_bytes());
        if views {
            info!("setting search to views");
            b_key = Vec::from(VALENTINUS_VIEWS.as_bytes());
        }
        info!("fetching keys embedding collection");
        let db: DatabaseEnvironment = DatabaseEnvironment::open(COLLECTIONS);
        let keys = DatabaseEnvironment::read(&db.env, &db.handle, &b_key);
        let indexer: KeyViewIndexer = bincode::deserialize(&keys[..]).unwrap();
        indexer
    }
    /// Send a cosine similarity query on a collection against a query string.
    ///
    /// Setting `num_results=0`, and metadata `None` will return all related results.
    pub fn cosine_query(
        query_string: String,
        view_name: String,
        num_results: usize,
        metadata: Option<Vec<String>>,
    ) -> CosineQueryResult {
        let filter: bool = metadata.is_some();
        info!("querying {} embedding collection", view_name);
        let collection: EmbeddingCollection = find(None, Some(view_name));
        let qv_string = vec![query_string];
        let qv_output = batch_embeddings(&collection.model_path, &qv_string);
        if qv_output.is_err() {
            error!("failed to generate embeddings for query vector");
            return Default::default();
        }
        let qv = qv_output.unwrap_or_default();
        let cv = collection.embeddings;
        let docs = collection.documents;
        info!("calculating cosine similarity");
        let mut r_docs: Vec<String> = Vec::new();
        let mut r_sims: Vec<f32> = Vec::new();
        let query = qv.index_axis(Axis(0), 0);
        let meta_filter: Vec<String> = metadata.unwrap_or_default();
        for (cv, sentence) in cv.axis_iter(Axis(0)).zip(docs.iter()) {
            let index: Option<usize> = docs.iter().rposition(|x| x == sentence);
            if !filter || meta_filter.contains(&collection.metadata[index.unwrap_or_default()]) {
                // Calculate cosine similarity against the 'query' sentence.
                let dot_product: f32 = query.iter().zip(cv.iter()).map(|(a, b)| a * b).sum();
                if dot_product > 0.0 {
                    r_docs.push(String::from(sentence));
                    r_sims.push(dot_product);
                }
            }
        }
        if r_docs.len() < num_results || num_results == 0 {
            CosineQueryResult::create(r_docs, r_sims)
        } else {
            CosineQueryResult::create(
                r_docs[0..num_results].to_vec(),
                r_sims[0..num_results].to_vec(),
            )
        }
    }
    /// Delete a collection from the database
    pub fn delete(view_name: String) {
        info!("deleting {} embedding collection", view_name);
        let collection: EmbeddingCollection = find(None, Some(view_name));
        let db: DatabaseEnvironment = DatabaseEnvironment::open(COLLECTIONS);
        let s_key = collection.key;
        let b_key: Vec<u8> = Vec::from(s_key.as_bytes());
        DatabaseEnvironment::delete(&db.env, &db.handle, &b_key);
    }
    /// Getter for documents
    pub fn get_documents(&self) -> &Vec<String> {
        &self.documents
    }
    /// Getter for metadata
    pub fn get_genres(&self) -> &Vec<String> {
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
    fn set_view_indexes(&self) {
        let db: DatabaseEnvironment = DatabaseEnvironment::open(COLLECTIONS);
        let b_key: Vec<u8> = Vec::from(VALENTINUS_VIEWS.as_bytes());
        // get the current indexes
        let b_keys: Vec<u8> = DatabaseEnvironment::read(&db.env, &db.handle, &b_key);
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
        let b_v_indexer: Vec<u8> = bincode::serialize(&v_indexer).unwrap();
        DatabaseEnvironment::delete(&db.env, &db.handle, &b_key);
        write_chunks(&db.env, &db.handle, &b_key, &b_v_indexer);
    }
    /// Sets the lists of keys in the database
    fn set_key_indexes(&self) {
        // set the keys indexer
        let db: DatabaseEnvironment = DatabaseEnvironment::open(COLLECTIONS);
        let b_key: Vec<u8> = Vec::from(VALENTINUS_KEYS.as_bytes());
        // get the current indexes
        let b_keys: Vec<u8> = DatabaseEnvironment::read(&db.env, &db.handle, &b_key);
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
        let b_k_indexer: Vec<u8> = bincode::serialize(&k_indexer).unwrap();
        write_chunks(&db.env, &db.handle, &b_key, &b_k_indexer);
    }
    /// Sets key-to-view lookups
    fn set_kv_index(&self) {
        let db: DatabaseEnvironment = DatabaseEnvironment::open(COLLECTIONS);
        let kv_lookup_key: String = format!("{}-{}", VALENTINUS_KEY, self.view);
        let b_kv_lookup_key: Vec<u8> = Vec::from(kv_lookup_key.as_bytes());
        let kv_lookup_value: String = String::from(&self.key);
        let b_v_indexer: Vec<u8> = Vec::from(kv_lookup_value.as_bytes());
        write_chunks(&db.env, &db.handle, &b_kv_lookup_key, &b_v_indexer);
    }
}

/// Look up a collection by key or view. If both key and view are passed,
///
/// then key lookup will override the latter.
fn find(key: Option<String>, view: Option<String>) -> EmbeddingCollection {
    if key.is_some() {
        let db = DatabaseEnvironment::open(COLLECTIONS);
        let s_key = key.unwrap_or_default();
        let b_key: Vec<u8> = Vec::from(s_key.as_bytes());
        let collection: Vec<u8> = DatabaseEnvironment::read(&db.env, &db.handle, &b_key);
        let result: EmbeddingCollection = bincode::deserialize(&collection[..]).unwrap();
        result
    } else {
        info!("performing key view lookup");
        let db = DatabaseEnvironment::open(COLLECTIONS);
        let s_view = view.unwrap_or_default();
        let kv_lookup: String = format!("{}-{}", VALENTINUS_KEY, s_view);
        let b_kv_lookup: Vec<u8> = Vec::from(kv_lookup.as_bytes());
        let key: Vec<u8> = DatabaseEnvironment::read(&db.env, &db.handle, &b_kv_lookup);
        let collection: Vec<u8> = DatabaseEnvironment::read(&db.env, &db.handle, &key);
        let result: EmbeddingCollection = bincode::deserialize(&collection[..]).unwrap();
        result
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    use std::{fs::File, path::Path};
    use serde::Deserialize;

    /// Let's extract reviews and ratings
    #[derive(Default, Deserialize)]
    #[serde(rename_all = "PascalCase")]
    struct Review {
        review: Option<String>,
        rating: Option<String>,
    }

    #[test]
    fn cosine_etl_test() {
        let mut documents: Vec<String> = Vec::new();
        let mut metadata: Vec<String> = Vec::new();
        // https://www.kaggle.com/datasets/ankkur13/edmundsconsumer-car-ratings-and-reviews?resource=download&select=Scraped_Car_Review_tesla.csv
        let file_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data").join("Scraped_Car_Review_tesla.csv");
        let file = File::open(file_path).expect("csv file not found");
        let mut rdr = csv::Reader::from_reader(file);
        for result in rdr.deserialize() {
            let record: Review = result.unwrap_or_default();
            documents.push(record.review.unwrap_or_default());
            metadata.push(record.rating.unwrap_or_default());
        }
        let mut ids: Vec<String> = Vec::new();
        for i in 0..documents.len() {
            ids.push(format!("id{}", i));
        }
        let model_path = String::from("all-Mini-LM-L6-v2_onnx");
        let model_type = ModelType::AllMiniLmL6V2.value();
        let name = String::from("test_collection");
        let expected: Vec<String> = documents.clone();
        let mut ec: EmbeddingCollection =
            EmbeddingCollection::new(documents, metadata, ids, name, model_type, model_path);
        let created_docs: &Vec<String> = ec.get_documents();
        assert_eq!(expected, created_docs.to_vec());
        // save collection to db
        ec.save();
        // query the collection
        let query_string: String = String::from("Find the best reviews.");
        let result: CosineQueryResult = EmbeddingCollection::cosine_query(
            query_string,
            String::from(ec.get_view()),
            3,
            Some(vec![String::from("5"),String::from("4"),String::from("3")]),
        );
        assert!(!result.get_docs().is_empty());
        // remove collection from db
        EmbeddingCollection::delete(String::from(ec.get_view()));
    }
}
