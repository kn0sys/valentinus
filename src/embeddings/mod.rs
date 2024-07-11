
//! Library for handling embeddings

use database::{VALENTINUS_KEYS, VALENTINUS_VIEWS};
use lmdb::Environment;
use ml::compute_nearest;
use regex::Regex;
use rust_bert::pipelines::sentence_embeddings::{SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType};
use uuid::Uuid;

use serde::{Deserialize, Serialize};
use log::*;
mod database;
mod ml;

const VIEWS_NAMING_CHECK: &str = "^[a-zA-Z0-9_]+$";

/// Use to write the vector of keys and indexes
#[derive(Serialize, Deserialize)]
struct KeyViewIndexer {
    values: Vec<String>,
}

impl Default for KeyViewIndexer {
    fn default() -> Self {
        KeyViewIndexer {
            values: Vec::new()
        }
    }
}

impl KeyViewIndexer {
    pub fn new(v: &Vec<String>) -> KeyViewIndexer {
        KeyViewIndexer {
            values: v.to_vec()
        }
    }
}

/// Want to write a collection to the db?
/// 
/// Look no further. Use `EmbeddingCollection::new()`
/// 
/// to create a new EmbeddingCollection. Write it to the
/// 
/// database with `EmbeddingCollection::save()`.
#[derive(Serialize, Deserialize)] 
struct EmbeddingCollection {
    /// Ideally an array of &str slices mapped to a vector
    documents: Vec<String>,
    /// What separates us from the other dbs. Embeddings are set when saving
    embeddings: Vec<Vec<f32>>,
    /// Genres mapped to their perspective document by index
    metadata: Vec<String>,
    /// Ids for each document
    ids: Vec<String>,
    /// key for the collection itself. Keys are recorded as `keys` as a `Vec<String>`
    key: String,
    /// View name for convenice sake. Lookup is recorded in `vews` as a `Vec<String>`
    view: String
}

impl Default for EmbeddingCollection {
    fn default() -> Self {
        EmbeddingCollection {
           documents: Vec::new(),
           embeddings: Vec::new(),
           metadata: Vec::new(),
           ids: Vec::new(),
           key: Default::default(),
           view: Default::default(),
        }
    }
}

impl EmbeddingCollection {
    /// Create a new collection of unstructured data. Must be saved with the `save` method
    pub fn new(documents: Vec<String>, metadata: Vec<String>, ids: Vec<String>, name: String)
        -> EmbeddingCollection {
            let re = Regex::new(&format!(r"{}", VIEWS_NAMING_CHECK)).unwrap();
            if !re.is_match(&name) {
                error!("views name {} must only contain alphanumerics/underscores", &name);
                return Default::default();
            }
            // check if  the views name is unique
            let dbenv = database::DatabaseEnvironment::open(database::TEST).env;
            let views_lookup: Vec<u8> = Vec::from(database::VALENTINUS_VIEWS.as_bytes());
            let views = database::DatabaseEnvironment::read(&dbenv, &views_lookup);
            let views_parsed = std::str::from_utf8(&views).unwrap_or(Default::default());
            let view_indexer: KeyViewIndexer = serde_json::from_str(views_parsed).unwrap_or(Default::default());
            if view_indexer.values.contains(&name) {
                error!("view name must be unique");
                return Default::default();
            }
            info!("creating new collection: {}", &name);
            let id: Uuid = Uuid::new_v4();
            let key: String = format!("{}-{}", database::VALENTINUS_KEY, id);
            let view: String = format!("{}-{}", database::VALENTINUS_VIEW, name);
            EmbeddingCollection {
                documents, embeddings: Vec::new(), metadata, ids, key, view
            }
    }
    /// Save a collection to the database. Error if the key already exists.
    pub fn save(&mut self) {
        self.set_indexes();
        // set the embeddings
        let model = SentenceEmbeddingsBuilder::remote(
            SentenceEmbeddingsModelType::AllMiniLmL12V2
        ).create_model().expect("model");
        let data_output: Vec<Vec<f32>> = model.encode(&self.documents).expect("embeddings");
        self.set_embeddings(data_output);
        let collection: String = serde_json::to_string(&self).unwrap_or(Default::default());
        if collection == String::new() {
            error!("failed to save collection: {}", &self.key);
        }
        let key = &self.key;
        let b_collection: Vec<u8> = Vec::from(collection.as_bytes());
        let b_key = Vec::from(key.as_bytes());
        let dbenv = database::DatabaseEnvironment::open(database::TEST).env;
        database::DatabaseEnvironment::write(&dbenv,&b_key, &b_collection);
    }
    /// Fetch all known keys or views in the database.
    ///
    /// By default the database will return keys. Set the
    /// 
    /// views argument to `true` to fetch all the views.
    pub fn fetch_collection_keys(views: bool) -> KeyViewIndexer {
        let mut b_key: Vec<u8> = Vec::from(VALENTINUS_KEYS.as_bytes());
        if views {
            b_key = Vec::from(VALENTINUS_VIEWS.as_bytes());
        }
        let dbenv = database::DatabaseEnvironment::open(database::TEST).env;
        let keys = database::DatabaseEnvironment::read(&dbenv,&b_key);
        let keys_parsed = std::str::from_utf8(&keys).unwrap_or(Default::default());
        let indexer: KeyViewIndexer = serde_json::from_str(keys_parsed).unwrap_or(Default::default());
        indexer
    }
    /// Send one query string to a particular set of collections.
    /// 
    /// The name of the query view must be valid. It is possible
    ///
    /// to restrict an embeddings query by setting a valid metadata string.
    pub fn query(query_string: String, view_name: String, metadata: Option<String>) -> String {
        let mut collection: EmbeddingCollection = find(None, Some(view_name));
        let s_metadata: String = metadata.unwrap_or(Default::default());
        let mut index = 0;
        let mut removals = 0;
        // run metadata filter
        if s_metadata != String::new() {
            for m in 0..collection.metadata.len() {
                if collection.metadata[index] == s_metadata {
                    collection.embeddings.remove(m-removals);
                    removals += 1;
                }
                index += 1;
            }
        }
        let model = SentenceEmbeddingsBuilder::remote(
            SentenceEmbeddingsModelType::AllMiniLmL12V2
        ).create_model().expect("model");
        let qv = [query_string];
        let qv_output = &model.encode(&qv).expect("embeddings")[0];
        let nearest = compute_nearest(collection.embeddings, qv_output.to_vec());
        String::from(&collection.documents[nearest])
    }
    // getters
    pub fn get_documents(&self) -> &Vec<String> {
        &self.documents
    }
    pub fn get_genres(&self) -> &Vec<String> {
        &self.metadata
    }
    pub fn get_ids(&self) -> &Vec<String> {
        &self.ids
    }
    pub fn get_key(&self) -> &String {
        &self.key
    }
    // embeddings setter
    fn set_embeddings(&mut self, embeddings: Vec<Vec<f32>>) {
        self.embeddings = embeddings;
    }
    /// Sets indexes for keys, views and key-to-view lookups
    fn set_indexes(&self) {
        // set the keys indexer
        let dbenv: Environment = database::DatabaseEnvironment::open(database::TEST).env;
        let mut k_indexer: KeyViewIndexer = KeyViewIndexer { values: Vec::new() };
        k_indexer.values.push(String::from(&self.key));
        let s_k_indexer: String = serde_json::to_string(&k_indexer).unwrap_or(Default::default());
        let b_k_indexer: Vec<u8> = Vec::from(s_k_indexer.as_bytes());
        let b_key: Vec<u8> = Vec::from(database::VALENTINUS_KEYS.as_bytes());
        database::DatabaseEnvironment::write(&dbenv, &b_key, &b_k_indexer);
        // set the view indexer
        let mut v_indexer: KeyViewIndexer = KeyViewIndexer { values: Vec::new() };
        v_indexer.values.push(String::from(&self.view));
        let s_v_indexer: String = serde_json::to_string(&v_indexer).unwrap_or(Default::default());
        let b_v_indexer: Vec<u8> = Vec::from(s_v_indexer.as_bytes());
        let b_key: Vec<u8> = Vec::from(database::VALENTINUS_VIEWS.as_bytes());
        database::DatabaseEnvironment::write(&dbenv,&b_key, &b_v_indexer);
        // set the key view lookup key-view-<VIEW_NAME> -> <KEY>
        let kv_lookup_key: String = format!("{}-{}-{}",
            database::VALENTINUS_KEY, database::VALENTINUS_VIEW, self.view);
        let b_kv_lookup_key: Vec<u8> = Vec::from(kv_lookup_key.as_bytes());
        let kv_lookup_value: String = String::from(&self.key);
        let b_v_indexer: Vec<u8> = Vec::from(kv_lookup_value.as_bytes());
        database::DatabaseEnvironment::write(&dbenv,&b_kv_lookup_key, &b_v_indexer);
    }
}

/// Look up a collection by key or view. If both key and view are passed,
/// 
/// then key lookup will override the latter.
fn find(key: Option<String>, view: Option<String>) -> EmbeddingCollection {
    if key.is_some() {
        let dbenv = database::DatabaseEnvironment::open(database::TEST).env;
        let s_key = key.unwrap_or(Default::default());
        let b_key: Vec<u8> = Vec::from(s_key.as_bytes());
        let collection = database::DatabaseEnvironment::read(&dbenv, &b_key);
        let collection_parsed = std::str::from_utf8(&collection).unwrap_or(Default::default());
        let result: EmbeddingCollection = serde_json::from_str(collection_parsed).unwrap_or(Default::default());
        result
    } else {
        let dbenv = database::DatabaseEnvironment::open(database::TEST).env;
        let s_view = view.unwrap_or(Default::default());
        let kv_lookup: String = format!("{}-{}-{}", 
            database::VALENTINUS_KEY, database::VALENTINUS_VIEW, s_view);
        let b_kv_lookup: Vec<u8> = Vec::from(kv_lookup.as_bytes());
        let key: Vec<u8> = database::DatabaseEnvironment::read(&dbenv, &b_kv_lookup);
        let collection = database::DatabaseEnvironment::read(&dbenv, &key);
        let collection_parsed = std::str::from_utf8(&collection).unwrap_or(Default::default());
        let result: EmbeddingCollection = serde_json::from_str(collection_parsed).unwrap_or(Default::default());
        result
    }
}
