
//! Library for handling embeddings

use uuid::Uuid;

use serde::{Deserialize, Serialize};
use log::*;
mod database;
mod ml;

/// Use to write the vector of keys and indexes
#[derive(Serialize, Deserialize)]
struct KeyViewIndexer {
    values: Vec<String>,
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
    embeddings: Vec<Vec<Vec<f32>>>,
    /// Genres mapped to their perspective document by index
    genres: Vec<String>,
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
           genres: Vec::new(),
           ids: Vec::new(),
           key: Default::default(),
           view: Default::default(),
        }
    }
}

impl EmbeddingCollection {
    /// Create a new collection of unstructured data. Must be saved with the `save` method
    pub fn new(documents: Vec<String>, genres: Vec<String>, ids: Vec<String>, name: String)
        -> EmbeddingCollection {
            info!("creating new collection: {}", &name);
            let id: Uuid = Uuid::new_v4();
            let key = format!("{}-{}", database::VALENTINUS_KEY, id);
            let view = format!("{}-{}", database::VALENTINUS_VIEW, name);
            EmbeddingCollection {
                documents, embeddings: Vec::new(), genres, ids, key, view
            }
    }
    /// Look up a collection by key or view. If both key and view are passed,
    /// 
    /// then key lookup will override the latter.
    pub fn find(key: Option<String>, view: Option<String>) -> EmbeddingCollection {

        if key.is_some() {
            let dbenv = database::DatabaseEnvironment::open(database::TEST).env;
            let s_key = key.unwrap_or(Default::default());
            let b_key: Vec<u8> = Vec::from(s_key.as_bytes());
            let r = database::DatabaseEnvironment::read(&dbenv, &b_key);
            let r_parsed = std::str::from_utf8(&r).unwrap_or(Default::default());
            let v: EmbeddingCollection = serde_json::from_str(r_parsed).unwrap_or(Default::default());
            v
        } else {
            let dbenv = database::DatabaseEnvironment::open(database::TEST).env;
            let s_view = view.unwrap_or(Default::default());
            let kv_lookup: String = format!("{}-{}-{}", database::VALENTINUS_KEY, database::VALENTINUS_VIEW, s_view);
            let b_kv_lookup: Vec<u8> = Vec::from(kv_lookup.as_bytes());
            let key: Vec<u8> = database::DatabaseEnvironment::read(&dbenv, &b_kv_lookup);
            let r = database::DatabaseEnvironment::read(&dbenv, &key);
            let r_parsed = std::str::from_utf8(&r).unwrap_or(Default::default());
            let v: EmbeddingCollection = serde_json::from_str(r_parsed).unwrap_or(Default::default());
            v
        }
    }
    /// Save a collection to the database. Error if the key already exists.
    pub fn save(&self) {
        // set the keys indexer

        // set the view indexer

        // set the key view lookup key-view-<VIEW_NAME> -> <KEY>
        
        // set the embeddings

        let value = serde_json::to_string(self).unwrap_or(Default::default());
        if value == String::new() {
            error!("failed to save collection: {}", self.key);
        }
        let key = &self.key;
        let b_value = Vec::from(value.as_bytes());
        let b_key = Vec::from(key.as_bytes());
        let dbenv = database::DatabaseEnvironment::open(database::TEST).env;
        database::DatabaseEnvironment::write(&dbenv,&b_key, &b_value);

        
        // TODO: Query which takes query string and name of view to use
        // convery query string to embeddings 


        // TODO: return all views and keys from db
    }
    // getters
    pub fn get_documents(&self) -> &Vec<String> {
        &self.documents
    }
    pub fn get_genres(&self) -> &Vec<String> {
        &self.genres
    }
    pub fn get_ids(&self) -> &Vec<String> {
        &self.ids
    }
    pub fn get_key(&self) -> &String {
        &self.key
    }
}
