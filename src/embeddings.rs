
//! Library for handling embeddings

use lmdb::Environment;
use regex::Regex;
use rust_bert::pipelines::sentence_embeddings::{SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType};
use uuid::Uuid;

use serde::{Deserialize, Serialize};
use log::*;
use crate::database::*;
use crate::ml::*;

/// Use to write the vector of keys and indexes
#[derive(Debug, Serialize, Deserialize)]
pub struct KeyViewIndexer {
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
    fn new(v: &Vec<String>) -> KeyViewIndexer {
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
#[derive(Debug, Serialize, Deserialize)] 
pub struct EmbeddingCollection {
    /// Ideally an array of &str slices mapped to a vector
    documents: Vec<String>,
    /// What separates us from the other dbs. Embeddings are set when saving
    embeddings: Vec<Vec<f32>>,
    /// Genres mapped to their perspective document by index
    metadata: Vec<String>,
    /// Ids for each document
    ids: Vec<String>,
    /// Key for the collection itself. Keys are recorded as `keys` as a `Vec<String>`
    key: String,
    /// View name for convenice sake. Lookup is recorded in `views` as a `Vec<String>`
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
            const VIEWS_NAMING_CHECK: &str = "^[a-zA-Z0-9_]+$";
            let re = Regex::new(&format!(r"{}", VIEWS_NAMING_CHECK)).unwrap();
            if !re.is_match(&name) {
                error!("views name {} must only contain alphanumerics/underscores", &name);
                return Default::default();
            }
            // check if  the views name is unique
            let dbenv = DatabaseEnvironment::open(TEST).env;
            let views_lookup: Vec<u8> = Vec::from(VALENTINUS_VIEWS.as_bytes());
            let views = DatabaseEnvironment::read(&dbenv, &views_lookup);
            let views_parsed = std::str::from_utf8(&views).unwrap_or(Default::default());
            let view_indexer: KeyViewIndexer = serde_json::from_str(views_parsed).unwrap_or(Default::default());
            if view_indexer.values.contains(&name) {
                error!("view name must be unique");
                return Default::default();
            }
            info!("creating new collection: {}", &name);
            let id: Uuid = Uuid::new_v4();
            let key: String = format!("{}-{}", VALENTINUS_KEY, id);
            let view: String = format!("{}-{}", VALENTINUS_VIEW, name);
            EmbeddingCollection {
                documents, embeddings: Vec::new(), metadata, ids, key, view
            }
    }
    /// Save a collection to the database. Error if the key already exists.
    pub fn save(&mut self) {
        info!("saving new embedding collection: {}", self.view);
        // self.set_indexes();
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
        let dbenv = DatabaseEnvironment::open(TEST).env;
        DatabaseEnvironment::write(&dbenv, &b_key, &b_collection);
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
        let dbenv = DatabaseEnvironment::open(TEST).env;
        let keys = DatabaseEnvironment::read(&dbenv,&b_key);
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
        info!("querying {} embedding collection", view_name);
        let mut collection: EmbeddingCollection = find(None, Some(view_name));
        let s_metadata: String = metadata.unwrap_or(Default::default());
        let mut removals = 0;
        // run metadata filter
        if s_metadata != String::new() {
            info!("running {} metadata filter", &s_metadata);
            for m in 0..collection.metadata.len() {
                let real_index = m-removals;
                if collection.metadata[real_index] == s_metadata {
                    collection.embeddings.remove(real_index);
                    removals += 1;
                }
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
    /// Send one query string to a particular set of collections.
    /// 
    /// The name of the query view must be valid. It is possible
    ///
    /// to restrict an embeddings query by setting a valid metadata string.
    pub fn delete(view_name: String) {
        info!("deleting {} embedding collection", view_name);
        let collection: EmbeddingCollection = find(None, Some(view_name));
        let dbenv = DatabaseEnvironment::open(TEST).env;
        let s_key = collection.key;
        let b_key: Vec<u8> = Vec::from(s_key.as_bytes());
        DatabaseEnvironment::delete(&dbenv, &b_key);
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
    pub fn set_embeddings(&mut self, embeddings: Vec<Vec<f32>>) {
        self.embeddings = embeddings;
    }
    /// Sets the list of views in the database
    pub fn set_view_indexes(&self) {
        let dbenv: Environment = DatabaseEnvironment::open(TEST).env;
        let b_key: Vec<u8> = Vec::from(VALENTINUS_VIEWS.as_bytes());
        // get the current indexes
        let b_keys: Vec<u8> = DatabaseEnvironment::read(&dbenv, &b_key);
        let keys_parsed = std::str::from_utf8(&b_keys).unwrap_or(Default::default());
        let kv_index: KeyViewIndexer = serde_json::from_str(keys_parsed)
            .unwrap_or(Default::default());
        let mut current_keys: Vec<String> = Vec::new();
        if !kv_index.values.is_empty() {
            for i in kv_index.values {
                current_keys.push(i);
            }
        }
        // set the new index
        current_keys.push(String::from(&self.view));
        let v_indexer: KeyViewIndexer = KeyViewIndexer::new(&current_keys);
        let s_v_indexer: String = serde_json::to_string(&v_indexer).unwrap_or(Default::default());
        let b_v_indexer: Vec<u8> = Vec::from(s_v_indexer.as_bytes());
        DatabaseEnvironment::delete(&dbenv, &b_key);
        DatabaseEnvironment::write(&dbenv, &b_key, &b_v_indexer);
    }
    /// Sets the lists of keys in the database
    pub fn set_key_indexes(&self) {
        // set the keys indexer
        let dbenv: Environment = DatabaseEnvironment::open(TEST).env;
        let b_key: Vec<u8> = Vec::from(VALENTINUS_KEYS.as_bytes());
        // get the current indexes
        let b_keys: Vec<u8> = DatabaseEnvironment::read(&dbenv, &b_key);
        let keys_parsed = std::str::from_utf8(&b_keys).unwrap_or(Default::default());
        let kv_index: KeyViewIndexer = serde_json::from_str(keys_parsed)
            .unwrap_or(Default::default());
        let mut current_keys: Vec<String> = Vec::new();
        if !kv_index.values.is_empty() {
            for i in kv_index.values {
                current_keys.push(i);
            }
        }
        // set the new index
        current_keys.push(String::from(&self.key));
        let k_indexer: KeyViewIndexer = KeyViewIndexer::new(&current_keys);
        let s_k_indexer: String = serde_json::to_string(&k_indexer).unwrap_or(Default::default());
        let b_k_indexer: Vec<u8> = Vec::from(s_k_indexer.as_bytes());
        DatabaseEnvironment::write(&dbenv, &b_key, &b_k_indexer);
    }
    /// Sets key-to-view lookups
    pub fn set_indexes(&self) {
        let dbenv: Environment = DatabaseEnvironment::open(TEST).env;
        let kv_lookup_key: String = format!("{}-{}-{}", VALENTINUS_KEY, VALENTINUS_VIEW, self.view);
        let b_kv_lookup_key: Vec<u8> = Vec::from(kv_lookup_key.as_bytes());
        let kv_lookup_value: String = String::from(&self.key);
        let b_v_indexer: Vec<u8> = Vec::from(kv_lookup_value.as_bytes());
        DatabaseEnvironment::write(&dbenv, &b_kv_lookup_key, &b_v_indexer);
    }
}

/// Look up a collection by key or view. If both key and view are passed,
/// 
/// then key lookup will override the latter.
pub fn find(key: Option<String>, view: Option<String>) -> EmbeddingCollection {
    if key.is_some() {
        let dbenv = DatabaseEnvironment::open(TEST).env;
        let s_key = key.unwrap_or(Default::default());
        let b_key: Vec<u8> = Vec::from(s_key.as_bytes());
        let collection = DatabaseEnvironment::read(&dbenv, &b_key);
        let collection_parsed = std::str::from_utf8(&collection).unwrap_or(Default::default());
        let result: EmbeddingCollection = serde_json::from_str(collection_parsed)
            .unwrap_or(Default::default());
        result
    } else {
        let dbenv = DatabaseEnvironment::open(TEST).env;
        let s_view = view.unwrap_or(Default::default());
        let kv_lookup: String = format!("{}-{}-{}", 
            VALENTINUS_KEY, VALENTINUS_VIEW, s_view);
        let b_kv_lookup: Vec<u8> = Vec::from(kv_lookup.as_bytes());
        let key: Vec<u8> = DatabaseEnvironment::read(&dbenv, &b_kv_lookup);
        let collection = DatabaseEnvironment::read(&dbenv, &key);
        let collection_parsed = std::str::from_utf8(&collection).unwrap_or(Default::default());
        let result: EmbeddingCollection = serde_json::from_str(collection_parsed)
            .unwrap_or(Default::default());
        result
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn new_collection_test() {
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
        let slice_metadata: [&str; 10] = [
            "technology",
            "travel",
            "science",
            "food",
            "history",
            "fitness",
            "art",
            "climate change",
            "business",
            "music",
        ];
        let mut documents: Vec<String> = Vec::new();
        for slice in 0..slice_documents.len() {
            documents.push(String::from(slice_documents[slice]));
        }
        let mut metadata: Vec<String> = Vec::new();
        for slice in 0..slice_metadata.len() {
            metadata.push(String::from(slice_metadata[slice]));
        }
        let mut ids: Vec<String> = Vec::new();
        for i in 0..documents.len() {
            ids.push(format!("id{}", i));
        }
        let name = String::from("test_collection");
        let expected: Vec<String> = documents.clone();
        let expected_doc: String = String::from(&expected[3]);
        let mut ec: EmbeddingCollection = EmbeddingCollection::new(documents, metadata, ids, name);
        let created_docs: &Vec<String> = ec.get_documents();
        assert_eq!(expected, created_docs.to_vec());
        // save collection to db
        ec.save();
        // query the collection
        let query_string: String = String::from("Find me some delicious food!");
        let result: String = EmbeddingCollection::query(query_string, String::from(&ec.view), None);
        assert_eq!(expected_doc, result);
        // remove collection from db
        EmbeddingCollection::delete(String::from(&ec.view));
    }

}
