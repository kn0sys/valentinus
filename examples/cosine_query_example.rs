use serde::Deserialize;
use serde_json::Value;
use std::{fs::File, path::Path};
use valentinus::embeddings::*;

/// Let's extract reviews and ratings
#[derive(Default, Deserialize)]
struct Review {
    review: Option<String>,
    rating: Option<String>,
    vehicle_title: Option<String>,
}

fn main() -> Result<(), ValentinusError> {
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
    let model_path = String::from("all-Mini-LM-L6-v2_onnx");
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
