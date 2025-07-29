use serde::Deserialize;
use std::{fs::File, path::Path};
use valentinus::embeddings::*;
use std::sync::Arc;

/// Let's extract reviews and ratings
#[derive(Default, Deserialize)]
struct Review {
    review: Option<String>,
    rating: Option<String>,
    vehicle_title: Option<String>,
}

fn main() -> Result<(), ValentinusError> {
    // 1. Create a single, shared Valentinus instance.
     let valentinus = Arc::new(Valentinus::new("test_env")?);

     // --- Data Loading ---
     let mut documents: Vec<String> = Vec::new();
     let mut metadata: Vec<Vec<String>> = Vec::new();
     let file_path = Path::new(env!("CARGO_MANIFEST_DIR"))
         .join("data")
         .join("Scraped_Car_Review_tesla.csv");
     let file = File::open(file_path).expect("csv file not found");
     let mut rdr = csv::Reader::from_reader(file);
     for result in rdr.deserialize() {
         let record: Review = result.unwrap_or_default();
         documents.push(record.review.unwrap_or_default());
         let rating: u64 = record.rating.unwrap_or_default().parse::<u64>().unwrap_or_default();
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

     // 2. Define collection parameters
     let model_path = String::from("all-MiniLM-L6-v2_onnx");
     let model_type = ModelType::AllMiniLmL6V2;
     let collection_name = String::from("test_collection");

     // 3. Create the collection using the new API
     valentinus.create_collection(
         collection_name.clone(),
         documents,
         metadata,
         ids,
         model_type,
         model_path,
     )?;

     // 4. Query the collection
     let query_string = String::from("Find the best reviews.");
     let result = valentinus.cosine_query(
         query_string.clone(),
         collection_name.clone(),
         10,
         Some(vec![
             String::from(r#"{ "Year": {"eq": 2017} }"#),
             String::from(r#"{ "Rating": {"gt": 3} }"#),
         ]),
     )?;

     assert_eq!(result.get_docs().len(), 10);

     // 5. Delete the collection
     valentinus.delete_collection(&collection_name)?;

     Ok(())
}
