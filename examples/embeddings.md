```rust
    use valentinus::embeddings::*;
    use std::fs::File;
    use serde::Deserialize;

    /// Let's extract reviews and ratings
    #[derive(Default, Deserialize)]
    #[serde(rename_all = "PascalCase")]
    struct Review {
        review: Option<String>,
        rating: Option<String>,
    }

    fn main() {
        let mut documents: Vec<String> = Vec::new();
        let mut metadata: Vec<String> = Vec::new();
        // https://www.kaggle.com/datasets/ankkur13/edmundsconsumer-car-ratings-and-reviews?resource=download&select=Scraped_Car_Review_tesla.csv
        let file_path = "Scraped_Car_Review_tesla.csv";
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
        println!("{:#?}", result);
        assert!(!result.get_docs().is_empty());
        // remove collection from db
        EmbeddingCollection::delete(String::from(ec.get_view()));
    }
```
