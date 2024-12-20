use valentinus::embeddings::*;

fn main() -> Result<(), ValentinusError> {
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
    let model_path = String::from("all-Mini-LM-L6-v2_onnx");
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
