```rust
fn foo() {
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
    let mut ec: EmbeddingCollection = EmbeddingCollection::new(documents, metadata, ids, name);
    let created_docs: &Vec<String> = ec.get_documents();
    assert_eq!(expected, created_docs.to_vec());
    // save collection to db
    ec.save();
    // query the collection
    let query_string: String = String::from("Find me some delicious food!");
    let result: String = EmbeddingCollection::query(query_string, String::from(&ec.view), None);
    assert_eq!(result, documents[3]);
    // remove collection from db
    EmbeddingCollection::delete(String::from(&ec.view));
}
```
