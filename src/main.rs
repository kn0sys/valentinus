use rust_bert::pipelines::question_answering::{QaInput, QuestionAnsweringModel};


fn main() {
    let qa_model = QuestionAnsweringModel::new(Default::default()).expect("model");
                                                        
    let question = String::from("Where does Amy live ?");
    let context = String::from("Amy lives in Amsterdam");

    let answers = qa_model.predict(&[QaInput { question, context }], 1, 32);
    println!("answers: {:?}", answers)
}