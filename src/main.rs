use regex::Regex;


fn main() {
    
    let check = "^[a-zA-Z0-9_]+$";
    let re = Regex::new(&format!(r"{}", check)).unwrap();
    if re.is_match("test haha") {
        
    }
    if re.is_match("test_haha") {
        println!("Hello, Valentinus");
    }
}
