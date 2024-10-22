use std::fs::File;
use std::io::{BufRead, BufReader};
use automatic_review_analyzer::DType;

fn parse_str(s: &str) -> DType { s.parse::<DType>().unwrap() }

pub fn load_toy_data(path_toy_data: &str) -> (Vec<Vec<DType>>, Vec<DType>) {
    let file = File::open(path_toy_data).unwrap();
    let reader = BufReader::new(file);
    let mut features = Vec::new();
    let mut labels = Vec::new();

    for line in reader.lines() {
        match line {
            Ok(line) => {
                let vec: Vec<&str> = line.split('\t').collect();
                assert_eq!(vec.len(), 3);
                features.push(vec![parse_str(vec[1]), parse_str(vec[2])]);
                labels.push(parse_str(vec[0]));
            }
            err=> {
                panic!("{:?}", err)
            }
        }

    }
    (features, labels)
}