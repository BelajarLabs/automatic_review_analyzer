use automatic_review_analyzer::{average_perceptron, pegasos, perceptron, DType};
use crate::utils::load_toy_data;

mod utils;

fn main() {
    let (toy_features, toy_labels) = load_toy_data("data/toy_data.tsv");
    let iteration = 10;
    let lambda = 0.2;
    let thetas_perceptron = perceptron(&toy_features, &toy_labels, iteration);
    let thetas_avg_perceptron = average_perceptron(&toy_features, &toy_labels, iteration);
    let thetas_pegasos = pegasos(&toy_features, &toy_labels, iteration, lambda);

    fn plot_toy_results(algo_name: &str, thetas: (Vec<DType>, DType)) {
        println!("theta for {algo_name} is {:?}", thetas.0);
        println!("theta_0 for {algo_name} is {}", thetas.1);
        // plot_toy_data(algo_name, toy_features, toy_labels, thetas)
    }

    plot_toy_results("Perceptron", thetas_perceptron);
    plot_toy_results("Average Perceptron", thetas_avg_perceptron);
    plot_toy_results("Pegasos", thetas_pegasos);
}
