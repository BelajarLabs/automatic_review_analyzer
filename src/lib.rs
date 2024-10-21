pub type DType = f32;

/// Finds the hinge loss on a single data point given specific classification
/// parameters.
///
/// Args:
///   - `feature_vector` - array describing the given data point.
///   - `label` - float, the correct classification of the data point.
///   - `theta` - array describing the linear classifier.
///   - `theta_0` - float representing the offset parameter.
///
/// Returns: the hinge loss, as a float, associated with the given data point and
///     parameters.
pub fn hinge_loss_single(
    feature_vector: &[DType],
    label: DType,
    theta: &[DType],
    theta_0: DType,
) -> DType {
    let output = feature_vector
        .iter()
        .zip(theta.iter())
        .map(|(&a, &b)| a * b)
        .sum::<DType>()
        + theta_0;
    let one = 1 as DType;
    let zero = 0 as DType;

    (one - output * label).max(zero)
}

/// Finds the hinge loss for given classification parameters averaged over a given dataset
///
/// Args:
/// - `feature_matrix` - matrix describing the given data. Each row represents a single data point.
/// - `labels` - array where the kth element of the array is the correct classification of
///     the kth row of the feature matrix.
/// - `theta` - array describing the linear classifier.
/// - `theta_0` - real valued number representing the offset parameter.
///
/// Returns: the hinge loss, as a float, associated with the given dataset and parameters.
///     This number should be the average hinge loss across all of
pub fn hinge_loss_full(
    feature_matrix: &Vec<Vec<DType>>,
    labels: &[DType],
    theta: &[DType],
    theta_0: DType,
) -> DType {
    feature_matrix
        .iter()
        .zip(labels.iter())
        .map(|(feature_vector, &label)| hinge_loss_single(feature_vector, label, theta, theta_0))
        .sum::<DType>()
        / DType::from(labels.len() as DType)
}

/// Updates the classification parameters `theta` and `theta_0` via a single
/// step of the perceptron algorithm. Returns new parameters rather than
/// modifying in-place.
///
/// Args:
/// * `feature_vector`: Array describing a single data point.
/// * `label` - The correct classification of the feature vector.
/// * `current_theta` - The current theta being used by the perceptron
///   algorithm before this update.
/// * `current_theta_0` - The current theta_0 being used by the perceptron
///    algorithm before this update.
///
/// Returns a tuple containing two values:
/// * the updated feature-coefficient parameter `theta` as a numpy array
/// * the updated offset parameter `theta_0` as a floating point number
pub fn perceptron_single_step_update(
    feature_vector: &[DType],
    label: DType,
    theta: &[DType],
    theta_0: DType,
) -> (Vec<DType>, DType) {
    let output = theta
        .iter()
        .zip(feature_vector.iter())
        .map(|(&a, &b)| a * b)
        .sum::<DType>()
        + theta_0;

    if label * output <= 1e-7 {
        let new_theta = theta
            .iter()
            .zip(
                feature_vector
                    .iter()
                    .map(|&x| x * label)
                    .collect::<Vec<DType>>()
                    .iter(),
            )
            .map(|(&a, &b)| a + b)
            .collect();
        (new_theta, theta_0 + label)
    } else {
        (theta.to_vec(), theta_0)
    }
}

/// Runs the full perceptron algorithm on a given set of data.
/// Runs t iterations through the data set: we do not stop early.
///
/// Args:
/// * `feature_matrix` - matrix describing the given data. Each row
///   represents a single data point.
/// * `labels` - array where the kth element of the array is the
///   correct classification of the kth row of the feature matrix.
/// * `t` - integer indicating how many times the perceptron algorithm
///   should iterate through the feature matrix.
///
/// Returns a tuple containing two values:
/// * the feature-coefficient parameter `theta` as a numpy array
///   (found after T iterations through the feature matrix)
/// * the offset parameter `theta_0` as a floating point number
///   (found also after T iterations through the feature matrix).
pub fn perceptron(
    feature_matrix: &Vec<Vec<DType>>,
    labels: &[DType],
    t: usize,
) -> (Vec<DType>, DType) {
    let n_sample = feature_matrix.len();
    let n_feature = feature_matrix[0].len();

    let mut theta = vec![0 as DType; n_feature];
    let mut theta_0 = 0 as DType;

    for _ in 0..t {
        for i in 0..n_sample {
            let feature_vector = &feature_matrix[i];
            let label = labels[i];
            (theta, theta_0) = perceptron_single_step_update(feature_vector, label, &theta, theta_0)
        }
    }
    (theta, theta_0)
}

/// Runs the average perceptron algorithm on a given dataset.
/// Runs `t` iterations through the dataset (we do not stop early) and
/// therefore averages over `t` many parameter values.
///
/// NOTE: It is more difficult to keep a running average than to sum and
/// divide.
///
/// Args:
/// * `feature_matrix` - A matrix describing the given data. Each row
///   represents a single data point.
/// * `labels` - An array where the kth element of the array is the
///   correct classification of the kth row of the feature matrix.
/// * `t` - An integer indicating how many times the perceptron algorithm
///   should iterate through the feature matrix.
///
/// Returns a tuple containing two values:
/// * the average feature-coefficient parameter `theta` as a numpy array
///   (averaged over T iterations through the feature matrix)
/// * the average offset parameter `theta_0` as a floating point number
///   (averaged also over T iterations through the feature matrix).
pub fn average_perceptron(
    feature_matrix: &Vec<Vec<DType>>,
    labels: &[DType],
    t: usize,
) -> (Vec<DType>, DType) {
    let n_sample = feature_matrix.len();
    let n_feature = feature_matrix[0].len();

    let mut theta = vec![0 as DType; n_feature];
    let mut theta_sum = vec![0 as DType; n_feature];
    let mut theta_0 = 0 as DType;
    let mut theta_0_sum = 0 as DType;

    for _ in 0..t {
        for i in 0..n_sample {
            let feature_vector = &feature_matrix[i];
            let label = labels[i];
            (theta, theta_0) =
                perceptron_single_step_update(feature_vector, label, &theta, theta_0);
            theta_sum = theta_sum
                .iter()
                .zip(theta.iter())
                .map(|(&a, &b)| a + b)
                .collect::<Vec<DType>>();
            theta_0_sum += theta_0;
        }
    }

    let all_iter = (t * n_sample) as DType;
    let new_theta = theta_sum
        .iter()
        .map(|&a| a / all_iter)
        .collect::<Vec<DType>>();
    (new_theta, theta_0_sum / all_iter)
}