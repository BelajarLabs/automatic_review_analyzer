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
    theta_0: DType
) -> DType {
    let output = feature_vector.iter()
        .zip(theta.iter())
        .map(|(&a, &b)| a * b)
        .sum::<DType>() + theta_0;
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
    theta_0: DType
) -> DType {
    feature_matrix.iter()
        .zip(labels.iter())
        .map(|(feature_vector, &label)| hinge_loss_single(feature_vector, label, theta, theta_0))
        .sum::<DType>() / DType::from(labels.len() as DType)
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
    current_theta: &[DType],
    current_theta_0: DType
) -> (Vec<DType>, DType) {
    let output = current_theta.iter()
        .zip(feature_vector.iter())
        .map(|(&a, &b)| a * b)
        .sum::<DType>()+ current_theta_0;

    if label * output <= 1e-7 {
        let new_theta = current_theta.iter()
            .zip(feature_vector.iter()
                .map(|&x| x * label)
                .collect::<Vec<DType>>()
                .iter())
            .map(|(&a, &b)| a + b)
            .collect();
        (new_theta, current_theta_0 + label)
    } else {
        (current_theta.to_vec(), current_theta_0)
    }
}