type DTYPE = f64;

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
pub fn hinge_loss_single(feature_vector: &[DTYPE], label: DTYPE, theta: &[DTYPE], theta_0: DTYPE) -> DTYPE {
    let y = feature_vector.iter()
        .zip(theta.iter())
        .map(|(&a, &b)| a * b)
        .sum::<DTYPE>() + theta_0;

    (DTYPE::from(1) - y * label).max(DTYPE::from(0))
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
pub fn hinge_loss_full(feature_matrix: &Vec<Vec<DTYPE>>, labels: &[DTYPE], theta: &[DTYPE], theta_0: DTYPE) -> f64 {
    feature_matrix.iter()
        .zip(labels.iter())
        .map(|(feature_vector, &label)| hinge_loss_single(feature_vector, label, theta, theta_0))
        .sum::<DTYPE>() / DTYPE::from(labels.len() as DTYPE)
}