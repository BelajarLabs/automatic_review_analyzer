use automatic_review_analyzer::{hinge_loss_single, hinge_loss_full};

const EPSILON: f64 = 1e-6;

fn assert_approx_eq(a: f64, b: f64) {
    assert!((a-b).abs() < EPSILON, "{a} is not approximately equal to {b}");
}

#[test]
fn hinge_loss_single_equal_to_one() {
    let feature_vector = vec![
        0.34321631, 0.42850303, 0.06123605, 0.72269115, 0.34960656, 0.44380283, 0.25489608, 0.69820867, 0.83814653, 0.1348976,
    ];
    let label = 1.0;
    let theta = vec![
        0.14568072, 0.11668529, 0.81651253, 0.06918585, 0.14301791, 0.11266264, 0.19615837, 0.07161183, 0.05965544, 0.37065151,
    ];
    let theta_0 = 0.5;
    let exp_result = 0.0;
    assert_eq!(hinge_loss_single(&feature_vector, label, &theta, theta_0), exp_result);
}
#[test]
fn hinge_loss_single_greater_than_one() {
    let feature_vector = vec![
        1.76046538, 2.57361973, 9.40053243, 9.6516573,  2.12785972, 7.5554249, 1.79837719, 9.43768774, 8.68358178, 7.99417603
    ];
    let label = 1.0;
    let theta = vec![
        2.1912703, 1.8087996, 6.24440048, 4.69697617, 9.95644732, 7.22254175, 1.873447, 2.29128435, 3.25084571, 5.40949074
    ];
    let theta_0 = 1.0;
    let exp_result = 0.0;
    assert_eq!(hinge_loss_single(&feature_vector, label, &theta, theta_0), exp_result);
}
#[test]
fn hinge_loss_single_less_than_one() {
    let feature_vector = vec![
        -0.07996878, -0.89657546, -0.58501489, -0.9317741,  -0.52415484, -0.11318748, -0.90875476, -0.5473938, -0.67133676, -0.48071535
    ];
    let label = 1.0;
    let theta = vec![
        0.8590804, 0.80549747, 0.21227458, 0.53494888, 0.06405215, 0.15044203, 0.35912591, 0.61315199, 0.23047684, 0.24151432
    ];
    let theta_0 = 0.8589315757256064;
    let exp_result = 2.5380141942765304;
    assert_eq!(hinge_loss_single(&feature_vector, label, &theta, theta_0), exp_result);
}
#[test]
fn hinge_loss_single_random() {
    let feature_vector = vec![
        0.48534204, 0.77595432, 0.40751524, 0.53417291, 0.08023877, 0.30582226, 0.53681376, 0.62489426, 0.92054914, 0.18845109
    ];
    let label = 1.0;
    let theta = vec![
        0.62405527, 0.47931317, 0.03947189, -0.08863187, -0.46235179, -0.01326435, -0.55618325, 0.52486819, 0.04419793, 0.98681757
    ];
    let theta_0 = 0.0;
    let exp_result = 0.14153552625742138;
    assert_eq!(hinge_loss_single(&feature_vector, label, &theta, theta_0), exp_result);
}
#[test]
fn hinge_loss_single_theta_equal_to_zero() {
    let feature_vector = vec![
        0.28403428, 0.36971559, 0.37635315, 0.13733214, 0.41916425, 0.06258066, 0.62662059, 0.95734666, 0.52591212, 0.52345726
    ];
    let label = 1.0;
    let theta = vec![ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ];
    let theta_0 = 0.0;
    let exp_result = 1.0;
    assert_eq!(hinge_loss_single(&feature_vector, label, &theta, theta_0), exp_result);
}
#[test]
fn hinge_loss_single_equal_to_one_2() {
    let feature_vector = vec![
        0.11692551, 0.35574243, 0.02077854, 0.27615784, 0.62738283, 0.25153235, 0.30315952, 0.29043195, 0.6466254, 0.65292947
    ];
    let label = -1.0;
    let theta = vec![ 0.42762268, 0.14055113, 2.40632867, 0.18105588, 0.07969616, 0.19878158, 0.16492967, 0.17215736, 0.07732452, 0.07657795 ];
    let theta_0 = 0.5;
    let exp_result = 2.0;
    assert_approx_eq(hinge_loss_single(&feature_vector, label, &theta, theta_0), exp_result);
}
#[test]
fn hinge_loss_single_greater_than_one_2() {
    let feature_vector = vec![
        1.75043533, 8.64777967, 1.74757793, 9.07478918, 2.46033903, 8.3400534, 5.32248838, 2.86874309, 7.78261674, 4.87619209
    ];
    let label = -1.0;
    let theta = vec![
        7.64630832, 1.07272516, 9.97913165, 8.06261283, 2.68206629, 8.37545408, 1.16088888, 7.36646127, 6.95468453, 8.63334651
    ];
    let theta_0 = 1.0;
    let exp_result = 315.2522102598708;
    assert_eq!(hinge_loss_single(&feature_vector, label, &theta, theta_0), exp_result);
}
#[test]
fn hinge_loss_single_less_than_one_2() {
    let feature_vector = vec![
        -0.58281663, -0.27973251, -0.26233262, -0.49380652, -0.4498024, -0.33573569, -0.23516868, -0.37900009, -0.59543208, -0.17001498
    ];
    let label = -1.0;
    let theta = vec![
        0.38226549, 0.3127761, 0.91440281, 0.97399304, 0.97354809, 0.1664733, 0.70587995, 0.68579601, 0.33822579, 0.76158155
    ];
    let theta_0 = 0.6751018838420377;
    let exp_result = 0.0;
    assert_eq!(hinge_loss_single(&feature_vector, label, &theta, theta_0), exp_result);
}
#[test]
fn hinge_loss_single_random_2() {
    let feature_vector = vec![
        0.71370452, 0.98124716, 0.26277638, 0.39845508, 0.27877508, 0.75421661, 0.02825992, 0.84484455, 0.91407868, 0.27702917
    ];
    let label = -1.0;
    let theta = vec![
        -0.0557507, -0.29242321, -0.58360369, -0.33118744, 0.1023671, -0.24114719, -0.62396963, 0.85022524, 0.90046016, -0.02341762
    ];
    let theta_0 = 0.0;
    let exp_result = 1.751889525585883;
    assert_eq!(hinge_loss_single(&feature_vector, label, &theta, theta_0), exp_result);
}
#[test]
fn hinge_loss_single_theta_equal_to_zero_2() {
    let feature_vector = vec![
        0.73649955, 0.97982298, 0.51558729, 0.89340377, 0.88036352, 0.85987263, 0.84843395, 0.61815788, 0.22666699, 0.62274939
    ];
    let label = -1.0;
    let theta = vec![ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ];
    let theta_0 = 0.0;
    let exp_result = 1.0;
    assert_eq!(hinge_loss_single(&feature_vector, label, &theta, theta_0), exp_result);
}

#[test]
fn hinge_loss_full_test() {
    let feature_vector = vec![vec![1., 2.], vec![1., 2.]];
    let labels = vec![1., 1.];
    let theta = vec![-1., 1.];
    let theta_0 = -0.2;
    let exp_result = 1.0 - 0.8;
    assert_eq!(hinge_loss_full(&feature_vector, &labels, &theta, theta_0), exp_result);
}