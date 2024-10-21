use automatic_review_analyzer::{perceptron, perceptron_single_step_update, DType};

const EPSILON: DType = 1e-6;

#[test]
fn perceptron_single_random_feature_theta() {
    let feature_vector = vec![
        -0.11132432,
        -0.0837816,
        0.16656621,
        -0.20512049,
        0.47665681,
        0.10200569,
        -0.04801947,
        0.45310278,
        0.2446836,
        0.0374317,
    ];
    let label = 1.;
    let current_theta = vec![
        0.07538525,
        0.16696633,
        -0.05041946,
        -0.33672255,
        0.19369962,
        -0.46239738,
        0.20816458,
        -0.30134817,
        -0.08002659,
        0.14513672,
    ];
    let current_theta_0 = -0.27227810528223273;
    let exp_result = (
        vec![
            -0.0359391, 0.0831847, 0.1161468, -0.5418430, 0.6703564, -0.3603917, 0.1601451,
            0.1517546, 0.1646570, 0.1825684,
        ],
        0.7277219,
    );
    let result =
        perceptron_single_step_update(&feature_vector, label, &current_theta, current_theta_0);

    assert_eq!(result.1, exp_result.1);
    for (i, (&l, &r)) in result.0.iter().zip(exp_result.0.iter()).enumerate() {
        if (l - r).abs() > EPSILON {
            panic!(
                "assertion failed at index {}: `(left ≈ right)`\n  left[{}]: `{:?}`,\n right[{}]: `{:?}`,\n epsilon: `{:?}`",
                i, i, l, i, r, EPSILON
            )
        }
    }
}
#[test]
fn perceptron_single_correct_prediction() {
    let feature_vector = [
        -0.23541166,
        -0.32890187,
        -0.06320096,
        -0.04480014,
        -0.05434104,
        0.43535076,
        -0.27332757,
        -0.01507195,
        0.24454381,
        0.1017814,
    ];
    let label = -1.;
    let current_theta = [
        -0.28096693,
        -0.23904662,
        0.02729207,
        -0.36202178,
        0.18769505,
        -0.06336949,
        -0.46306607,
        -0.42713717,
        -0.07065512,
        0.14324034,
    ];
    let current_theta_0 = -0.9692423790147044;

    let exp_result = (
        [
            -0.2809669, -0.2390466, 0.0272921, -0.3620218, 0.1876951, -0.0633695, -0.4630661,
            -0.4271372, -0.0706551, 0.1432403,
        ],
        -0.9692424,
    );
    let result =
        perceptron_single_step_update(&feature_vector, label, &current_theta, current_theta_0);
    assert_eq!(result.1, exp_result.1);
    for (i, (&l, &r)) in result.0.iter().zip(exp_result.0.iter()).enumerate() {
        if (l - r).abs() > EPSILON {
            panic!(
                "assertion failed at index {}: `(left ≈ right)`\n  left[{}]: `{:?}`,\n right[{}]: `{:?}`,\n epsilon: `{:?}`",
                i, i, l, i, r, EPSILON
            )
        }
    }
}
#[test]
fn perceptron_single_incorrect_prediction() {
    let feature_vector = [
        0.41321807,
        -0.49555544,
        0.46957006,
        0.31949067,
        0.09035456,
        -0.39419081,
        -0.34367874,
        -0.37607894,
        0.49160941,
        -0.10755936,
    ];
    let label = 1.;
    let current_theta = [
        0.04027332,
        -0.43638651,
        -0.19767345,
        -0.4464023,
        -0.23264801,
        0.45526707,
        -0.1898896,
        0.22145077,
        -0.0493539,
        0.25795827,
    ];
    let current_theta_0 = -0.02897141227932598;

    let exp_result = (
        [
            0.4534914, -0.9319419, 0.2718966, -0.1269116, -0.1422934, 0.0610763, -0.5335683,
            -0.1546282, 0.4422555, 0.1503989,
        ],
        0.9710286,
    );
    let result =
        perceptron_single_step_update(&feature_vector, label, &current_theta, current_theta_0);
    assert!(
        (result.1 - exp_result.1).abs() < EPSILON,
        "{} is not approximately equal to {}",
        result.1,
        exp_result.1
    );
    for (i, (&l, &r)) in result.0.iter().zip(exp_result.0.iter()).enumerate() {
        if (l - r).abs() > EPSILON {
            panic!(
                "assertion failed at index {}: `(left ≈ right)`\n  left[{}]: `{:?}`,\n right[{}]: `{:?}`,\n epsilon: `{:?}`",
                i, i, l, i, r, EPSILON
            )
        }
    }
}
#[test]
fn perceptron_single_boundary_case_positive() {
    let feature_vector = [
        -0.14607097,
        0.3892182,
        -0.45167393,
        0.13363913,
        -0.46040408,
        0.08095903,
        -0.37159382,
        0.29497734,
        0.23746193,
        -0.31641207,
    ];
    let label = 1.;
    let current_theta = [
        -0.2143812,
        0.08752648,
        0.39263208,
        -0.34392773,
        -0.14620263,
        -0.03038324,
        0.32025154,
        0.05368624,
        -0.39489839,
        0.14601083,
    ];
    let current_theta_0 = 0.33620981883250234;

    let exp_result = (
        [
            -0.3604522, 0.4767447, -0.0590419, -0.2102886, -0.6066067, 0.0505758, -0.0513423,
            0.3486636, -0.1574365, -0.1704012,
        ],
        1.3362098,
    );
    let result =
        perceptron_single_step_update(&feature_vector, label, &current_theta, current_theta_0);
    assert!(
        (result.1 - exp_result.1).abs() < EPSILON,
        "{} is not approximately equal to {}",
        result.1,
        exp_result.1
    );
    for (i, (&l, &r)) in result.0.iter().zip(exp_result.0.iter()).enumerate() {
        if (l - r).abs() > EPSILON {
            panic!(
                "assertion failed at index {}: `(left ≈ right)`\n  left[{}]: `{:?}`,\n right[{}]: `{:?}`,\n epsilon: `{:?}`",
                i, i, l, i, r, EPSILON
            )
        }
    }
}
#[test]
fn perceptron_single_boundary_case_negative() {
    let feature_vector = [
        0.30898376,
        0.09788877,
        -0.33066594,
        -0.19155077,
        -0.37597604,
        -0.35184549,
        -0.48478526,
        -0.41782858,
        -0.3721566,
        -0.05259197,
    ];
    let label = -1.;
    let current_theta = [
        -0.19252581,
        -0.32804873,
        -0.44909965,
        0.22620279,
        0.17853993,
        -0.46764514,
        0.48327554,
        -0.02037014,
        0.1419018,
        -0.07241366,
    ];
    let current_theta_0 = 0.16378985030175483;

    let exp_result = (
        [
            -0.5015096, -0.4259375, -0.1184337, 0.4177536, 0.5545160, -0.1157996, 0.9680608,
            0.3974584, 0.5140584, -0.0198217,
        ],
        -0.8362101,
    );
    let result =
        perceptron_single_step_update(&feature_vector, label, &current_theta, current_theta_0);
    assert!(
        (result.1 - exp_result.1).abs() < EPSILON,
        "{} is not approximately equal to {}",
        result.1,
        exp_result.1
    );
    for (i, (&l, &r)) in result.0.iter().zip(exp_result.0.iter()).enumerate() {
        if (l - r).abs() > EPSILON {
            panic!(
                "assertion failed at index {}: `(left ≈ right)`\n  left[{}]: `{:?}`,\n right[{}]: `{:?}`,\n epsilon: `{:?}`",
                i, i, l, i, r, EPSILON
            )
        }
    }
}
#[test]
fn perceptron_test_1() {
    let feature_matrix = vec![vec![1., 2.]];
    let labels = [1.];
    let t = 1;

    let exp_result = (vec![1., 2.], 1.);
    let result = perceptron(&feature_matrix, &labels, t);

    assert!(
        (result.1 - exp_result.1).abs() < EPSILON,
        "{} is not approximately equal to {}",
        result.1,
        exp_result.1
    );
    for (i, (&l, &r)) in result.0.iter().zip(exp_result.0.iter()).enumerate() {
        if (l - r).abs() > EPSILON {
            panic!(
                "assertion failed at index {}: `(left ≈ right)`\n  left[{}]: `{:?}`,\n right[{}]: `{:?}`,\n epsilon: `{:?}`",
                i, i, l, i, r, EPSILON
            )
        }
    }
}
#[test]
fn perceptron_test_2() {
    let feature_matrix = vec![vec![1., 2.], vec![-1., 0.]];
    let labels = [1., 1.];
    let t = 1;

    let exp_result = (vec![0., 2.], 2.);
    let result = perceptron(&feature_matrix, &labels, t);

    assert!(
        (result.1 - exp_result.1).abs() < EPSILON,
        "{} is not approximately equal to {}",
        result.1,
        exp_result.1
    );
    for (i, (&l, &r)) in result.0.iter().zip(exp_result.0.iter()).enumerate() {
        if (l - r).abs() > EPSILON {
            panic!(
                "assertion failed at index {}: `(left ≈ right)`\n  left[{}]: `{:?}`,\n right[{}]: `{:?}`,\n epsilon: `{:?}`",
                i, i, l, i, r, EPSILON
            )
        }
    }
}
#[test]
fn perceptron_test_3() {
    let feature_matrix = vec![vec![1., 2.]];
    let labels = [1.];
    let t = 2;

    let exp_result = (vec![1., 2.], 1.);
    let result = perceptron(&feature_matrix, &labels, t);

    assert!(
        (result.1 - exp_result.1).abs() < EPSILON,
        "{} is not approximately equal to {}",
        result.1,
        exp_result.1
    );
    for (i, (&l, &r)) in result.0.iter().zip(exp_result.0.iter()).enumerate() {
        if (l - r).abs() > EPSILON {
            panic!(
                "assertion failed at index {}: `(left ≈ right)`\n  left[{}]: `{:?}`,\n right[{}]: `{:?}`,\n epsilon: `{:?}`",
                i, i, l, i, r, EPSILON
            )
        }
    }
}
#[test]
fn perceptron_test_4() {
    let feature_matrix = vec![vec![1., 2.], vec![-1., 0.]];
    let labels = [1., 1.];
    let t = 2;

    let exp_result = (vec![0., 2.], 2.);
    let result = perceptron(&feature_matrix, &labels, t);

    assert!(
        (result.1 - exp_result.1).abs() < EPSILON,
        "{} is not approximately equal to {}",
        result.1,
        exp_result.1
    );
    for (i, (&l, &r)) in result.0.iter().zip(exp_result.0.iter()).enumerate() {
        if (l - r).abs() > EPSILON {
            panic!(
                "assertion failed at index {}: `(left ≈ right)`\n  left[{}]: `{:?}`,\n right[{}]: `{:?}`,\n epsilon: `{:?}`",
                i, i, l, i, r, EPSILON
            )
        }
    }
}