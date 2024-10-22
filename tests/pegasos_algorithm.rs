use automatic_review_analyzer::{pegasos, pegasos_single_step_update, DType};

const EPSILON: DType = 1e-6;

#[test]
fn pegasos_single_test_1() {
    let feature_vector = [1., 2.];
    let label = 1.;
    let theta = [-1., 1.];
    let theta_0 = -1.5;
    let lambda = 0.2;
    let eta = 0.1;
    let exp_result = ([-0.88, 1.18], -1.4);
    let result = pegasos_single_step_update(&feature_vector, label, lambda, eta, &theta, theta_0);
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
fn pegasos_single_test_2() {
    let feature_vector = [1., 1.];
    let label = 1.;
    let theta = [-1., 1.];
    let theta_0 = 1.;
    let lambda = 0.2;
    let eta = 0.1;
    let exp_result = ([-0.88, 1.08], 1.1);
    let result = pegasos_single_step_update(&feature_vector, label, lambda, eta, &theta, theta_0);
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
fn pegasos_single_test_3() {
    let feature_vector = [1., 2.];
    let label = 1.;
    let theta = [-1., 1.];
    let theta_0 = -2.;
    let lambda = 0.2;
    let eta = 0.1;
    let exp_result = ([-0.88, 1.18], -1.9);
    let result = pegasos_single_step_update(&feature_vector, label, lambda, eta, &theta, theta_0);
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
fn pegasos_single_label_gt_1() {
    let feature_vector = [
        -0.37358111,
        -0.28667788,
        0.00343874,
        -0.25983078,
        -0.34535844,
        -0.37319494,
        0.45127405,
        0.35516106,
        0.40528442,
        -0.09580812,
    ];
    let label = -1.;
    let lambda = 0.24089844712406283;
    let eta = 0.5168857402198094;
    let theta = [
        -0.41488326,
        -0.10390025,
        0.18732405,
        0.43495735,
        -0.30999745,
        -0.25767909,
        -0.48378453,
        0.39740575,
        -0.29970124,
        -0.08284869,
    ];
    let theta_0 = -2.0204571095479325;
    let exp_result = (
        [
            -0.3632233, -0.0909629, 0.1639990, 0.3807978, -0.2713975, -0.2255937, -0.4235451,
            0.3479220, -0.2623834, -0.0725326,
        ],
        -2.0204571,
    );
    let result = pegasos_single_step_update(&feature_vector, label, lambda, eta, &theta, theta_0);
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
fn pegasos_single_label_eq_1() {
    let feature_vector = [
        -0.20632084,
        0.11238741,
        0.48331221,
        -0.02966988,
        0.42802245,
        0.25940516,
        0.09517402,
        -0.03264273,
        0.08849776,
        -0.43854802,
    ];
    let label = -1.;
    let lambda = 0.04337518486336811;
    let eta = 0.0695201052095793;
    let theta = [
        -0.21201162,
        0.36154625,
        0.05500601,
        0.059696,
        0.16452909,
        -0.03122524,
        -0.42386493,
        -0.24502413,
        0.24110044,
        -0.37880782,
    ];
    let theta_0 = -1.3266313105176373;
    let exp_result = (
        [
            -0.1970289, 0.3526428, 0.0212402, 0.0615786, 0.1342768, -0.0491650, -0.4292033,
            -0.2420159, 0.2342210, -0.3471776,
        ],
        -1.3961514,
    );
    let result = pegasos_single_step_update(&feature_vector, label, lambda, eta, &theta, theta_0);
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
fn pegasos_single_label_lt_1() {
    let feature_vector = [
        -0.29650835,
        0.0896379,
        0.46686835,
        -0.24049415,
        0.38694984,
        -0.35871562,
        -0.48093296,
        -0.13035812,
        -0.34276856,
        0.1912422,
    ];
    let label = 1.;
    let lambda = 0.32416069337508546;
    let eta = 0.5364086696380413;
    let theta = [
        0.0059779,
        0.44160801,
        -0.27448235,
        0.23280701,
        0.06278054,
        0.01822841,
        0.06544701,
        -0.29923289,
        -0.36164645,
        0.45184443,
    ];
    let theta_0 = 0.3378773458758522;
    let exp_result = (
        [
            -0.1541112, 0.4129026, 0.0236776, 0.0633228, 0.2594273, -0.1773594, -0.2039097,
            -0.3171267, -0.4826265, 0.4758605,
        ],
        0.8742860,
    );
    let result = pegasos_single_step_update(&feature_vector, label, lambda, eta, &theta, theta_0);
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
fn pegasos_single_label_le_0() {
    let feature_vector = [
        -0.45151152,
        -0.11974364,
        0.48888076,
        -0.28406762,
        0.04288069,
        -0.34642206,
        -0.14262421,
        0.18286305,
        0.02447251,
        0.08378412,
    ];
    let label = -1.;
    let lambda = 0.02939510585980909;
    let eta = 0.667267169844567;
    let theta = [
        -0.19667858,
        0.48563424,
        0.09320106,
        -0.2464211,
        0.06995773,
        0.32889011,
        0.132287,
        -0.4546192,
        0.12421149,
        0.13986867,
    ];
    let theta_0 = 0.7520686178896695;
    let exp_result = (
        [
            0.1084580, 0.5560098, -0.2348411, -0.0520387, 0.0399727, 0.5535952, 0.2248607,
            -0.5677206, 0.1054455, 0.0812188,
        ],
        0.0848014,
    );
    let result = pegasos_single_step_update(&feature_vector, label, lambda, eta, &theta, theta_0);
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
fn pegasos_single_theta_eq_0() {
    let feature_vector = [
        -0.11112083,
        0.12834158,
        -0.37218221,
        -0.03296448,
        0.36785532,
        0.21392332,
        0.22816974,
        -0.48573043,
        0.13555109,
        0.18751242,
    ];
    let label = 1.;
    let lambda = 0.021908374483356585;
    let eta = 0.8978079036720401;
    let theta = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.];
    let theta_0 = 0.;
    let exp_result = (
        [
            -0.0997652, 0.1152261, -0.3341481, -0.0295958, 0.3302634, 0.1920620, 0.2048526,
            -0.4360926, 0.1216988, 0.1683501,
        ],
        0.8978079,
    );
    let result = pegasos_single_step_update(&feature_vector, label, lambda, eta, &theta, theta_0);
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
fn pegasos_test_1() {
    let feature_matrix = vec![vec![1., 2.]];
    let labels = [1.];
    let t = 1;
    let lambda = 0.2;

    let exp_result = ([1., 2.], 1.);
    let result = pegasos(&feature_matrix, &labels, t, lambda);
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
fn pegasos_test_2() {
    let feature_matrix = vec![vec![1., 1.], vec![1., 1.]];
    let labels = [1., 1.];
    let t = 1;
    let lambda = 1.;

    let exp_result = (
        [1. - 1. / DType::sqrt(2f32), 1. - 1. / DType::sqrt(2f32)],
        1.,
    );
    let result = pegasos(&feature_matrix, &labels, t, lambda);
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
fn pegasos_high_dimension() {
    let feature_matrix = vec![
        vec![
            0.1837462,
            0.29989789,
            -0.35889786,
            -0.30780561,
            -0.44230703,
            -0.03043835,
            0.21370063,
            0.33344998,
            -0.40850817,
            -0.13105809,
        ],
        vec![
            0.08254096,
            0.06012654,
            0.19821234,
            0.40958367,
            0.07155838,
            -0.49830717,
            0.09098162,
            0.19062183,
            -0.27312663,
            0.39060785,
        ],
        vec![
            -0.20112519,
            -0.00593087,
            0.05738862,
            0.16811148,
            -0.10466314,
            -0.21348009,
            0.45806193,
            -0.27659307,
            0.2901038,
            -0.29736505,
        ],
        vec![
            -0.14703536,
            -0.45573697,
            -0.47563745,
            -0.08546162,
            -0.08562345,
            0.07636098,
            -0.42087389,
            -0.16322197,
            -0.02759763,
            0.0297091,
        ],
        vec![
            -0.18082261,
            0.28644149,
            -0.47549449,
            -0.3049562,
            0.13967768,
            0.34904474,
            0.20627692,
            0.28407868,
            0.21849356,
            -0.01642202,
        ],
    ];
    let labels = [-1., -1., -1., 1., -1.];
    let t = 10;
    let lambda = 0.1456692551041303;

    let exp_result = (
        [
            -0.0850387, -0.7286435, -0.3440130, -0.0560494, -0.0260993, 0.1446894, -0.8172203,
            -0.3200453, -0.0729161, 0.1008662,
        ],
        1.,
    );
    let result = pegasos(&feature_matrix, &labels, t, lambda);
    // assert!(
    //     (result.1 - exp_result.1).abs() < EPSILON,
    //     "{} is not approximately equal to {}",
    //     result.1,
    //     exp_result.1
    // );
    for (i, (&l, &r)) in result.0.iter().zip(exp_result.0.iter()).enumerate() {
        if (l - r).abs() > EPSILON {
            panic!(
                "assertion failed at index {}: `(left ≈ right)`\n  left[{}]: `{:?}`,\n right[{}]: `{:?}`,\n epsilon: `{:?}`",
                i, i, l, i, r, EPSILON
            )
        }
    }
}
