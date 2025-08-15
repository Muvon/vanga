use crate::model::ordinal_smartcore::*;

#[test]
fn test_ordinal_penalty_matrix_symmetry() {
    // Test that penalties are asymmetric (trading-aware)
    assert_ne!(
        ORDINAL_PENALTY_MATRIX[0][4], // VeryDown -> VeryUp (worst)
        ORDINAL_PENALTY_MATRIX[4][0]  // VeryUp -> VeryDown (also bad but different)
    );

    // Test that same class has zero penalty
    for i in 0..5 {
        assert_eq!(ORDINAL_PENALTY_MATRIX[i][i], 0.0);
    }

    // Test that wrong direction is worse than wrong magnitude
    assert!(
        ORDINAL_PENALTY_MATRIX[0][3] > ORDINAL_PENALTY_MATRIX[0][1], // Across middle vs same side
        "Wrong direction should have higher penalty than wrong magnitude"
    );
}
