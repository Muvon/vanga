//! Comprehensive tests for ReduceOnPlateau scheduler
//!
//! Tests the scheduler's behavior with various loss patterns to ensure
//! learning rate is reduced correctly when validation loss plateaus.

use crate::model::lstm::training::ReduceOnPlateauScheduler;

#[test]
fn test_reduce_on_plateau_basic() {
    let initial_lr = 0.001;
    let patience = 3;
    let factor = 0.5;

    let mut scheduler = ReduceOnPlateauScheduler::new(initial_lr, patience, factor);

    // First step: record initial loss
    let (lr, reduced) = scheduler.step(1.0);
    assert_eq!(lr, initial_lr, "First step should keep initial LR");
    assert!(!reduced, "First step should not reduce");

    // Improvement: loss decreases
    let (lr, reduced) = scheduler.step(0.9);
    assert_eq!(lr, initial_lr, "Improvement should keep LR");
    assert!(!reduced, "Improvement should not reduce");

    // No improvement: loss stays same
    let (lr, reduced) = scheduler.step(0.9);
    assert_eq!(
        lr, initial_lr,
        "First plateau should keep LR (patience 1/3)"
    );
    assert!(!reduced, "Should not reduce yet");

    // No improvement: loss increases
    let (lr, reduced) = scheduler.step(1.0);
    assert_eq!(
        lr, initial_lr,
        "Second plateau should keep LR (patience 2/3)"
    );
    assert!(!reduced, "Should not reduce yet");

    // No improvement: loss stays high
    let (lr, reduced) = scheduler.step(1.0);
    assert_eq!(
        lr, initial_lr,
        "Third plateau should keep LR (patience 3/3)"
    );
    assert!(!reduced, "Should not reduce yet");

    // Patience exceeded: should reduce now
    let (lr, reduced) = scheduler.step(1.0);
    assert_eq!(
        lr,
        initial_lr * factor,
        "Should reduce LR after patience exceeded"
    );
    assert!(reduced, "Should indicate reduction occurred");
}

#[test]
fn test_reduce_on_plateau_multiple_reductions() {
    let initial_lr = 0.001;
    let patience = 2;
    let factor = 0.5;

    let mut scheduler = ReduceOnPlateauScheduler::new(initial_lr, patience, factor);

    // First step
    scheduler.step(1.0);

    // Trigger first reduction
    scheduler.step(1.0); // patience 1/2
    scheduler.step(1.0); // patience 2/2
    let (lr1, reduced1) = scheduler.step(1.0); // should reduce
    assert_eq!(lr1, initial_lr * factor, "First reduction");
    assert!(reduced1, "Should reduce");

    // Continue with no improvement - trigger second reduction
    scheduler.step(1.0); // patience 1/2
    scheduler.step(1.0); // patience 2/2
    let (lr2, reduced2) = scheduler.step(1.0); // should reduce again
    assert_eq!(lr2, initial_lr * factor * factor, "Second reduction");
    assert!(reduced2, "Should reduce again");
}

#[test]
fn test_reduce_on_plateau_reset_on_improvement() {
    let initial_lr = 0.001;
    let patience = 3;
    let factor = 0.5;

    let mut scheduler = ReduceOnPlateauScheduler::new(initial_lr, patience, factor);

    // First step
    scheduler.step(1.0);

    // Build up patience
    scheduler.step(1.0); // patience 1/3
    scheduler.step(1.0); // patience 2/3

    // Improvement resets patience
    let (lr, reduced) = scheduler.step(0.8);
    assert_eq!(lr, initial_lr, "Improvement should keep LR");
    assert!(!reduced, "Improvement should not reduce");

    // Now need full patience again
    scheduler.step(0.8); // patience 1/3
    scheduler.step(0.8); // patience 2/3
    scheduler.step(0.8); // patience 3/3
    let (lr, reduced) = scheduler.step(0.8); // should reduce
    assert_eq!(lr, initial_lr * factor, "Should reduce after full patience");
    assert!(reduced, "Should reduce");
}

#[test]
fn test_reduce_on_plateau_increasing_loss() {
    let initial_lr = 0.001;
    let patience = 2;
    let factor = 0.5;

    let mut scheduler = ReduceOnPlateauScheduler::new(initial_lr, patience, factor);

    // First step
    scheduler.step(1.0);

    // Loss increases (no improvement)
    scheduler.step(1.1); // patience 1/2
    scheduler.step(1.2); // patience 2/2
    let (lr, reduced) = scheduler.step(1.3); // should reduce

    assert_eq!(lr, initial_lr * factor, "Should reduce when loss increases");
    assert!(reduced, "Should reduce");
}

#[test]
fn test_reduce_on_plateau_exact_patience() {
    let initial_lr = 0.001;
    let patience = 5;
    let factor = 0.1;

    let mut scheduler = ReduceOnPlateauScheduler::new(initial_lr, patience, factor);

    // First step
    scheduler.step(1.0);

    // Exactly patience epochs with no improvement
    for i in 1..=patience {
        let (lr, reduced) = scheduler.step(1.0);
        if i < patience {
            assert_eq!(
                lr, initial_lr,
                "Should not reduce before patience (epoch {})",
                i
            );
            assert!(!reduced, "Should not reduce before patience (epoch {})", i);
        } else {
            assert_eq!(
                lr, initial_lr,
                "Should not reduce AT patience (epoch {})",
                i
            );
            assert!(!reduced, "Should not reduce AT patience (epoch {})", i);
        }
    }

    // One more epoch should trigger reduction
    let (lr, reduced) = scheduler.step(1.0);
    assert_eq!(
        lr,
        initial_lr * factor,
        "Should reduce AFTER patience exceeded"
    );
    assert!(reduced, "Should reduce AFTER patience exceeded");
}

#[test]
fn test_reduce_on_plateau_current_lr() {
    let initial_lr = 0.001;
    let patience = 2;
    let factor = 0.5;

    let mut scheduler = ReduceOnPlateauScheduler::new(initial_lr, patience, factor);

    assert_eq!(
        scheduler.current_lr(),
        initial_lr,
        "Initial LR should be correct"
    );

    scheduler.step(1.0);
    assert_eq!(
        scheduler.current_lr(),
        initial_lr,
        "LR should not change after first step"
    );

    // Trigger reduction
    scheduler.step(1.0);
    scheduler.step(1.0);
    scheduler.step(1.0);

    assert_eq!(
        scheduler.current_lr(),
        initial_lr * factor,
        "LR should be reduced"
    );
}

#[test]
fn test_reduce_on_plateau_small_improvements() {
    let initial_lr = 0.001;
    let patience = 2;
    let factor = 0.5;

    let mut scheduler = ReduceOnPlateauScheduler::new(initial_lr, patience, factor);

    // First step
    scheduler.step(1.0);

    // Very small improvement (still counts as improvement)
    let (lr, reduced) = scheduler.step(0.9999);
    assert_eq!(lr, initial_lr, "Small improvement should count");
    assert!(!reduced, "Small improvement should reset patience");

    // Now plateau
    scheduler.step(0.9999); // patience 1/2
    scheduler.step(0.9999); // patience 2/2
    let (lr, reduced) = scheduler.step(0.9999); // should reduce

    assert_eq!(lr, initial_lr * factor, "Should reduce after plateau");
    assert!(reduced, "Should reduce");
}

#[test]
fn test_reduce_on_plateau_alternating_pattern() {
    let initial_lr = 0.001;
    let patience = 3;
    let factor = 0.5;

    let mut scheduler = ReduceOnPlateauScheduler::new(initial_lr, patience, factor);

    // First step
    scheduler.step(1.0);

    // Alternating: improve, plateau, improve, plateau
    scheduler.step(0.9); // improve - reset
    scheduler.step(0.9); // plateau - patience 1/3
    scheduler.step(0.8); // improve - reset
    scheduler.step(0.8); // plateau - patience 1/3
    scheduler.step(0.8); // plateau - patience 2/3
    scheduler.step(0.8); // plateau - patience 3/3
    let (lr, reduced) = scheduler.step(0.8); // should reduce

    assert_eq!(
        lr,
        initial_lr * factor,
        "Should reduce after sustained plateau"
    );
    assert!(reduced, "Should reduce");
}

#[test]
fn test_reduce_on_plateau_zero_patience() {
    let initial_lr = 0.001;
    let patience = 0; // Reduce immediately on no improvement
    let factor = 0.5;

    let mut scheduler = ReduceOnPlateauScheduler::new(initial_lr, patience, factor);

    // First step
    scheduler.step(1.0);

    // No improvement should reduce immediately
    let (lr, reduced) = scheduler.step(1.0);
    assert_eq!(
        lr,
        initial_lr * factor,
        "Should reduce immediately with patience=0"
    );
    assert!(reduced, "Should reduce immediately");
}

#[test]
fn test_reduce_on_plateau_large_factor() {
    let initial_lr = 0.001;
    let patience = 2;
    let factor = 0.1; // Aggressive reduction

    let mut scheduler = ReduceOnPlateauScheduler::new(initial_lr, patience, factor);

    scheduler.step(1.0);
    scheduler.step(1.0);
    scheduler.step(1.0);
    let (lr, reduced) = scheduler.step(1.0);

    assert_eq!(
        lr,
        initial_lr * factor,
        "Should apply large factor correctly"
    );
    assert!(reduced, "Should reduce");

    // Verify it can reduce multiple times
    scheduler.step(1.0);
    scheduler.step(1.0);
    let (lr2, reduced2) = scheduler.step(1.0);

    assert_eq!(
        lr2,
        initial_lr * factor * factor,
        "Should apply factor multiple times"
    );
    assert!(reduced2, "Should reduce again");
}
