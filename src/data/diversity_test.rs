//! Comprehensive tests for the adaptive diverse target selection algorithm
//!
//! These tests validate the improved selection quality, diversity metrics,
//! and integration with the existing balance pipeline.

#[cfg(test)]
mod tests {
    use crate::data::balance::{BalanceConfig, SequenceBalancer, SequenceWithTargets};
    use crate::data::diversity::{DiversityConfig, DiversitySelector};
    use crate::targets::TargetType;
    use ndarray::Array2;
    use std::collections::HashMap;

    /// Create test sequences with varying characteristics for diversity testing
    fn create_test_sequences_with_diversity(count: usize) -> Vec<SequenceWithTargets> {
        let mut sequences = Vec::new();

        for i in 0..count {
            // Create sequences with different statistical properties
            let base_value = (i as f64) * 0.1;
            let volatility = 0.01 + (i % 5) as f64 * 0.005; // Different volatility levels
            let trend = if i % 3 == 0 {
                0.001
            } else if i % 3 == 1 {
                -0.001
            } else {
                0.0
            }; // Different trends

            // Generate sequence data with different characteristics
            let mut sequence_data = Vec::new();
            for j in 0..50 {
                // 50 time steps
                let time_factor = j as f64;
                let noise = (i * j) as f64 * 0.0001; // Deterministic "noise"

                // OHLCV data with different patterns
                let base_price = 100.0 + base_value + trend * time_factor + noise;
                let open = base_price;
                let high = base_price + volatility * (1.0 + (j % 3) as f64);
                let low = base_price - volatility * (1.0 + (j % 2) as f64);
                let close = base_price + trend;
                let volume = 1000.0 + (i % 10) as f64 * 100.0; // Different volume levels

                sequence_data.extend_from_slice(&[open, high, low, close, volume]);
            }

            let sequence_array = Array2::from_shape_vec((50, 5), sequence_data).unwrap();

            // Create targets for this sequence
            let mut targets = HashMap::new();
            let target_class = i as i32 % 5; // Distribute across 5 classes
            targets.insert((TargetType::PriceLevel, "1h".to_string()), target_class);
            targets.insert((TargetType::Direction, "1h".to_string()), target_class);
            targets.insert((TargetType::Volatility, "1h".to_string()), target_class);

            sequences.push(SequenceWithTargets {
                sequence_idx: i,
                start_idx: i * 10, // Different temporal positions
                end_idx: i * 10 + 50,
                sequence_data: sequence_array,
                targets,
            });
        }

        sequences
    }

    #[test]
    fn test_diversity_metrics_calculation() {
        let sequences = create_test_sequences_with_diversity(20);
        let diversity_selector = DiversitySelector::new(DiversityConfig::default());

        // Test diversity calculation for a sequence
        let class_indices: Vec<usize> = (0..20).collect();
        let metrics = diversity_selector
            .calculate_sequence_diversity(
                &sequences,
                0,
                &class_indices,
                TargetType::PriceLevel,
                "1h",
            )
            .unwrap();

        // Verify diversity metrics are in valid range
        assert!(metrics.feature_diversity >= 0.0 && metrics.feature_diversity <= 1.0);
        assert!(metrics.temporal_diversity >= 0.0 && metrics.temporal_diversity <= 1.0);
        assert!(metrics.market_diversity >= 0.0 && metrics.market_diversity <= 1.0);
        assert!(metrics.target_diversity >= 0.0 && metrics.target_diversity <= 1.0);
        assert!(metrics.composite_score >= 0.0 && metrics.composite_score <= 1.0);

        println!("Diversity metrics: {:?}", metrics);
    }

    #[test]
    fn test_diverse_selection_vs_chronological() {
        let sequences = create_test_sequences_with_diversity(100);
        let diversity_selector = DiversitySelector::new(DiversityConfig::default());

        // Create overloaded class (all sequences belong to class 0)
        let class_indices: Vec<usize> = (0..100).collect();
        let target_count = 20; // Select 20 out of 100

        // Test diverse selection
        let diverse_selection = diversity_selector
            .select_diverse_sequences(
                &sequences,
                &class_indices,
                target_count,
                TargetType::PriceLevel,
                "1h",
                &[], // No exclusions
            )
            .unwrap();

        // Verify selection count
        assert_eq!(diverse_selection.len(), target_count);

        // Verify diversity: selected sequences should be spread across different indices
        let mut sorted_selection = diverse_selection.clone();
        sorted_selection.sort();

        // Check that selection is not just the first N sequences (chronological bias)
        let chronological_selection: Vec<usize> = (0..target_count).collect();
        assert_ne!(
            sorted_selection, chronological_selection,
            "Diverse selection should not be chronological"
        );

        // Calculate diversity metrics for selected sequences
        let mut total_diversity = 0.0;
        for &idx in &diverse_selection {
            let metrics = diversity_selector
                .calculate_sequence_diversity(
                    &sequences,
                    idx,
                    &class_indices,
                    TargetType::PriceLevel,
                    "1h",
                )
                .unwrap();
            total_diversity += metrics.composite_score;
        }

        let avg_diversity = total_diversity / diverse_selection.len() as f64;
        println!(
            "Average diversity of selected sequences: {:.3}",
            avg_diversity
        );

        // Diverse selection should have reasonable diversity
        assert!(
            avg_diversity > 0.1,
            "Selected sequences should have meaningful diversity"
        );
    }

    #[test]
    fn test_balance_maintenance_with_diversity() {
        let sequences = create_test_sequences_with_diversity(100);
        let balancer = SequenceBalancer::new(BalanceConfig::default());

        // Test window-based balancing which uses the diversity selection internally
        let window_range = (0, 1000);
        let validation_indices = vec![5, 15, 25, 35, 45];

        let result = balancer.balance_sequences_for_window(
            &sequences,
            &validation_indices,
            window_range,
            TargetType::PriceLevel,
            "1h",
        );

        // Should succeed with diversity selection
        assert!(result.is_ok(), "Balance with diversity should work");

        let selection = result.unwrap();

        // Verify perfect balance
        assert!(!selection.selected_indices.is_empty());
        assert!(selection.sequences_per_class > 0);
        assert_eq!(selection.class_distribution.len(), 5); // 5 classes

        // Verify each class has exactly the same amount
        let expected_per_class = selection.sequences_per_class;
        for (class, count) in &selection.class_distribution {
            assert_eq!(
                *count, expected_per_class,
                "Class {} should have exactly {} sequences",
                class, expected_per_class
            );
        }

        println!(
            "Balance test passed: {} sequences selected, {} per class",
            selection.selected_indices.len(),
            expected_per_class
        );
    }

    #[test]
    fn test_temporal_diversity_calculation() {
        let sequences = create_test_sequences_with_diversity(50);
        let diversity_selector = DiversitySelector::new(DiversityConfig::default());

        // Test temporal diversity for sequences at different time positions
        let class_indices: Vec<usize> = (0..50).collect();

        // Sequence at beginning of time series
        let early_metrics = diversity_selector
            .calculate_sequence_diversity(
                &sequences,
                0,
                &class_indices,
                TargetType::PriceLevel,
                "1h",
            )
            .unwrap();

        // Sequence in middle of time series
        let middle_metrics = diversity_selector
            .calculate_sequence_diversity(
                &sequences,
                25,
                &class_indices,
                TargetType::PriceLevel,
                "1h",
            )
            .unwrap();

        // Sequence at end of time series
        let late_metrics = diversity_selector
            .calculate_sequence_diversity(
                &sequences,
                49,
                &class_indices,
                TargetType::PriceLevel,
                "1h",
            )
            .unwrap();

        println!(
            "Temporal diversity - Early: {:.3}, Middle: {:.3}, Late: {:.3}",
            early_metrics.temporal_diversity,
            middle_metrics.temporal_diversity,
            late_metrics.temporal_diversity
        );

        // Sequences at edges should have higher temporal diversity than middle
        assert!(early_metrics.temporal_diversity > middle_metrics.temporal_diversity * 0.8);
        assert!(late_metrics.temporal_diversity > middle_metrics.temporal_diversity * 0.8);
    }

    #[test]
    fn test_market_condition_diversity() {
        let sequences = create_test_sequences_with_diversity(30);
        let diversity_selector = DiversitySelector::new(DiversityConfig::default());

        // Test market condition extraction
        let market_conditions = diversity_selector
            .extract_market_conditions(&sequences[0].sequence_data)
            .unwrap();

        // Should extract 4 market condition features
        assert_eq!(market_conditions.len(), 4);

        // All values should be finite
        for &value in market_conditions.iter() {
            assert!(
                value.is_finite(),
                "Market condition values should be finite"
            );
        }

        println!("Market conditions: {:?}", market_conditions.to_vec());

        // Test market condition diversity calculation
        let class_indices: Vec<usize> = (0..30).collect();
        let metrics = diversity_selector
            .calculate_sequence_diversity(
                &sequences,
                0,
                &class_indices,
                TargetType::PriceLevel,
                "1h",
            )
            .unwrap();

        assert!(metrics.market_diversity >= 0.0 && metrics.market_diversity <= 1.0);
        println!("Market diversity: {:.3}", metrics.market_diversity);
    }

    #[test]
    fn test_diversity_config_customization() {
        let custom_config = DiversityConfig {
            feature_weight: 0.5,
            temporal_weight: 0.3,
            market_weight: 0.15,
            target_weight: 0.05,
            min_diversity_threshold: 0.2,
            max_similarity_threshold: 0.7,
        };

        let diversity_selector = DiversitySelector::new(custom_config);
        let sequences = create_test_sequences_with_diversity(20);

        let class_indices: Vec<usize> = (0..20).collect();
        let metrics = diversity_selector
            .calculate_sequence_diversity(
                &sequences,
                0,
                &class_indices,
                TargetType::PriceLevel,
                "1h",
            )
            .unwrap();

        // Verify composite score uses custom weights
        let expected_composite = 0.5 * metrics.feature_diversity
            + 0.3 * metrics.temporal_diversity
            + 0.15 * metrics.market_diversity
            + 0.05 * metrics.target_diversity;

        assert!(
            (metrics.composite_score - expected_composite).abs() < 0.001,
            "Composite score should use custom weights"
        );

        println!(
            "Custom weighted composite score: {:.3}",
            metrics.composite_score
        );
    }

    #[test]
    fn test_selection_quality_improvement() {
        let sequences = create_test_sequences_with_diversity(200);
        let diversity_selector = DiversitySelector::new(DiversityConfig::default());

        // Create large overloaded class
        let class_indices: Vec<usize> = (0..200).collect();
        let target_count = 40;

        // Test diverse selection
        let diverse_selection = diversity_selector
            .select_diverse_sequences(
                &sequences,
                &class_indices,
                target_count,
                TargetType::PriceLevel,
                "1h",
                &[],
            )
            .unwrap();

        // Calculate average pairwise diversity of selected sequences
        let mut total_pairwise_diversity = 0.0;
        let mut pair_count = 0;

        for i in 0..diverse_selection.len() {
            for j in i + 1..diverse_selection.len() {
                let idx1 = diverse_selection[i];
                let idx2 = diverse_selection[j];

                // Calculate statistical distance between sequences
                let stats1 = diversity_selector
                    .calculate_sequence_statistics(&sequences[idx1].sequence_data)
                    .unwrap();
                let stats2 = diversity_selector
                    .calculate_sequence_statistics(&sequences[idx2].sequence_data)
                    .unwrap();

                let distance = diversity_selector.euclidean_distance(&stats1, &stats2);
                total_pairwise_diversity += distance;
                pair_count += 1;
            }
        }

        let avg_pairwise_diversity = total_pairwise_diversity / pair_count as f64;

        // Compare with chronological selection (first N sequences)
        let chronological_selection: Vec<usize> = (0..target_count).collect();
        let mut chronological_pairwise_diversity = 0.0;
        let mut chronological_pair_count = 0;

        for i in 0..chronological_selection.len() {
            for j in i + 1..chronological_selection.len() {
                let idx1 = chronological_selection[i];
                let idx2 = chronological_selection[j];

                let stats1 = diversity_selector
                    .calculate_sequence_statistics(&sequences[idx1].sequence_data)
                    .unwrap();
                let stats2 = diversity_selector
                    .calculate_sequence_statistics(&sequences[idx2].sequence_data)
                    .unwrap();

                let distance = diversity_selector.euclidean_distance(&stats1, &stats2);
                chronological_pairwise_diversity += distance;
                chronological_pair_count += 1;
            }
        }

        let avg_chronological_diversity =
            chronological_pairwise_diversity / chronological_pair_count as f64;

        println!(
            "Diverse selection avg pairwise diversity: {:.3}",
            avg_pairwise_diversity
        );
        println!(
            "Chronological selection avg pairwise diversity: {:.3}",
            avg_chronological_diversity
        );

        // Diverse selection should have higher pairwise diversity
        assert!(
            avg_pairwise_diversity > avg_chronological_diversity * 1.1,
            "Diverse selection should have at least 10% higher pairwise diversity"
        );
    }

    #[test]
    fn test_integration_with_existing_pipeline() {
        // Test that the new diversity selection integrates seamlessly with existing balance pipeline
        let sequences = create_test_sequences_with_diversity(100);
        let balancer = SequenceBalancer::new(BalanceConfig::default());

        // Simulate window-based balancing (existing functionality)
        let window_range = (0, 1000);
        let validation_indices = vec![5, 15, 25, 35, 45]; // Some validation sequences

        let result = balancer.balance_sequences_for_window(
            &sequences,
            &validation_indices,
            window_range,
            TargetType::PriceLevel,
            "1h",
        );

        // Should succeed with diversity selection
        assert!(
            result.is_ok(),
            "Integration with existing pipeline should work"
        );

        let selection = result.unwrap();

        // Verify balance properties are maintained
        assert!(!selection.selected_indices.is_empty());
        assert!(selection.sequences_per_class > 0);
        assert_eq!(selection.class_distribution.len(), 5); // 5 classes

        // Verify no validation sequences were selected
        for &selected_idx in &selection.selected_indices {
            assert!(
                !validation_indices.contains(&selected_idx),
                "Validation sequences should be excluded from training selection"
            );
        }

        println!(
            "Integration test passed: {} sequences selected with perfect balance",
            selection.selected_indices.len()
        );
    }
}
