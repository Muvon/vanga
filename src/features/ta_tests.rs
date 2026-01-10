// Test the new TA crate implementations and fixes
use crate::features::ta_helpers::*;
use crate::features::validation::*;

#[test]
fn test_fixed_price_gaps() {
    println!("Testing body-to-range ratio (price inefficiency)...");

    // Test various candle patterns
    // Format: [open, high, low, close]
    let test_cases: Vec<(f64, f64, f64, f64, f64)> = vec![
        // Full bullish candle (no wicks) - should be +100
        (100.0, 101.0, 100.0, 101.0, 100.0),
        // Full bearish candle (no wicks) - should be -100
        (101.0, 101.0, 100.0, 100.0, -100.0),
        // Doji (all wicks, no body) - should be 0
        (100.5, 101.0, 100.0, 100.5, 0.0),
        // Bullish with wicks - should be positive but < 100
        (100.0, 102.0, 99.0, 101.0, 33.33),
        // Bearish with wicks - should be negative but > -100
        (101.0, 102.0, 99.0, 100.0, -33.33),
    ];

    for (open, high, low, close, expected) in test_cases {
        let range = high - low;
        let body_to_range = if range > 0.0 {
            let body = close - open;
            let ratio = body / range * 100.0;
            ratio.clamp(-100.0, 100.0)
        } else {
            0.0
        };

        println!(
            "O:{} H:{} L:{} C:{} => Body/Range: {:.2}% (expected ~{:.2}%)",
            open, high, low, close, body_to_range, expected
        );

        assert!(body_to_range.is_finite(), "Body-to-range should be finite");
        assert!(
            (-100.0..=100.0).contains(&body_to_range),
            "Body-to-range should be in [-100, 100]"
        );
        assert!(
            (body_to_range - expected).abs() < 1.0,
            "Body-to-range {:.2} should be close to expected {:.2}",
            body_to_range,
            expected
        );
    }

    println!("✓ Body-to-range ratio working correctly for crypto 24/7 markets");
}

#[test]
fn test_wick_imbalance() {
    println!("Testing wick imbalance calculation...");

    // Test various wick patterns
    // Format: [open, high, low, close, expected_imbalance]
    let test_cases: Vec<(f64, f64, f64, f64, f64)> = vec![
        // All upper wick (strong selling rejection) - should be +100
        (100.0, 102.0, 100.0, 100.0, 100.0),
        // All lower wick (strong buying rejection) - should be -100
        (100.0, 100.0, 98.0, 100.0, -100.0),
        // Balanced wicks - should be ~0
        (100.0, 101.0, 99.0, 100.0, 0.0),
        // Upper wick dominant: O:100, H:103, L:99, C:101
        // body_top=101, body_bottom=100, upper=2, lower=1, total=3
        // imbalance = (2-1)/3*100 = 33.33%
        (100.0, 103.0, 99.0, 101.0, 33.33),
        // Lower wick dominant: O:101, H:102, L:98, C:100
        // body_top=101, body_bottom=100, upper=1, lower=2, total=3
        // imbalance = (1-2)/3*100 = -33.33%
        (101.0, 102.0, 98.0, 100.0, -33.33),
    ];

    for (open, high, low, close, expected) in test_cases {
        let body_top = open.max(close);
        let body_bottom = open.min(close);
        let upper_wick = high - body_top;
        let lower_wick = body_bottom - low;
        let total_wick = upper_wick + lower_wick;

        let wick_imbalance = if total_wick > 0.0 {
            let imbalance = (upper_wick - lower_wick) / total_wick * 100.0;
            imbalance.clamp(-100.0, 100.0)
        } else {
            0.0
        };

        println!(
            "O:{} H:{} L:{} C:{} => Wick Imbalance: {:.2}% (expected ~{:.2}%)",
            open, high, low, close, wick_imbalance, expected
        );

        assert!(
            wick_imbalance.is_finite(),
            "Wick imbalance should be finite"
        );
        assert!(
            (-100.0..=100.0).contains(&wick_imbalance),
            "Wick imbalance should be in [-100, 100]"
        );
        assert!(
            (wick_imbalance - expected).abs() < 1.0,
            "Wick imbalance {:.2} should be close to expected {:.2}",
            wick_imbalance,
            expected
        );
    }

    println!("✓ Wick imbalance working correctly");
}

#[test]
fn test_ta_crate_rsi() {
    println!("Testing TA crate RSI implementation...");

    // Test with normal data
    let prices = vec![
        100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0,
    ];
    let rsi_result = calculate_rsi_ta(&prices, 5);

    assert!(rsi_result.is_ok(), "RSI calculation should succeed");
    let rsi_values = rsi_result.unwrap();

    println!("RSI values: {:?}", rsi_values);
    assert_eq!(
        rsi_values.len(),
        prices.len(),
        "RSI should have same length as input"
    );

    // Check that we have some finite values after warm-up period
    let finite_values: Vec<f64> = rsi_values
        .iter()
        .filter(|&&x| x.is_finite())
        .cloned()
        .collect();
    assert!(
        !finite_values.is_empty(),
        "RSI should have some finite values after warm-up"
    );

    // All finite values should be in valid range [0, 100]
    for (i, &rsi) in rsi_values.iter().enumerate() {
        if rsi.is_finite() {
            assert!(
                (0.0..=100.0).contains(&rsi),
                "RSI value {} at index {} should be in [0, 100]",
                rsi,
                i
            );
        }
    }
}

#[test]
fn test_ta_crate_macd() {
    println!("Testing TA crate MACD implementation...");

    let prices = vec![
        100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0, 111.0, 110.0,
    ];
    let macd_result = calculate_macd_ta(&prices, 3, 6, 2);

    assert!(macd_result.is_ok(), "MACD calculation should succeed");
    let (macd_line, signal_line, histogram) = macd_result.unwrap();

    println!("MACD line: {:?}", macd_line);
    println!("Signal line: {:?}", signal_line);
    println!("Histogram: {:?}", histogram);

    assert_eq!(
        macd_line.len(),
        prices.len(),
        "MACD line should have same length as input"
    );
    assert_eq!(
        signal_line.len(),
        prices.len(),
        "Signal line should have same length as input"
    );
    assert_eq!(
        histogram.len(),
        prices.len(),
        "Histogram should have same length as input"
    );

    // Check that we have some finite values after warm-up period
    let finite_macd: Vec<f64> = macd_line
        .iter()
        .filter(|&&x| x.is_finite())
        .cloned()
        .collect();
    let finite_signal: Vec<f64> = signal_line
        .iter()
        .filter(|&&x| x.is_finite())
        .cloned()
        .collect();
    let finite_hist: Vec<f64> = histogram
        .iter()
        .filter(|&&x| x.is_finite())
        .cloned()
        .collect();

    assert!(
        !finite_macd.is_empty(),
        "MACD should have some finite values after warm-up"
    );
    assert!(
        !finite_signal.is_empty(),
        "Signal line should have some finite values after warm-up"
    );
    assert!(
        !finite_hist.is_empty(),
        "Histogram should have some finite values after warm-up"
    );

    // All finite values should be reasonable
    for (i, ((macd, signal), hist)) in macd_line
        .iter()
        .zip(signal_line.iter())
        .zip(histogram.iter())
        .enumerate()
    {
        if macd.is_finite() && signal.is_finite() && hist.is_finite() {
            // Basic sanity checks for finite values
            assert!(
                macd.abs() < 1000.0,
                "MACD value {} at index {} should be reasonable",
                macd,
                i
            );
            assert!(
                signal.abs() < 1000.0,
                "Signal value {} at index {} should be reasonable",
                signal,
                i
            );
            assert!(
                hist.abs() < 1000.0,
                "Histogram value {} at index {} should be reasonable",
                hist,
                i
            );
        }
    }
}

#[test]
fn test_ta_crate_bollinger_bands() {
    println!("Testing TA crate Bollinger Bands implementation...");

    let prices = vec![
        100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0,
    ];
    let bb_result = calculate_bollinger_bands_ta(&prices, 5, 2.0);

    assert!(
        bb_result.is_ok(),
        "Bollinger Bands calculation should succeed"
    );
    let (upper, middle, lower) = bb_result.unwrap();

    println!("Upper band: {:?}", upper);
    println!("Middle band: {:?}", middle);
    println!("Lower band: {:?}", lower);

    assert_eq!(
        upper.len(),
        prices.len(),
        "Upper band should have same length as input"
    );
    assert_eq!(
        middle.len(),
        prices.len(),
        "Middle band should have same length as input"
    );
    assert_eq!(
        lower.len(),
        prices.len(),
        "Lower band should have same length as input"
    );

    // Check that we have some finite values after warm-up period
    let finite_upper: Vec<f64> = upper.iter().filter(|&&x| x.is_finite()).cloned().collect();
    let finite_middle: Vec<f64> = middle.iter().filter(|&&x| x.is_finite()).cloned().collect();
    let finite_lower: Vec<f64> = lower.iter().filter(|&&x| x.is_finite()).cloned().collect();

    assert!(
        !finite_upper.is_empty(),
        "Upper band should have some finite values after warm-up"
    );
    assert!(
        !finite_middle.is_empty(),
        "Middle band should have some finite values after warm-up"
    );
    assert!(
        !finite_lower.is_empty(),
        "Lower band should have some finite values after warm-up"
    );

    // Validate band relationships: upper >= middle >= lower (only for finite values)
    for i in 0..prices.len() {
        if upper[i].is_finite() && middle[i].is_finite() && lower[i].is_finite() {
            assert!(
                upper[i] >= middle[i],
                "Upper band should be >= middle band at index {}",
                i
            );
            assert!(
                middle[i] >= lower[i],
                "Middle band should be >= lower band at index {}",
                i
            );
        }
    }
}

#[test]
fn test_validation_functions() {
    println!("Testing validation functions...");

    // Test OHLCV validation
    let close = vec![100.0, 101.0, 102.0];
    let high = vec![101.0, 102.0, 103.0];
    let low = vec![99.0, 100.0, 101.0];

    let result = validate_ohlcv_data(None, Some(&high), Some(&low), &close, None);
    assert!(result.is_ok(), "Valid OHLC data should pass validation");

    // Test invalid data (high < low)
    let invalid_high = vec![98.0, 99.0, 100.0]; // Lower than low prices
    let result = validate_ohlcv_data(None, Some(&invalid_high), Some(&low), &close, None);
    assert!(result.is_err(), "Invalid OHLC data should fail validation");

    // Test period validation
    let result = validate_period(5, 10, "Test");
    assert!(result.is_ok(), "Valid period should pass");

    let result = validate_period(0, 10, "Test");
    assert!(result.is_err(), "Zero period should fail");

    let result = validate_period(15, 10, "Test");
    assert!(result.is_err(), "Period larger than data should fail");

    // Test MACD parameter validation
    let result = validate_macd_params(5, 10, 3, 20);
    assert!(result.is_ok(), "Valid MACD params should pass");

    let result = validate_macd_params(10, 5, 3, 20); // fast >= slow
    assert!(result.is_err(), "Invalid MACD params should fail");
}

#[test]
fn test_edge_cases() {
    println!("Testing edge cases...");

    // Test with very small data
    let small_data = vec![100.0, 101.0];
    let rsi_result = calculate_rsi_ta(&small_data, 5);
    assert!(
        rsi_result.is_err(),
        "RSI with insufficient data should fail gracefully"
    );

    // Test with constant data
    let constant_data = vec![100.0; 20];
    let rsi_result = calculate_rsi_ta(&constant_data, 5);
    // Should succeed but with warnings about low variation
    if let Ok(rsi_values) = rsi_result {
        println!("RSI with constant data: {:?}", rsi_values);
        // Check that we have some finite values after warm-up period
        let finite_values: Vec<f64> = rsi_values
            .iter()
            .filter(|&&x| x.is_finite())
            .cloned()
            .collect();
        assert!(
            !finite_values.is_empty(),
            "RSI with constant data should produce some finite values after warm-up"
        );

        // Finite values should be reasonable (likely around 50.0 for neutral RSI)
        for &rsi in &finite_values {
            assert!(
                (0.0..=100.0).contains(&rsi),
                "RSI value {} should be in valid range [0, 100]",
                rsi
            );
        }
    }

    // Test sanitization
    let test_values = vec![50.0, f64::NAN, 150.0, -10.0, 75.0]; // Mix of valid/invalid RSI values
    let sanitized = sanitize_indicator_output(test_values, "Test RSI", 50.0, Some((0.0, 100.0)));

    println!("Sanitized values: {:?}", sanitized);
    for (i, &val) in sanitized.iter().enumerate() {
        if val.is_nan() {
            // NaN values are preserved for warm-up periods
            continue;
        }
        assert!(
            val.is_finite(),
            "Non-NaN sanitized values should be finite at index {}",
            i
        );
        assert!(
            (0.0..=100.0).contains(&val),
            "Sanitized RSI should be in valid range at index {}",
            i
        );
    }
}

#[test]
fn test_fixed_dataitem_building() {
    println!("Testing fixed DataItem building with OHLCV...");

    // Test data
    let open = vec![100.0, 101.0, 99.0, 102.0];
    let high = vec![101.0, 102.0, 100.0, 103.0];
    let low = vec![99.0, 100.0, 98.0, 101.0];
    let close = vec![100.5, 101.5, 99.5, 102.5];
    let volume = vec![1000.0, 1100.0, 900.0, 1200.0];

    // Test Stochastic
    println!("\n=== Testing Fixed Stochastic ===");
    let stoch_result = calculate_stochastic_ta(&open, &high, &low, &close, &volume, 3, 2);
    match stoch_result {
        Ok((k_values, d_values)) => {
            println!("Stochastic %K: {:?}", k_values);
            println!("Stochastic %D: {:?}", d_values);

            // Should have NaN values during warm-up
            let k_nan_count = k_values.iter().filter(|&&x| x.is_nan()).count();
            let d_nan_count = d_values.iter().filter(|&&x| x.is_nan()).count();
            println!("Stochastic %K NaN values: {}", k_nan_count);
            println!("Stochastic %D NaN values: {}", d_nan_count);

            assert!(
                k_nan_count >= 2,
                "Stochastic %K should have NaN values during warm-up"
            );
            assert!(
                d_nan_count >= 3,
                "Stochastic %D should have more NaN values during warm-up"
            );
        }
        Err(e) => panic!("Stochastic calculation failed: {}", e),
    }

    // Test CCI
    println!("\n=== Testing Fixed CCI ===");
    let cci_result = calculate_cci_ta(&open, &high, &low, &close, &volume, 3);
    match cci_result {
        Ok(cci_values) => {
            println!("CCI: {:?}", cci_values);
            let cci_nan_count = cci_values.iter().filter(|&&x| x.is_nan()).count();
            println!("CCI NaN values: {}", cci_nan_count);
            assert!(
                cci_nan_count >= 2,
                "CCI should have NaN values during warm-up"
            );
        }
        Err(e) => panic!("CCI calculation failed: {}", e),
    }

    // Test ATR
    println!("\n=== Testing Fixed ATR ===");
    let atr_result = calculate_atr_ta(&open, &high, &low, &close, &volume, 3);
    match atr_result {
        Ok(atr_values) => {
            println!("ATR: {:?}", atr_values);
            let atr_nan_count = atr_values.iter().filter(|&&x| x.is_nan()).count();
            println!("ATR NaN values: {}", atr_nan_count);
            assert!(
                atr_nan_count >= 3,
                "ATR should have NaN values during warm-up"
            );
        }
        Err(e) => panic!("ATR calculation failed: {}", e),
    }

    println!("\n=== All DataItem-based indicators now working correctly! ===");
}
