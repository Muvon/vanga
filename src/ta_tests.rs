// Test ta crate capabilities
use ta::indicators::*;
use ta::Next;

#[test]
fn test_ta_indicators() {
    println!("Testing ta crate indicators...");

    // Sample price data
    let prices = vec![
        100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0,
    ];

    // Test RSI
    println!("\n=== RSI Test ===");
    let mut rsi = RelativeStrengthIndex::new(5).unwrap(); // Shorter period for test data
    for price in &prices {
        let rsi_value = rsi.next(*price);
        println!("Price: {}, RSI: {}", price, rsi_value);
    }

    // Test SMA
    println!("\n=== SMA Test ===");
    let mut sma = SimpleMovingAverage::new(3).unwrap(); // Shorter period
    for price in &prices {
        let sma_value = sma.next(*price);
        println!("Price: {}, SMA: {}", price, sma_value);
    }

    // Test EMA
    println!("\n=== EMA Test ===");
    let mut ema = ExponentialMovingAverage::new(3).unwrap(); // Shorter period
    for price in &prices {
        let ema_value = ema.next(*price);
        println!("Price: {}, EMA: {}", price, ema_value);
    }

    // Test MACD
    println!("\n=== MACD Test ===");
    let mut macd = MovingAverageConvergenceDivergence::new(3, 6, 2).unwrap(); // Shorter periods
    for price in &prices {
        let macd_value = macd.next(*price);
        println!("Price: {}, MACD: {:?}", price, macd_value);
    }

    // Test Bollinger Bands
    println!("\n=== Bollinger Bands Test ===");
    let mut bb = BollingerBands::new(5, 2.0).unwrap(); // Shorter period
    for price in &prices {
        let bb_value = bb.next(*price);
        println!("Price: {}, BB: {:?}", price, bb_value);
    }

    println!("\nAll ta crate tests completed successfully!");
}
