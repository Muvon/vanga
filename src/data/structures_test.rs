use crate::data::structures::*;

#[test]
fn test_market_data_row_creation() {
    let row = MarketDataRow::new(1_640_995_200, 47000.0, 47100.0, 46900.0, 47050.0, 1234.56);

    assert_eq!(row.timestamp, 1_640_995_200);
    assert_eq!(row.open, 47000.0);
    assert_eq!(row.high, 47100.0);
    assert_eq!(row.low, 46900.0);
    assert_eq!(row.close, 47050.0);
    assert_eq!(row.volume, 1234.56);
}

#[test]
fn test_market_data_validation() {
    let valid_row = MarketDataRow::new(1_640_995_200, 47000.0, 47100.0, 46900.0, 47050.0, 1234.56);
    assert!(valid_row.is_valid());

    let invalid_row =
        MarketDataRow::new(1_640_995_200, 47000.0, 46800.0, 46900.0, 47050.0, 1234.56); // high < low
    assert!(!invalid_row.is_valid());

    let negative_price =
        MarketDataRow::new(1_640_995_200, -47000.0, 47100.0, 46900.0, 47050.0, 1234.56);
    assert!(!negative_price.is_valid());
}

#[test]
fn test_price_calculations() {
    let row = MarketDataRow::new(1_640_995_200, 47000.0, 47100.0, 46900.0, 47050.0, 1234.56);

    assert_eq!(row.typical_price(), (47100.0 + 46900.0 + 47050.0) / 3.0);
    assert_eq!(row.price_range(), 47100.0 - 46900.0);
    assert_eq!(row.price_change(), 47050.0 - 47000.0);

    let expected_percent = ((47050.0 - 47000.0) / 47000.0) * 100.0;
    assert!((row.price_change_percent() - expected_percent).abs() < 0.001);
}

#[test]
fn test_datetime_conversion() {
    let row = MarketDataRow::new(1_640_995_200, 47000.0, 47100.0, 46900.0, 47050.0, 1234.56);
    let dt = row.datetime();
    assert_eq!(dt.timestamp(), 1_640_995_200);
}

#[test]
fn test_serialization() {
    let row = MarketDataRow::new(1_640_995_200, 47000.0, 47100.0, 46900.0, 47050.0, 1234.56);

    let json = serde_json::to_string(&row).unwrap();
    let deserialized: MarketDataRow = serde_json::from_str(&json).unwrap();

    assert_eq!(row, deserialized);
}

#[test]
fn test_extended_market_data() {
    let base = MarketDataRow::new(1_640_995_200, 47000.0, 47100.0, 46900.0, 47050.0, 1234.56);
    let extended = ExtendedMarketData::from_base(base.clone());

    assert_eq!(extended.base, base);
    assert!(extended.vwap.is_none());
    assert!(extended.trade_count.is_none());
    assert!(extended.spread.is_none());
}
