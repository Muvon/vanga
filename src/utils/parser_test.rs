use crate::utils::parser::*;

#[test]
fn test_sort_horizons_numerically_empty() {
    let horizons: Vec<String> = vec![];
    let sorted = sort_horizons_numerically(&horizons);
    assert_eq!(sorted.len(), 0);
}

#[test]
fn test_sort_horizons_numerically_single() {
    let horizons = vec!["16h".to_string()];
    let sorted = sort_horizons_numerically(&horizons);
    assert_eq!(sorted, vec!["16h"]);
}

#[test]
fn test_sort_horizons_numerically_hours_only() {
    let horizons = vec!["32h".to_string(), "8h".to_string(), "16h".to_string()];
    let sorted = sort_horizons_numerically(&horizons);
    assert_eq!(sorted, vec!["8h", "16h", "32h"]);
}

#[test]
fn test_sort_horizons_numerically_days_only() {
    let horizons = vec!["3d".to_string(), "1d".to_string(), "2d".to_string()];
    let sorted = sort_horizons_numerically(&horizons);
    assert_eq!(sorted, vec!["1d", "2d", "3d"]);
}

#[test]
fn test_sort_horizons_numerically_mixed_hours_days() {
    let horizons = vec![
        "2d".to_string(),   // 48h
        "16h".to_string(),  // 16h
        "1d".to_string(),   // 24h
        "8h".to_string(),   // 8h
        "32h".to_string(),  // 32h
    ];
    let sorted = sort_horizons_numerically(&horizons);
    assert_eq!(sorted, vec!["8h", "16h", "1d", "32h", "2d"]);
}

#[test]
fn test_sort_horizons_numerically_already_sorted() {
    let horizons = vec!["8h".to_string(), "16h".to_string(), "32h".to_string()];
    let sorted = sort_horizons_numerically(&horizons);
    assert_eq!(sorted, vec!["8h", "16h", "32h"]);
}

#[test]
fn test_sort_horizons_numerically_reverse_order() {
    let horizons = vec!["64h".to_string(), "32h".to_string(), "16h".to_string()];
    let sorted = sort_horizons_numerically(&horizons);
    assert_eq!(sorted, vec!["16h", "32h", "64h"]);
}

#[test]
fn test_sort_horizons_numerically_complex_case() {
    // Real-world scenario from VANGA config
    let horizons = vec![
        "64h".to_string(),
        "16h".to_string(),
        "32h".to_string(),
    ];
    let sorted = sort_horizons_numerically(&horizons);
    assert_eq!(sorted, vec!["16h", "32h", "64h"]);
}

#[test]
fn test_sort_horizons_numerically_with_duplicates() {
    let horizons = vec![
        "16h".to_string(),
        "32h".to_string(),
        "16h".to_string(),
        "8h".to_string(),
    ];
    let sorted = sort_horizons_numerically(&horizons);
    // Duplicates preserved, but sorted
    assert_eq!(sorted, vec!["8h", "16h", "16h", "32h"]);
}

#[test]
fn test_sort_horizons_numerically_edge_case_24h_vs_1d() {
    // 24h and 1d are equivalent (both 24 hours)
    let horizons = vec!["1d".to_string(), "24h".to_string(), "16h".to_string()];
    let sorted = sort_horizons_numerically(&horizons);
    // Both 24h and 1d should be adjacent (order between them is stable)
    assert_eq!(sorted[0], "16h");
    assert!(sorted[1] == "1d" || sorted[1] == "24h");
    assert!(sorted[2] == "1d" || sorted[2] == "24h");
}

#[test]
fn test_parse_horizon_to_steps_hours() {
    assert_eq!(parse_horizon_to_steps("8h").unwrap(), 8);
    assert_eq!(parse_horizon_to_steps("16h").unwrap(), 16);
    assert_eq!(parse_horizon_to_steps("32h").unwrap(), 32);
}

#[test]
fn test_parse_horizon_to_steps_days() {
    assert_eq!(parse_horizon_to_steps("1d").unwrap(), 24);
    assert_eq!(parse_horizon_to_steps("2d").unwrap(), 48);
    assert_eq!(parse_horizon_to_steps("3d").unwrap(), 72);
}

#[test]
fn test_parse_horizon_to_steps_invalid() {
    assert!(parse_horizon_to_steps("invalid").is_err());
    assert!(parse_horizon_to_steps("16").is_err());
    assert!(parse_horizon_to_steps("h16").is_err());
    assert!(parse_horizon_to_steps("").is_err());
}
