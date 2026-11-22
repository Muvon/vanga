use super::training::pooled_forest_depth;

#[test]
fn pooled_forest_depth_scales_with_total_rows() {
	assert_eq!(pooled_forest_depth(10, 26_591, 26_591), 10);
	assert_eq!(pooled_forest_depth(10, 26_591, 132_955), 13);
	assert_eq!(pooled_forest_depth(7, 100, 100), 7);
}
