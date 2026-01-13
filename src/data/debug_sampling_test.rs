#[test]
fn debug_systematic_sampling() {
    let sorted_len = 600usize;
    let train_size = 480usize;

    let step = sorted_len as f64 / train_size as f64;
    println!(
        "sorted_len={}, train_size={}, step={:.4}",
        sorted_len, train_size, step
    );

    let mut selected_positions = std::collections::HashSet::new();
    let mut train_indices = Vec::new();

    for i in 0..train_size {
        let target_pos = (i as f64 * step) as usize;
        let mut actual_pos = target_pos.min(sorted_len - 1);

        // Find nearest unselected position
        let mut offset = 0;
        while selected_positions.contains(&actual_pos) {
            offset += 1;
            actual_pos = if offset % 2 == 0 {
                (target_pos + offset / 2).min(sorted_len - 1)
            } else {
                target_pos.saturating_sub(offset / 2)
            };

            if offset > sorted_len {
                actual_pos = (0..sorted_len)
                    .find(|&p| !selected_positions.contains(&p))
                    .unwrap_or(target_pos);
                break;
            }
        }

        selected_positions.insert(actual_pos);
        train_indices.push(actual_pos);
    }

    println!("train_indices.len() = {}", train_indices.len());
    println!("unique positions: {}", selected_positions.len());

    // Check if we have exactly train_size elements
    assert_eq!(
        train_indices.len(),
        train_size,
        "Expected {} but got {}",
        train_size,
        train_indices.len()
    );
}
