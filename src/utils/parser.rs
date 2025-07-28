use crate::utils::error::Result;

/// Parse horizon string to number of steps
pub fn parse_horizon_to_steps(horizon: &str) -> Result<usize> {
    if let Some(num_str) = horizon.strip_suffix('h') {
        num_str.parse::<usize>().map_err(|_| {
            crate::utils::error::VangaError::DataError(format!(
                "Invalid horizon format: {}",
                horizon
            ))
        })
    } else if let Some(num_str) = horizon.strip_suffix('d') {
        num_str.parse::<usize>().map(|d| d * 24).map_err(|_| {
            crate::utils::error::VangaError::DataError(format!(
                "Invalid horizon format: {}",
                horizon
            ))
        })
    } else {
        Err(crate::utils::error::VangaError::DataError(format!(
            "Unsupported horizon format: {}",
            horizon
        )))
    }
}
