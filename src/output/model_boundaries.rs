use crate::output::prediction_types::{PriceBin, PriceLevelPrediction};

/// Unified model boundaries for consistent exit generation and optimization
#[derive(Debug, Clone)]
pub struct ModelBoundaries {
    /// The center of the most extreme suitable bin (this is where middle exit should reach)
    pub max_exit_boundary_price: f64,
    pub max_exit_boundary_percent: f64,

    /// The absolute limit (edge of most extreme bin) - exits should never exceed this
    pub absolute_boundary_price: f64,
    pub absolute_boundary_percent: f64,

    /// Suitable bins for this direction (centers in profitable direction)
    pub suitable_bins: Vec<(String, f64, PriceBin)>, // (name, probability, bin)

    /// Direction this boundary is calculated for
    pub direction: String,
}

impl ModelBoundaries {
    /// Calculate unified model boundaries for consistent use across all components
    pub fn calculate(
        price_levels: &PriceLevelPrediction,
        current_price: f64,
        direction: &str,
        volatility_fallback_percent: f64,
    ) -> Self {
        // Find suitable bins (same logic as generate_smart_exits)
        let suitable_bins: Vec<(String, f64, PriceBin)> = price_levels
            .bins
            .iter()
            .filter_map(|(name, bin)| {
                let center = (bin.price[0] + bin.price[1]) / 2.0;
                let is_suitable = if direction == "LONG" {
                    center > current_price // LONG needs bins above current price
                } else {
                    center < current_price // SHORT needs bins below current price
                };

                if is_suitable {
                    Some((name.clone(), bin.probability, bin.clone()))
                } else {
                    None
                }
            })
            .collect();

        if suitable_bins.is_empty() {
            // Fallback to volatility-based boundaries
            let fallback_distance = volatility_fallback_percent / 100.0;
            let (boundary_price, absolute_price) = if direction == "LONG" {
                let boundary = current_price * (1.0 + fallback_distance);
                let absolute = current_price * (1.0 + fallback_distance * 1.5);
                (boundary, absolute)
            } else {
                let boundary = current_price * (1.0 - fallback_distance);
                let absolute = current_price * (1.0 - fallback_distance * 1.5);
                (boundary, absolute)
            };

            return Self {
                max_exit_boundary_price: boundary_price,
                max_exit_boundary_percent: ((boundary_price - current_price).abs() / current_price)
                    * 100.0,
                absolute_boundary_price: absolute_price,
                absolute_boundary_percent: ((absolute_price - current_price).abs() / current_price)
                    * 100.0,
                suitable_bins: Vec::new(),
                direction: direction.to_string(),
            };
        }

        // Find the most extreme suitable bin
        let most_extreme_bin = if direction == "LONG" {
            // For LONG, find bin with highest center (most optimistic)
            suitable_bins.iter().max_by(|a, b| {
                let center_a = (a.2.price[0] + a.2.price[1]) / 2.0;
                let center_b = (b.2.price[0] + b.2.price[1]) / 2.0;
                center_a.partial_cmp(&center_b).unwrap()
            })
        } else {
            // For SHORT, find bin with lowest center (most optimistic)
            suitable_bins.iter().min_by(|a, b| {
                let center_a = (a.2.price[0] + a.2.price[1]) / 2.0;
                let center_b = (b.2.price[0] + b.2.price[1]) / 2.0;
                center_a.partial_cmp(&center_b).unwrap()
            })
        };

        let (boundary_price, absolute_price) = if let Some((_, _, bin)) = most_extreme_bin {
            let center = (bin.price[0] + bin.price[1]) / 2.0;
            let edge = if direction == "LONG" {
                bin.price[1] // Upper edge for LONG
            } else {
                bin.price[0] // Lower edge for SHORT
            };
            (center, edge)
        } else {
            // This shouldn't happen since we checked suitable_bins.is_empty() above
            let fallback_distance = volatility_fallback_percent / 100.0;
            if direction == "LONG" {
                let boundary = current_price * (1.0 + fallback_distance);
                let absolute = current_price * (1.0 + fallback_distance * 1.5);
                (boundary, absolute)
            } else {
                let boundary = current_price * (1.0 - fallback_distance);
                let absolute = current_price * (1.0 - fallback_distance * 1.5);
                (boundary, absolute)
            }
        };

        Self {
            max_exit_boundary_price: boundary_price,
            max_exit_boundary_percent: ((boundary_price - current_price).abs() / current_price)
                * 100.0,
            absolute_boundary_price: absolute_price,
            absolute_boundary_percent: ((absolute_price - current_price).abs() / current_price)
                * 100.0,
            suitable_bins,
            direction: direction.to_string(),
        }
    }

    /// Validate that an exit price respects the model boundaries
    pub fn validate_exit_price(&self, exit_price: f64, current_price: f64) -> Result<(), String> {
        // Check profitable direction
        let is_profitable = if self.direction == "LONG" {
            exit_price > current_price
        } else {
            exit_price < current_price
        };

        if !is_profitable {
            return Err(format!(
                "Exit price ${:.5} is not profitable for {} (current: ${:.5})",
                exit_price, self.direction, current_price
            ));
        }

        // Check absolute boundary
        let within_absolute_boundary = if self.direction == "LONG" {
            exit_price <= self.absolute_boundary_price
        } else {
            exit_price >= self.absolute_boundary_price
        };

        if !within_absolute_boundary {
            return Err(format!(
                "Exit price ${:.5} exceeds absolute model boundary ${:.5} for {}",
                exit_price, self.absolute_boundary_price, self.direction
            ));
        }

        Ok(())
    }

    /// Get the maximum scale factor that keeps middle exit within boundary
    pub fn get_max_scale_factor(&self, current_middle_exit_distance_percent: f64) -> f64 {
        if current_middle_exit_distance_percent <= 0.0 {
            return 1.0;
        }

        // Middle exit should reach the boundary center, not exceed it
        self.max_exit_boundary_percent / current_middle_exit_distance_percent
    }
}
