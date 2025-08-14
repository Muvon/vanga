//! Memory management utilities for VANGA LSTM
//!
//! Provides utilities for monitoring and managing memory usage during training
//! to prevent memory leaks and OOM errors.

use crate::utils::error::Result;
use log;
use std::sync::Arc;
use std::sync::Mutex;

/// Memory manager for tracking and controlling memory usage
#[derive(Debug, Clone)]
pub struct MemoryManager {
    /// Maximum allowed memory usage in MB
    max_memory_mb: usize,
    /// Frequency of memory checks (every N steps)
    check_frequency: usize,
    /// Current step counter
    step_counter: Arc<Mutex<usize>>,
    /// Enable aggressive memory cleanup
    aggressive_cleanup: bool,
}

impl MemoryManager {
    /// Create a new memory manager
    pub fn new(max_memory_mb: usize, check_frequency: usize, aggressive_cleanup: bool) -> Self {
        Self {
            max_memory_mb,
            check_frequency,
            step_counter: Arc::new(Mutex::new(0)),
            aggressive_cleanup,
        }
    }

    /// Get maximum memory limit in MB
    pub fn max_memory_mb(&self) -> usize {
        self.max_memory_mb
    }

    /// Check if memory cleanup is needed
    pub fn should_cleanup(&self) -> bool {
        let counter = self.step_counter.lock().unwrap();
        *counter % self.check_frequency == 0
    }

    /// Increment step counter
    pub fn increment_step(&self) {
        let mut counter = self.step_counter.lock().unwrap();
        *counter += 1;
    }

    /// Get current step count
    pub fn get_step_count(&self) -> usize {
        *self.step_counter.lock().unwrap()
    }

    /// Log memory usage statistics
    pub fn log_memory_stats(&self, context: &str) {
        let step = self.get_step_count();

        // Get system memory info (platform-specific)
        #[cfg(target_os = "linux")]
        {
            if let Ok(mem_info) = procfs::Meminfo::new() {
                let total_mb = mem_info.mem_total / 1024;
                let available_mb = mem_info.mem_available.unwrap_or(0) / 1024;
                let used_mb = total_mb - available_mb;
                let usage_percent = (used_mb as f64 / total_mb as f64) * 100.0;

                log::info!(
                    "💾 Memory Stats [{}] Step {}: Used {:.1} GB / {:.1} GB ({:.1}%)",
                    context,
                    step,
                    used_mb as f64 / 1024.0,
                    total_mb as f64 / 1024.0,
                    usage_percent
                );

                // Warn if memory usage is high
                if usage_percent > 85.0 {
                    log::warn!(
                        "⚠️ High memory usage detected: {:.1}% - consider reducing batch size or sequence length",
                        usage_percent
                    );
                }

                // Check against max memory limit
                if used_mb > self.max_memory_mb {
                    log::error!(
                        "❌ Memory usage ({} MB) exceeds limit ({} MB)",
                        used_mb,
                        self.max_memory_mb
                    );
                }
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            log::debug!(
                "Memory stats [{}] at step {}: not available on this platform",
                context,
                step
            );
        }
    }

    /// Force garbage collection (platform-specific)
    pub fn force_gc(&self) {
        // In Rust, we can't directly force GC, but we can suggest memory cleanup
        // by dropping unused allocations and shrinking collections
        log::debug!(
            "Suggesting memory cleanup at step {}",
            self.get_step_count()
        );

        // This is a hint to the allocator to return memory to the OS
        #[cfg(target_os = "linux")]
        {
            unsafe {
                libc::malloc_trim(0);
            }
        }
    }

    /// Check memory usage and trigger cleanup if needed
    pub fn check_and_cleanup(&self) -> Result<()> {
        if self.should_cleanup() {
            self.log_memory_stats("Periodic Check");

            if self.aggressive_cleanup {
                self.force_gc();
            }
        }

        Ok(())
    }
}

/// Memory optimization strategies
#[derive(Debug, Clone)]
pub struct MemoryOptimizationStrategy {
    /// Enable gradient checkpointing to trade compute for memory
    pub gradient_checkpointing: bool,
    /// Clear intermediate tensors aggressively
    pub clear_intermediates: bool,
    /// Compact optimizer state periodically
    pub compact_optimizer_state: bool,
    /// Clear attention routing history periodically
    pub clear_attention_history: bool,
    /// Frequency of memory optimization (every N steps)
    pub optimization_frequency: usize,
}

impl Default for MemoryOptimizationStrategy {
    fn default() -> Self {
        Self {
            gradient_checkpointing: false,
            clear_intermediates: true,
            compact_optimizer_state: true,
            clear_attention_history: true,
            optimization_frequency: 100,
        }
    }
}

impl MemoryOptimizationStrategy {
    /// Create an aggressive memory optimization strategy
    pub fn aggressive() -> Self {
        Self {
            gradient_checkpointing: true,
            clear_intermediates: true,
            compact_optimizer_state: true,
            clear_attention_history: true,
            optimization_frequency: 50,
        }
    }

    /// Create a balanced memory optimization strategy
    pub fn balanced() -> Self {
        Self {
            gradient_checkpointing: false,
            clear_intermediates: true,
            compact_optimizer_state: true,
            clear_attention_history: true,
            optimization_frequency: 100,
        }
    }

    /// Create a minimal memory optimization strategy
    pub fn minimal() -> Self {
        Self {
            gradient_checkpointing: false,
            clear_intermediates: false,
            compact_optimizer_state: true,
            clear_attention_history: false,
            optimization_frequency: 500,
        }
    }
}

/// Batch size adjustment based on memory usage
pub struct AdaptiveBatchSize {
    /// Initial batch size
    initial_batch_size: usize,
    /// Minimum batch size
    min_batch_size: usize,
    /// Maximum batch size
    max_batch_size: usize,
    /// Current batch size
    current_batch_size: usize,
    /// Memory threshold for reducing batch size (percentage)
    reduce_threshold: f64,
    /// Memory threshold for increasing batch size (percentage)
    increase_threshold: f64,
}

impl AdaptiveBatchSize {
    pub fn new(initial_batch_size: usize, min_batch_size: usize, max_batch_size: usize) -> Self {
        Self {
            initial_batch_size,
            min_batch_size,
            max_batch_size,
            current_batch_size: initial_batch_size,
            reduce_threshold: 85.0,
            increase_threshold: 60.0,
        }
    }

    /// Adjust batch size based on memory usage
    pub fn adjust(&mut self, memory_usage_percent: f64) -> usize {
        if memory_usage_percent > self.reduce_threshold
            && self.current_batch_size > self.min_batch_size
        {
            // Reduce batch size
            self.current_batch_size = (self.current_batch_size * 3 / 4).max(self.min_batch_size);
            log::warn!(
                "📉 Reducing batch size to {} due to high memory usage ({:.1}%)",
                self.current_batch_size,
                memory_usage_percent
            );
        } else if memory_usage_percent < self.increase_threshold
            && self.current_batch_size < self.max_batch_size
        {
            // Increase batch size
            self.current_batch_size = (self.current_batch_size * 5 / 4).min(self.max_batch_size);
            log::info!(
                "📈 Increasing batch size to {} (memory usage: {:.1}%)",
                self.current_batch_size,
                memory_usage_percent
            );
        }

        self.current_batch_size
    }

    /// Get current batch size
    pub fn get_batch_size(&self) -> usize {
        self.current_batch_size
    }

    /// Reset to initial batch size
    pub fn reset(&mut self) {
        self.current_batch_size = self.initial_batch_size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_manager_creation() {
        let manager = MemoryManager::new(8192, 100, true);
        assert_eq!(manager.max_memory_mb, 8192);
        assert_eq!(manager.check_frequency, 100);
        assert!(manager.aggressive_cleanup);
    }

    #[test]
    fn test_adaptive_batch_size() {
        let mut adaptive = AdaptiveBatchSize::new(32, 8, 128);
        assert_eq!(adaptive.get_batch_size(), 32);

        // High memory usage should reduce batch size
        let new_size = adaptive.adjust(90.0);
        assert_eq!(new_size, 24);

        // Low memory usage should increase batch size
        let new_size = adaptive.adjust(50.0);
        assert_eq!(new_size, 30);
    }

    #[test]
    fn test_memory_optimization_strategies() {
        let aggressive = MemoryOptimizationStrategy::aggressive();
        assert!(aggressive.gradient_checkpointing);
        assert_eq!(aggressive.optimization_frequency, 50);

        let balanced = MemoryOptimizationStrategy::balanced();
        assert!(!balanced.gradient_checkpointing);
        assert_eq!(balanced.optimization_frequency, 100);

        let minimal = MemoryOptimizationStrategy::minimal();
        assert!(!minimal.clear_intermediates);
        assert_eq!(minimal.optimization_frequency, 500);
    }
}
