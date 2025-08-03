// src/utils/device.rs
use crate::utils::error::{Result, VangaError};
use candle_core::Device;

/// Manages device selection for compute operations (CUDA, Metal, CPU).
pub struct DeviceManager;

impl DeviceManager {
    /// Creates a new compute device based on the provided configuration.
    ///
    /// It intelligently selects the best available hardware based on a prioritized
    /// auto-detection strategy (CUDA > Metal > CPU) or a specific user request.
    ///
    /// # Arguments
    ///
    /// * `device_config` - A string specifying the desired device.
    ///   - `"auto"`: (Default) Automatically selects the best available device.
    ///   - `"cpu"`: Forces CPU usage.
    ///   - `"gpu:0"`: Selects the first available GPU (NVIDIA CUDA or Apple Metal).
    ///   - `"metal:0"`: Selects the first Apple Metal device (macOS only).
    /// * `seed` - Optional seed for reproducible training (0 = random, >0 = reproducible)
    /// # Returns
    ///
    /// A `Result` containing the selected `Device` or a `VangaError` if initialization fails.
    pub fn create_device(device_config: &str) -> Result<Device> {
        Self::create_device_with_seed(device_config, None)
    }

    /// Creates a new compute device with optional seed for reproducible training
    pub fn create_device_with_seed(device_config: &str, seed: Option<u64>) -> Result<Device> {
        let device = match device_config.to_lowercase().as_str() {
            "auto" | "gpu:auto" => Self::auto_select_device(),
            "cpu" => Self::cpu_device(),
            "gpu:0" => Self::specific_gpu_device(0),
            #[cfg(target_os = "macos")]
            device_str if device_str.starts_with("metal:") => {
                let index = device_str[6..].parse::<usize>().map_err(|_| {
                    VangaError::ConfigError(format!("Invalid Metal index in device config: {}", device_str))
                })?;
                Self::specific_metal_device(index)
            }
            _ => Err(VangaError::ConfigError(format!(
                "Invalid device configuration: '{}'. Supported options: 'auto', 'cpu', 'gpu:0', 'metal:0'.",
                device_config
            ))),
        }?;

        // Set seed for reproducible training if provided
        if let Some(seed_value) = seed {
            if seed_value == 0 {
                log::info!("🎲 Seed = 0: Using random device initialization");
                // Don't call set_seed for random initialization
            } else {
                log::info!(
                    "🎲 Setting device seed to {} for reproducible training",
                    seed_value
                );
                device.set_seed(seed_value).map_err(|e| {
                    VangaError::ModelError(format!("Failed to set device seed: {}", e))
                })?;
            }
        }

        Ok(device)
    }

    /// Automatically selects the best available device with a prioritized search.
    fn auto_select_device() -> Result<Device> {
        // 1. Prioritize CUDA for NVIDIA GPUs
        if let Ok(device) = Device::new_cuda(0) {
            log::info!("✅ Found and selected CUDA device.");
            return Ok(device);
        }

        // 2. Check for Apple Silicon Metal support
        #[cfg(target_os = "macos")]
        if let Ok(device) = Device::new_metal(0) {
            log::info!("✅ Found and selected Apple Metal device.");
            return Ok(device);
        }

        // 3. Fallback to CPU
        log::warn!("⚠️ No GPU detected (CUDA or Metal). Falling back to CPU. For GPU acceleration, ensure you have the correct drivers (NVIDIA) or are on an Apple Silicon Mac.");
        Self::cpu_device()
    }

    /// Forces the selection of the CPU device.
    fn cpu_device() -> Result<Device> {
        log::info!("🔧 Using CPU device");
        Ok(Device::Cpu)
    }

    /// Selects a specific GPU device by index.
    fn specific_gpu_device(ordinal: usize) -> Result<Device> {
        // Try CUDA first
        if let Ok(device) = Device::new_cuda(ordinal) {
            log::info!("✅ Selected specific CUDA device: gpu:{}", ordinal);
            return Ok(device);
        }

        // Try Metal next (for macOS)
        #[cfg(target_os = "macos")]
        if let Ok(device) = Device::new_metal(ordinal) {
            log::info!("✅ Selected specific Apple Metal device: gpu:{}", ordinal);
            return Ok(device);
        }

        Err(VangaError::ModelError(format!(
            "GPU device at index {} not found. No CUDA or Metal device available.",
            ordinal
        )))
    }

    /// Selects a specific Apple Metal device by index (macOS only).
    #[cfg(target_os = "macos")]
    fn specific_metal_device(ordinal: usize) -> Result<Device> {
        #[cfg(target_os = "macos")]
        {
            if let Ok(device) = Device::new_metal(ordinal) {
                log::info!("✅ Selected specific Apple Metal device: metal:{}", ordinal);
                return Ok(device);
            }
            Err(VangaError::ModelError(format!(
                "Metal device at index {} not found. Ensure you're on macOS with Apple Silicon.",
                ordinal
            )))
        }

        #[cfg(not(target_os = "macos"))]
        {
            Err(VangaError::ModelError(format!(
                "Metal device not supported on this platform. Metal is only available on macOS."
            )))
        }
    }
}
