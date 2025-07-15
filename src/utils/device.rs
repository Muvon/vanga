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
    ///
    /// # Returns
    ///
    /// A `Result` containing the selected `Device` or a `VangaError` if initialization fails.
    pub fn new(device_config: &str) -> Result<Device> {
        match device_config.to_lowercase().as_str() {
            "auto" | "gpu:auto" => Self::auto_select_device(),
            "cpu" => Self::cpu_device(),
            "gpu:0" => Self::specific_gpu_device(0),
            _ => Err(VangaError::ConfigError(format!(
                "Invalid device configuration: '{}'. Supported options: 'auto', 'cpu', 'gpu:0'.",
                device_config
            ))),
        }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_selection_cpu() {
        let device = DeviceManager::new("cpu").unwrap();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_device_selection_auto_falls_back_to_cpu() {
        // This test will pass if no GPU is available, which is a safe default.
        let device = DeviceManager::new("auto").unwrap();
        // The actual device depends on the test environment, but it should succeed.
        log::info!("Auto-selected device: {:?}", device);
    }

    #[test]
    fn test_invalid_device_config() {
        let result = DeviceManager::new("invalid-device");
        assert!(result.is_err());
        if let Err(VangaError::ConfigError(msg)) = result {
            assert!(msg.contains("Invalid device configuration"));
        } else {
            panic!("Expected a ConfigError for invalid device configuration");
        }
    }
}
