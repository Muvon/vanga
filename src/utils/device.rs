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
    /// # Returns
    ///
    /// A `Result` containing the selected `Device` or a `VangaError` if initialization fails.
    pub fn create_device(device_config: &str) -> Result<Device> {
        match device_config.to_lowercase().as_str() {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_selection_cpu() {
        let device = DeviceManager::create_device("cpu").unwrap();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_device_selection_auto_falls_back_to_cpu() {
        // This test will pass if no GPU is available, which is a safe default.
        let device = DeviceManager::create_device("auto").unwrap();
        // The actual device depends on the test environment, but it should succeed.
        log::info!("Auto-selected device: {:?}", device);
    }

    #[test]
    fn test_invalid_device_config() {
        let result = DeviceManager::create_device("invalid-device");
        assert!(result.is_err());
        if let Err(VangaError::ConfigError(msg)) = result {
            assert!(msg.contains("Invalid device configuration"));
        } else {
            panic!("Expected a ConfigError for invalid device configuration");
        }
    }
}

#[test]
fn test_device_selection_metal_format() {
    // Test that metal:0 format is parsed correctly
    // Note: This test may fail on non-macOS systems, which is expected
    let result = DeviceManager::create_device("metal:0");

    #[cfg(target_os = "macos")]
    {
        // On macOS, it should either succeed (if Metal available) or fail with a specific error
        match result {
            Ok(_) => log::info!("Metal device successfully selected"),
            Err(VangaError::ModelError(msg)) => {
                assert!(msg.contains("Metal device at index 0 not found"));
            }
            Err(e) => panic!("Unexpected error type: {:?}", e),
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        // On non-macOS systems, should fail with platform error
        assert!(result.is_err());
        if let Err(VangaError::ModelError(msg)) = result {
            assert!(msg.contains("Metal device not supported on this platform"));
        } else {
            panic!("Expected ModelError for Metal on non-macOS platform");
        }
    }
}

#[test]
fn test_invalid_metal_index() {
    let result = DeviceManager::create_device("metal:invalid");
    assert!(result.is_err());
    if let Err(VangaError::ConfigError(msg)) = result {
        assert!(msg.contains("Invalid Metal index in device config"));
    } else {
        panic!("Expected ConfigError for invalid Metal index");
    }
}

#[test]
fn test_updated_error_message_includes_metal() {
    let result = DeviceManager::create_device("invalid-device");
    assert!(result.is_err());
    if let Err(VangaError::ConfigError(msg)) = result {
        assert!(
            msg.contains("metal:0"),
            "Error message should mention metal:0 as supported option"
        );
    } else {
        panic!("Expected ConfigError for invalid device configuration");
    }
}
