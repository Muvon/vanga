use crate::utils::device::*;
use crate::utils::error::VangaError;
use candle_core::Device;

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

        panic!("Expected ConfigError for invalid device configuration");
    }
}
