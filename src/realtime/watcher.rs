//! File watching functionality for real-time streaming
//!
//! This module provides cross-platform file watching capabilities using the notify crate.
//! It monitors CSV files for changes and emits events when new data is appended.

use crate::utils::error::{Result, VangaError};
use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::Path;
use tokio::sync::mpsc;

/// Cross-platform file watcher for monitoring CSV file changes
///
/// Uses the notify crate to provide efficient file system monitoring across
/// different operating systems (Linux, macOS, Windows).
pub struct FileWatcher {
    _watcher: RecommendedWatcher,
    receiver: mpsc::UnboundedReceiver<Event>,
}

impl FileWatcher {
    /// Create a new file watcher for the specified path
    ///
    /// # Arguments
    /// * `path` - Path to the file to watch
    ///
    /// # Returns
    /// * `Result<Self>` - New FileWatcher instance or error
    ///
    /// # Example
    /// ```rust,no_run
    /// use vanga::realtime::FileWatcher;
    /// use std::path::PathBuf;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let watcher = FileWatcher::new(PathBuf::from("data/live.csv"))?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let (tx, receiver) = mpsc::unbounded_channel();

        let mut watcher = RecommendedWatcher::new(
            move |res: notify::Result<Event>| {
                match res {
                    Ok(event) => {
                        // Filter for relevant events (file modifications)
                        match &event.kind {
                            EventKind::Modify(_) | EventKind::Create(_) => {
                                if let Err(e) = tx.send(event) {
                                    log::error!("Failed to send file event: {}", e);
                                }
                            }
                            _ => {
                                // Ignore other event types (access, remove, etc.)
                                log::debug!("Ignoring file event: {:?}", event.kind);
                            }
                        }
                    }
                    Err(e) => {
                        log::error!("File watcher error: {}", e);
                    }
                }
            },
            Config::default(),
        )
        .map_err(|e| VangaError::IoError(format!("Failed to create file watcher: {}", e)))?;

        // Watch the specific file (not recursive)
        watcher
            .watch(path.as_ref(), RecursiveMode::NonRecursive)
            .map_err(|e| {
                VangaError::IoError(format!(
                    "Failed to watch file {}: {}",
                    path.as_ref().display(),
                    e
                ))
            })?;

        log::info!("File watcher created for: {}", path.as_ref().display());

        Ok(Self {
            _watcher: watcher,
            receiver,
        })
    }

    /// Wait for the next file change event
    ///
    /// This method blocks until a file change event is received or the watcher is closed.
    ///
    /// # Returns
    /// * `Option<Event>` - File change event or None if watcher is closed
    pub async fn next_event(&mut self) -> Option<Event> {
        self.receiver.recv().await
    }

    /// Check if there are pending events without blocking
    ///
    /// # Returns
    /// * `bool` - True if events are available
    pub fn has_pending_events(&self) -> bool {
        !self.receiver.is_empty()
    }
}

impl Drop for FileWatcher {
    fn drop(&mut self) {
        log::debug!("File watcher dropped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::Write;
    use tempfile::NamedTempFile;
    use tokio::time::{timeout, Duration};

    #[tokio::test]
    async fn test_file_watcher_creation() {
        let temp_file = NamedTempFile::new().unwrap();
        let watcher = FileWatcher::new(temp_file.path());
        assert!(watcher.is_ok());
    }

    #[tokio::test]
    async fn test_file_modification_detection() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let mut watcher = FileWatcher::new(temp_file.path()).unwrap();

        // Write to file to trigger modification event
        writeln!(temp_file, "test data").unwrap();
        temp_file.flush().unwrap();

        // Wait for event with timeout
        let event = timeout(Duration::from_secs(2), watcher.next_event()).await;
        assert!(event.is_ok());
        assert!(event.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_nonexistent_file_error() {
        let result = FileWatcher::new("/nonexistent/path/file.csv");
        assert!(result.is_err());
    }
}
