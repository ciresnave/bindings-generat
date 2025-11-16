# Archive and URL Support Examples

This document demonstrates how to use `bindings-generat` with various archive formats and remote URLs.

## Local Archive Files

### ZIP Files
```bash
# Generate bindings from a local ZIP file
bindings-generat --source path/to/library.zip --output bindings.rs

# The tool will automatically:
# 1. Extract the ZIP to a temporary directory
# 2. Find the source files within
# 3. Generate bindings
# 4. Clean up the temporary directory
```

### Tar.gz Files
```bash
bindings-generat --source path/to/library.tar.gz --output bindings.rs
```

### Tar Files
```bash
bindings-generat --source path/to/library.tar --output bindings.rs
```

### Gzipped Files
```bash
bindings-generat --source path/to/library.gz --output bindings.rs
```

## Remote URLs

### GitHub Releases
```bash
# Download and extract a release directly from GitHub
bindings-generat \
  --source https://github.com/owner/repo/releases/download/v1.0.0/library.tar.gz \
  --output bindings.rs
```

### Direct Archive URLs
```bash
# Any HTTP/HTTPS URL pointing to a supported archive
bindings-generat \
  --source https://example.com/downloads/mylibrary-1.0.0.zip \
  --output bindings.rs
```

## Using Configuration Files

You can also specify archives and URLs in the configuration file:

**bindings-generat.toml:**
```toml
[library]
source = "https://github.com/libffi/libffi/releases/download/v3.4.4/libffi-3.4.4.tar.gz"
output_path = "bindings.rs"

[generation]
# ... other settings
```

Then simply run:
```bash
bindings-generat
```

## How It Works

1. **Detection**: The tool detects whether the source is:
   - A directory path (used directly)
   - A local archive file (extracted to temp directory)
   - A URL (downloaded then extracted to temp directory)

2. **Download** (if URL):
   - Uses `reqwest` to download the file
   - Saves to a temporary file
   - Preserves the original filename

3. **Extraction** (if archive):
   - Automatically detects format from file extension
   - Extracts to a temporary directory
   - Intelligently finds the source directory (handles nested archives)

4. **Cleanup**:
   - Temporary directories are automatically cleaned up
   - Uses Rust's Drop trait for guaranteed cleanup
   - Works even if the program exits with an error

## Tips

- **Nested Archives**: If your archive contains a single top-level directory, the tool automatically uses that directory as the source.
  
- **Network Issues**: If a download fails, you'll get a clear error message. You can download the archive manually and use the local file instead.

- **Cache**: Currently, archives are re-downloaded each time. To avoid this, download once and use the local file.

- **Progress**: For large downloads, consider using a download manager and then passing the local file to the tool.

## Security Considerations

- Always verify URLs before using them
- Be cautious with archives from untrusted sources
- The tool will extract archives to system temp directories with restricted permissions
- Consider using checksums to verify downloaded files (not yet implemented in tool)
