# Archive and URL Support Examples

## Supported Formats

`bindings-generat` can work with sources in multiple formats:

1. **Local directories** - Traditional file system paths
2. **Local archives** - .zip, .tar.gz, .tar, .gz files
3. **Remote URLs** - HTTP/HTTPS links to archives

## Examples

### Local Directory

```bash
bindings-generat --source /usr/local/cuda/include --output cudnn-rs
```

### ZIP Archive

```bash
# Local ZIP file
bindings-generat --source ./library-v1.2.3.zip --output library-rs

# From URL
bindings-generat --source https://example.com/releases/library-v1.2.3.zip --output library-rs
```

### Tar.gz Archive

```bash
# Local tar.gz
bindings-generat --source ./library-v1.2.3.tar.gz --output library-rs

# From GitHub release
bindings-generat \
  --source https://github.com/user/library/archive/refs/tags/v1.2.3.tar.gz \
  --output library-rs
```

### GitHub Releases

Many libraries publish releases as archives on GitHub:

```bash
# Using GitHub's archive endpoint
bindings-generat \
  --source https://github.com/nvidia/cudnn/archive/v9.0.0.tar.gz \
  --output cudnn-rs

# Using a release asset
bindings-generat \
  --source https://github.com/user/repo/releases/download/v1.0/library-headers.tar.gz \
  --output library-rs
```

### Other Archives

```bash
# Plain tar file
bindings-generat --source library.tar --output library-rs

# Gzipped file (will be decompressed)
bindings-generat --source headers.h.gz --output library-rs
```

## How It Works

1. **URL Detection**: If the source starts with `http://` or `https://`, it's downloaded
2. **Archive Detection**: The file extension determines the extraction method
3. **Automatic Extraction**: Archives are extracted to a temporary directory
4. **Smart Directory Selection**: If the archive contains a single top-level directory, that's used as the source
5. **Auto-cleanup**: Temporary directories are cleaned up when processing is complete

## Tips

### Working with GitHub

GitHub provides automatic archive generation for any tag or commit:

```bash
# Format: https://github.com/<owner>/<repo>/archive/<ref>.tar.gz
# Where <ref> can be:
# - A tag: refs/tags/v1.0.0
# - A branch: refs/heads/main
# - A commit: <commit-sha>

# Example with tag
bindings-generat \
  --source https://github.com/user/library/archive/refs/tags/v1.2.3.tar.gz \
  --output library-rs

# Example with branch
bindings-generat \
  --source https://github.com/user/library/archive/refs/heads/main.tar.gz \
  --output library-rs
```

### Private URLs

For private URLs that require authentication, download the archive first:

```bash
# Download with authentication
curl -H "Authorization: token YOUR_TOKEN" \
  https://private-server.com/library.tar.gz \
  -o library.tar.gz

# Then use the local file
bindings-generat --source library.tar.gz --output library-rs
```

### Nested Archives

If your archive contains another archive, extract manually first:

```bash
# Extract outer archive
tar xzf outer.tar.gz
cd extracted/

# Use the inner source
bindings-generat --source inner-library/ --output library-rs
```

## Error Handling

If download or extraction fails, you'll see clear error messages:

```bash
$ bindings-generat --source https://invalid-url.com/lib.tar.gz --output out
Error: Failed to download file: HTTP 404

$ bindings-generat --source broken-archive.tar.gz --output out
Error: Failed to extract tar.gz archive: unexpected end of file
```

## Performance

- **Download time**: Depends on file size and network speed
- **Extraction time**: Usually <1 second for typical libraries
- **Cleanup**: Automatic when processing completes or on error
- **Caching**: Not implemented yet (planned for future release)

## Security Considerations

- Always verify the authenticity of downloaded archives
- Use HTTPS URLs when possible
- Check checksums if provided by the library maintainer
- Be cautious with archives from unknown sources

## Future Enhancements

Planned features:
- [ ] Archive caching (avoid re-downloading)
- [ ] Checksum verification (SHA256, etc.)
- [ ] Support for more formats (7z, rar)
- [ ] Proxy support for downloads
- [ ] Progress bars for large downloads
