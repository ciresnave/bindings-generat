# Integrating Library Database with Discovery System

## Goal

Connect the remote database fetcher to the library discovery system so that when users encounter linker errors, we automatically fetch library metadata and suggest installation methods.

## Current State

### What We Have

1. **Remote fetcher** (`src/database/remote.rs`)
   - Fetches TOML from GitHub
   - Caches locally for 7 days
   - Works offline with stale cache
   - Symbol-based search

2. **Example libraries** (bindings-generat-db)
   - zlib, OpenSSL, SQLite, cuDNN
   - Multiple installation sources
   - Platform-specific metadata

3. **Discovery system** (`src/discovery/`)
   - Parses linker errors
   - Extracts library names
   - Currently uses embedded database only

### What We Need

Connect the remote fetcher when embedded database lookups fail.

## Integration Points

### 1. Update `try_discover_unknown_library()`

**File:** `src/discovery/mod.rs` or similar

**Current flow:**
```rust
fn try_discover_unknown_library(lib_name: &str) -> Result<Option<LibraryInfo>> {
    // 1. Check embedded database
    if let Some(entry) = EMBEDDED_DB.find_by_name(lib_name) {
        return Ok(Some(convert_entry(entry)));
    }
    
    // 2. Return None - library unknown
    Ok(None)
}
```

**Updated flow:**
```rust
use crate::database::remote::RemoteDatabaseFetcher;

fn try_discover_unknown_library(lib_name: &str) -> Result<Option<LibraryInfo>> {
    // 1. Check embedded database first (fast)
    if let Some(entry) = EMBEDDED_DB.find_by_name(lib_name) {
        debug!("Found {} in embedded database", lib_name);
        return Ok(Some(convert_entry(entry)));
    }
    
    // 2. Try remote database (with caching)
    match RemoteDatabaseFetcher::new() {
        Ok(fetcher) => {
            if let Ok(metadata) = fetcher.get_library(lib_name) {
                info!("Found {} in remote database", lib_name);
                let entry = LibraryEntry::from(metadata);
                return Ok(Some(convert_entry(&entry)));
            }
        }
        Err(e) => {
            warn!("Failed to create remote fetcher: {}", e);
        }
    }
    
    // 3. Not found anywhere
    debug!("Library {} not found in any database", lib_name);
    Ok(None)
}
```

### 2. Add User Feedback

When fetching from remote database, inform the user:

```rust
// Before fetching
println!("🔍 Library '{}' not in local database, checking online...", lib_name);

// On success
println!("✅ Found installation instructions for '{}'", lib_name);
println!("   Source: bindings-generat-db (community database)");

// On failure
println!("⚠️  Library '{}' not found in database", lib_name);
println!("   Consider contributing it: https://github.com/ciresnave/bindings-generat-db");
```

### 3. Cache Warming (Optional)

For popular libraries, pre-populate cache on first run:

```rust
pub fn warm_cache() -> Result<()> {
    let popular_libs = vec!["openssl", "zlib", "sqlite", "libpng", "libjpeg"];
    let fetcher = RemoteDatabaseFetcher::new()?;
    
    println!("📦 Warming library database cache...");
    
    for lib_name in popular_libs {
        if let Err(e) = fetcher.get_library(lib_name) {
            debug!("Failed to cache {}: {}", lib_name, e);
        }
    }
    
    println!("✅ Cache warmed with popular libraries");
    Ok(())
}
```

### 4. CLI Commands (Optional)

Add database management commands:

```rust
// Search command
pub fn cmd_db_search(query: &str) -> Result<()> {
    let fetcher = RemoteDatabaseFetcher::new()?;
    
    if let Ok(metadata) = fetcher.get_library(query) {
        println!("📚 {}", metadata.display_name);
        println!("   {}", metadata.description);
        println!("   Homepage: {}", metadata.homepage);
        println!("   License: {}", metadata.license);
        println!();
        
        for (platform, info) in metadata.platforms {
            println!("   Platform: {}", platform);
            println!("   Installation methods: {}", info.sources.len());
        }
        
        Ok(())
    } else {
        println!("❌ Library '{}' not found", query);
        Err(anyhow!("Library not found"))
    }
}

// Clear cache command
pub fn cmd_db_clear_cache() -> Result<()> {
    let fetcher = RemoteDatabaseFetcher::new()?;
    fetcher.clear_cache()?;
    println!("✅ Library database cache cleared");
    println!("   Location: {:?}", fetcher.cache_path());
    Ok(())
}

// Cache info command
pub fn cmd_db_info() -> Result<()> {
    let fetcher = RemoteDatabaseFetcher::new()?;
    let cache_path = fetcher.cache_path();
    
    println!("📊 Library Database Info");
    println!("   Cache location: {:?}", cache_path);
    
    if cache_path.exists() {
        let count = std::fs::read_dir(cache_path)?
            .filter(|e| e.is_ok())
            .count();
        println!("   Cached libraries: {}", count);
    } else {
        println!("   Cache: empty");
    }
    
    println!("   Freshness: 7 days");
    println!("   Source: https://github.com/ciresnave/bindings-generat-db");
    Ok(())
}
```

## Testing Plan

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remote_database_integration() {
        // Create fetcher
        let fetcher = RemoteDatabaseFetcher::new().unwrap();
        
        // Try fetching a known library (requires network)
        if let Ok(metadata) = fetcher.get_library("zlib") {
            assert_eq!(metadata.name, "zlib");
            assert!(!metadata.platforms.is_empty());
        }
    }

    #[test]
    fn test_offline_mode() {
        let fetcher = RemoteDatabaseFetcher::new_offline().unwrap();
        
        // Should work if cache exists, otherwise fails gracefully
        let result = fetcher.get_library("zlib");
        // Don't assert on result - depends on cache state
    }
}
```

### Integration Tests

```rust
#[test]
fn test_discovery_with_remote_database() {
    // Simulate linker error for a library only in remote database
    let lib_name = "openssl";  // Assuming not in embedded DB
    
    let result = try_discover_unknown_library(lib_name).unwrap();
    
    if let Some(lib_info) = result {
        assert_eq!(lib_info.name, lib_name);
        assert!(!lib_info.sources.is_empty());
    }
}
```

### Manual Testing

1. **Fresh fetch:**
   ```bash
   cargo run -- generate path/to/headers.h
   # Encounter linker error
   # Should see: "🔍 Library 'xxx' not in local database, checking online..."
   # Should see: "✅ Found installation instructions..."
   ```

2. **Cache hit:**
   ```bash
   # Run again immediately
   # Should be instant (no network call)
   ```

3. **Offline mode:**
   ```bash
   # Disconnect network
   cargo run -- generate path/to/headers.h
   # Should use cached data if available
   # Should fail gracefully if not cached
   ```

4. **Unknown library:**
   ```bash
   # Try library not in database
   # Should see: "⚠️ Library 'xxx' not found in database"
   # Should suggest contributing
   ```

## Error Handling

### Network Failures

```rust
match RemoteDatabaseFetcher::new() {
    Ok(fetcher) => {
        match fetcher.get_library(lib_name) {
            Ok(metadata) => /* success */,
            Err(e) => {
                debug!("Remote database lookup failed: {}", e);
                // Fall through to other discovery methods
            }
        }
    }
    Err(e) => {
        debug!("Failed to create remote fetcher: {}", e);
        // Continue without remote database
    }
}
```

### Timeout Handling

Already built into `reqwest::blocking::Client`:
- 30 second timeout
- Automatic retry not implemented (could add)

### Cache Corruption

```rust
fn load_toml_file(&self, path: &Path) -> Result<RemoteLibraryMetadata> {
    let content = fs::read_to_string(path)?;
    
    match toml::from_str(&content) {
        Ok(metadata) => Ok(metadata),
        Err(e) => {
            warn!("Corrupted cache file: {:?}, deleting", path);
            let _ = fs::remove_file(path);  // Delete corrupted cache
            Err(e.into())
        }
    }
}
```

## Performance Considerations

### Cache First Strategy

- **Fast path:** Cache hit (< 1ms)
- **Slow path:** Network fetch (100-500ms)
- **Degraded:** Stale cache (< 1ms, slightly outdated)

### Parallel Lookups (Future)

If looking up multiple libraries:

```rust
use rayon::prelude::*;

let libraries = vec!["zlib", "openssl", "sqlite"];
let results: Vec<_> = libraries.par_iter()
    .filter_map(|lib| fetcher.get_library(lib).ok())
    .collect();
```

### Background Updates (Future)

Update cache in background:

```rust
use std::thread;

fn refresh_cache_background(lib_names: Vec<String>) {
    thread::spawn(move || {
        let fetcher = match RemoteDatabaseFetcher::new() {
            Ok(f) => f,
            Err(_) => return,
        };
        
        for lib_name in lib_names {
            let _ = fetcher.get_library(&lib_name);
        }
    });
}
```

## Configuration (Future)

Allow users to configure database behavior:

```toml
# .bindings-generat.toml

[database]
# Enable remote database lookups
remote_enabled = true

# Cache freshness in days
cache_freshness_days = 7

# Custom database repository
remote_repo = "github.com/your-org/custom-db"

# Offline mode
offline_mode = false
```

## Rollout Plan

### Phase 1: Silent Integration (Current)
- Add remote lookup to discovery
- Log to debug only
- No user-facing changes
- Verify it works

### Phase 2: User Notifications
- Add progress messages
- Show installation instructions
- Link to database repo
- Encourage contributions

### Phase 3: CLI Commands
- `bindings-generat db search <name>`
- `bindings-generat db clear-cache`
- `bindings-generat db info`
- `bindings-generat db list`

### Phase 4: Advanced Features
- Background cache updates
- Parallel lookups
- Custom database repos
- Statistics/analytics

## Documentation Updates Needed

1. **README.md** - Mention community database
2. **INSTALLATION.md** - Explain auto-discovery
3. **TROUBLESHOOTING.md** - Database cache issues
4. **CONTRIBUTING.md** - Link to bindings-generat-db repo

## Next Steps

1. ✅ Create remote fetcher module
2. ✅ Test with example libraries
3. ⏳ **Integrate with discovery** (THIS DOCUMENT)
4. ⏳ Add user feedback messages
5. ⏳ Test end-to-end workflow
6. ⏳ Update documentation
7. ⏳ Push bindings-generat-db to GitHub
8. ⏳ Announce to community

## Success Criteria

- ✅ User hits linker error
- ✅ Tool checks remote database automatically
- ✅ Found library displays installation instructions
- ✅ Caching works (second lookup is instant)
- ✅ Offline mode degrades gracefully
- ✅ Unknown libraries suggest contributing

---

*Integration Guide for Sprint 2 Task #7*
*Last Updated: Current session*
