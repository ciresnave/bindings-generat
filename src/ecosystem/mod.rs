// src/ecosystem/mod.rs

//! Ecosystem integration framework for generated wrappers
//!
//! Automatically detects and generates integration code for popular Rust
//! ecosystem crates based on the library being wrapped.

pub mod detector;
pub mod serde_integration;
pub mod tokio_integration;

pub use detector::{EcosystemIntegration, IntegrationDetector};

/// Standard ecosystem crates we can integrate with (Top 100 from community)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EcosystemCrate {
    // === TIER 1: Universal (almost always useful) ===
    /// Serialization/deserialization - THE most important integration
    Serde,
    /// Custom error types with derive macros
    Thiserror,
    /// Structured logging/instrumentation
    Tracing,
    /// Legacy logging facade
    Log,
    /// Lazy static initialization
    OnceCell,

    // === TIER 2: Async & Concurrency ===
    /// Async runtime (most popular)
    Tokio,
    /// Alternative async runtime
    AsyncStd,
    /// Core async traits
    Futures,
    /// Data parallelism
    Rayon,
    /// Concurrency primitives
    Crossbeam,
    /// Better locks
    ParkingLot,
    /// Concurrent hashmap
    Dashmap,
    /// Atomic Arc swapping
    ArcSwap,
    /// Tokio utilities
    TokioUtil,
    /// Async trait support
    AsyncTrait,
    /// Pin projection macros
    PinProject,
    /// Tokio streams
    TokioStream,

    // === TIER 3: Serialization Formats ===
    /// JSON backend
    SerdeJson,
    /// YAML config format
    SerdeYaml,
    /// CBOR binary format
    SerdeCbor,
    /// Serde helpers
    SerdeWith,
    /// Fast binary encoding
    Bincode,
    /// Rusty Object Notation
    Ron,
    /// JSON Schema generation
    Schemars,
    /// MessagePack format
    RmpSerde,
    /// Tiny no-std binary format
    Postcard,
    /// Query string encoding
    SerdeQs,
    /// URL-encoded forms
    SerdeUrlencoded,
    /// Enum numeric representation
    SerdeRepr,
    /// Efficient byte slice serialization
    SerdeBytes,

    // === TIER 4: Error Handling & Observability ===
    /// Application-level errors
    Anyhow,
    /// Anyhow alternative
    Eyre,
    /// Pretty CLI errors
    ColorEyre,
    /// Tracing subscriber
    TracingSubscriber,
    /// Tracing error integration
    TracingError,
    /// Sentry error reporting
    Sentry,

    // === TIER 5: CLI & Terminal ===
    /// CLI argument parsing
    Clap,
    /// Minimalist CLI parser
    Argh,
    /// Interactive prompts
    Dialoguer,
    /// Progress bars
    Indicatif,
    /// Terminal manipulation
    Console,
    /// Cross-platform terminal
    Crossterm,
    /// Interactive prompts (alternative)
    Inquire,
    /// Lightweight terminal colors
    OwoColors,

    // === TIER 6: HTTP & Networking ===
    /// Low-level HTTP
    Hyper,
    /// High-level HTTP client
    Reqwest,
    /// Modern web framework
    Axum,
    /// Filter-based web framework
    Warp,
    /// Service abstraction
    Tower,
    /// Tower HTTP middleware
    TowerHttp,
    /// Zero-copy HTTP parsing
    Httparse,
    /// Core HTTP types
    Http,
    /// URL parsing
    Url,
    /// Typed HTTP headers
    Headers,

    // === TIER 7: Time & IDs ===
    /// DateTime library (legacy but ubiquitous)
    Chrono,
    /// Modern time API
    Time,
    /// Human-readable durations
    Humantime,
    /// Random number generation
    Rand,
    /// ChaCha PRNG
    RandChacha,
    /// UUIDs
    Uuid,
    /// Sortable unique IDs
    Ulid,
    /// Short URL-safe IDs
    Nanoid,
    /// Distributed ID generation
    Snowflake,
    /// OS random number source
    Getrandom,

    // === TIER 8: Data Structures ===
    /// HashMap with stable iteration
    Indexmap,
    /// Stack-allocated vectors
    Smallvec,
    /// Efficient byte buffers
    Bytes,
    /// Bitmask types
    Bitflags,
    /// Numeric traits
    Num,
    NumTraits,
    /// Big integers
    NumBigint,
    /// Iterator extensions
    Itertools,
    /// Either<L,R> type
    Either,
    /// Graph algorithms
    Petgraph,

    // === TIER 9: Arrays & Numerics ===
    /// N-dimensional arrays
    Ndarray,
    /// Linear algebra
    Nalgebra,
    /// Statistics
    Statrs,
    /// Approximate equality
    Approx,

    // === TIER 10: Formats & Parsing ===
    /// SIMD JSON parsing
    SimdJson,
    /// CSV reading/writing
    Csv,
    /// URL encoding
    FormUrlencoded,
    /// XML parsing
    QuickXml,
    /// Image processing
    Image,

    // === TIER 11: Databases ===
    /// Async compile-time SQL
    Sqlx,
    /// Sync ORM
    Diesel,
    /// Async ORM
    SeaOrm,
    /// MongoDB driver
    Mongodb,
    /// Redis client
    Redis,
    /// SQLite bindings
    Rusqlite,
    /// Embedded KV store
    Sled,
    /// RocksDB bindings
    Rocksdb,
    /// Connection pooling
    Bb8,
    /// Connection pooling (alternative)
    Deadpool,

    // === TIER 12: Protocols & Low-Level ===
    /// Protocol Buffers
    Prost,
    /// gRPC
    Tonic,
    /// Zero-copy type casting
    Bytemuck,
    /// Safe byte casting
    Zerocopy,
    /// Parking lot internals
    ParkingLotCore,
    /// Non-blocking I/O
    Mio,
    /// Simple logger
    EnvLogger,
}
impl EcosystemCrate {
    /// Get the crate name for Cargo.toml
    pub fn crate_name(&self) -> &'static str {
        match self {
            // Tier 1: Universal
            Self::Serde => "serde",
            Self::Thiserror => "thiserror",
            Self::Tracing => "tracing",
            Self::Log => "log",
            Self::OnceCell => "once_cell",

            // Tier 2: Async & Concurrency
            Self::Tokio => "tokio",
            Self::AsyncStd => "async-std",
            Self::Futures => "futures",
            Self::Rayon => "rayon",
            Self::Crossbeam => "crossbeam",
            Self::ParkingLot => "parking_lot",
            Self::Dashmap => "dashmap",
            Self::ArcSwap => "arc-swap",
            Self::TokioUtil => "tokio-util",
            Self::AsyncTrait => "async-trait",
            Self::PinProject => "pin-project",
            Self::TokioStream => "tokio-stream",

            // Tier 3: Serialization Formats
            Self::SerdeJson => "serde_json",
            Self::SerdeYaml => "serde_yaml",
            Self::SerdeCbor => "serde_cbor",
            Self::SerdeWith => "serde_with",
            Self::Bincode => "bincode",
            Self::Ron => "ron",
            Self::Schemars => "schemars",
            Self::RmpSerde => "rmp-serde",
            Self::Postcard => "postcard",
            Self::SerdeQs => "serde_qs",
            Self::SerdeUrlencoded => "serde_urlencoded",
            Self::SerdeRepr => "serde_repr",
            Self::SerdeBytes => "serde_bytes",

            // Tier 4: Error Handling & Observability
            Self::Anyhow => "anyhow",
            Self::Eyre => "eyre",
            Self::ColorEyre => "color-eyre",
            Self::TracingSubscriber => "tracing-subscriber",
            Self::TracingError => "tracing-error",
            Self::Sentry => "sentry",

            // Tier 5: CLI & Terminal
            Self::Clap => "clap",
            Self::Argh => "argh",
            Self::Dialoguer => "dialoguer",
            Self::Indicatif => "indicatif",
            Self::Console => "console",
            Self::Crossterm => "crossterm",
            Self::Inquire => "inquire",
            Self::OwoColors => "owo-colors",

            // Tier 6: HTTP & Networking
            Self::Hyper => "hyper",
            Self::Reqwest => "reqwest",
            Self::Axum => "axum",
            Self::Warp => "warp",
            Self::Tower => "tower",
            Self::TowerHttp => "tower-http",
            Self::Httparse => "httparse",
            Self::Http => "http",
            Self::Url => "url",
            Self::Headers => "headers",

            // Tier 7: Time & IDs
            Self::Chrono => "chrono",
            Self::Time => "time",
            Self::Humantime => "humantime",
            Self::Rand => "rand",
            Self::RandChacha => "rand_chacha",
            Self::Uuid => "uuid",
            Self::Ulid => "ulid",
            Self::Nanoid => "nanoid",
            Self::Snowflake => "snowflake",
            Self::Getrandom => "getrandom",

            // Tier 8: Data Structures
            Self::Indexmap => "indexmap",
            Self::Smallvec => "smallvec",
            Self::Bytes => "bytes",
            Self::Bitflags => "bitflags",
            Self::Num => "num",
            Self::NumTraits => "num-traits",
            Self::NumBigint => "num-bigint",
            Self::Itertools => "itertools",
            Self::Either => "either",
            Self::Petgraph => "petgraph",

            // Tier 9: Arrays & Numerics
            Self::Ndarray => "ndarray",
            Self::Nalgebra => "nalgebra",
            Self::Statrs => "statrs",
            Self::Approx => "approx",

            // Tier 10: Formats & Parsing
            Self::SimdJson => "simd-json",
            Self::Csv => "csv",
            Self::FormUrlencoded => "form_urlencoded",
            Self::QuickXml => "quick-xml",
            Self::Image => "image",

            // Tier 11: Databases
            Self::Sqlx => "sqlx",
            Self::Diesel => "diesel",
            Self::SeaOrm => "sea-orm",
            Self::Mongodb => "mongodb",
            Self::Redis => "redis",
            Self::Rusqlite => "rusqlite",
            Self::Sled => "sled",
            Self::Rocksdb => "rocksdb",
            Self::Bb8 => "bb8",
            Self::Deadpool => "deadpool",

            // Tier 12: Protocols & Low-Level
            Self::Prost => "prost",
            Self::Tonic => "tonic",
            Self::Bytemuck => "bytemuck",
            Self::Zerocopy => "zerocopy",
            Self::ParkingLotCore => "parking_lot_core",
            Self::Mio => "mio",
            Self::EnvLogger => "env_logger",
        }
    }

    /// Get typical version constraint
    pub fn version(&self) -> &'static str {
        match self {
            // Tier 1
            Self::Serde => "1.0",
            Self::Thiserror => "1.0",
            Self::Tracing => "0.1",
            Self::Log => "0.4",
            Self::OnceCell => "1.0",

            // Tier 2
            Self::Tokio => "1.0",
            Self::AsyncStd => "1.0",
            Self::Futures => "0.3",
            Self::Rayon => "1.0",
            Self::Crossbeam => "0.8",
            Self::ParkingLot => "0.12",
            Self::Dashmap => "5.0",
            Self::ArcSwap => "1.0",
            Self::TokioUtil => "0.7",
            Self::AsyncTrait => "0.1",
            Self::PinProject => "1.0",
            Self::TokioStream => "0.1",

            // Tier 3
            Self::SerdeJson => "1.0",
            Self::SerdeYaml => "0.9",
            Self::SerdeCbor => "0.11",
            Self::SerdeWith => "3.0",
            Self::Bincode => "1.0",
            Self::Ron => "0.8",
            Self::Schemars => "0.8",
            Self::RmpSerde => "1.0",
            Self::Postcard => "1.0",
            Self::SerdeQs => "0.12",
            Self::SerdeUrlencoded => "0.7",
            Self::SerdeRepr => "0.1",
            Self::SerdeBytes => "0.11",

            // Tier 4
            Self::Anyhow => "1.0",
            Self::Eyre => "0.6",
            Self::ColorEyre => "0.6",
            Self::TracingSubscriber => "0.3",
            Self::TracingError => "0.2",
            Self::Sentry => "0.32",

            // Tier 5
            Self::Clap => "4.0",
            Self::Argh => "0.1",
            Self::Dialoguer => "0.11",
            Self::Indicatif => "0.17",
            Self::Console => "0.15",
            Self::Crossterm => "0.27",
            Self::Inquire => "0.6",
            Self::OwoColors => "3.0",

            // Tier 6
            Self::Hyper => "1.0",
            Self::Reqwest => "0.12",
            Self::Axum => "0.7",
            Self::Warp => "0.3",
            Self::Tower => "0.4",
            Self::TowerHttp => "0.5",
            Self::Httparse => "1.0",
            Self::Http => "1.0",
            Self::Url => "2.0",
            Self::Headers => "0.4",

            // Tier 7
            Self::Chrono => "0.4",
            Self::Time => "0.3",
            Self::Humantime => "2.0",
            Self::Rand => "0.8",
            Self::RandChacha => "0.3",
            Self::Uuid => "1.0",
            Self::Ulid => "1.0",
            Self::Nanoid => "0.4",
            Self::Snowflake => "1.0",
            Self::Getrandom => "0.2",

            // Tier 8
            Self::Indexmap => "2.0",
            Self::Smallvec => "1.0",
            Self::Bytes => "1.0",
            Self::Bitflags => "2.0",
            Self::Num => "0.4",
            Self::NumTraits => "0.2",
            Self::NumBigint => "0.4",
            Self::Itertools => "0.12",
            Self::Either => "1.0",
            Self::Petgraph => "0.6",

            // Tier 9
            Self::Ndarray => "0.15",
            Self::Nalgebra => "0.32",
            Self::Statrs => "0.16",
            Self::Approx => "0.5",

            // Tier 10
            Self::SimdJson => "0.13",
            Self::Csv => "1.0",
            Self::FormUrlencoded => "1.0",
            Self::QuickXml => "0.31",
            Self::Image => "0.25",

            // Tier 11
            Self::Sqlx => "0.7",
            Self::Diesel => "2.0",
            Self::SeaOrm => "0.12",
            Self::Mongodb => "2.0",
            Self::Redis => "0.24",
            Self::Rusqlite => "0.30",
            Self::Sled => "0.34",
            Self::Rocksdb => "0.21",
            Self::Bb8 => "0.8",
            Self::Deadpool => "0.10",

            // Tier 12
            Self::Prost => "0.12",
            Self::Tonic => "0.10",
            Self::Bytemuck => "1.0",
            Self::Zerocopy => "0.7",
            Self::ParkingLotCore => "0.9",
            Self::Mio => "0.8",
            Self::EnvLogger => "0.11",
        }
    }

    /// Get feature name for Cargo.toml
    pub fn feature_name(&self) -> &'static str {
        // Most crates use their name as feature name
        self.crate_name()
    }

    /// Description for feature documentation
    pub fn description(&self) -> &'static str {
        match self {
            // Tier 1
            Self::Serde => "Enable serialization and deserialization support",
            Self::Thiserror => "Enable custom error types with derive macros",
            Self::Tracing => "Enable structured logging and instrumentation",
            Self::Log => "Enable legacy logging facade support",
            Self::OnceCell => "Enable lazy static initialization",

            // Tier 2
            Self::Tokio => "Enable async/await support with tokio runtime",
            Self::AsyncStd => "Enable async/await support with async-std runtime",
            Self::Futures => "Enable core async traits and combinators",
            Self::Rayon => "Enable parallel iteration support",
            Self::Crossbeam => "Enable advanced concurrency primitives",
            Self::ParkingLot => "Enable better synchronization primitives",
            Self::Dashmap => "Enable concurrent hashmap support",
            Self::ArcSwap => "Enable atomic Arc swapping",
            Self::TokioUtil => "Enable tokio utility functions",
            Self::AsyncTrait => "Enable async trait implementations",
            Self::PinProject => "Enable safe pin projection",
            Self::TokioStream => "Enable tokio stream utilities",

            // Tier 3
            Self::SerdeJson => "Enable JSON serialization support",
            Self::SerdeYaml => "Enable YAML serialization support",
            Self::SerdeCbor => "Enable CBOR binary format support",
            Self::SerdeWith => "Enable additional serde helpers",
            Self::Bincode => "Enable fast binary encoding",
            Self::Ron => "Enable Rusty Object Notation support",
            Self::Schemars => "Enable JSON Schema generation",
            Self::RmpSerde => "Enable MessagePack serialization",
            Self::Postcard => "Enable tiny no-std binary format",
            Self::SerdeQs => "Enable query string serialization",
            Self::SerdeUrlencoded => "Enable URL-encoded form data",
            Self::SerdeRepr => "Enable numeric enum representation",
            Self::SerdeBytes => "Enable efficient byte slice serialization",

            // Tier 4
            Self::Anyhow => "Enable enhanced error handling with anyhow",
            Self::Eyre => "Enable enhanced error handling with eyre",
            Self::ColorEyre => "Enable pretty CLI error reports",
            Self::TracingSubscriber => "Enable tracing subscriber support",
            Self::TracingError => "Enable tracing error integration",
            Self::Sentry => "Enable Sentry error reporting",

            // Tier 5
            Self::Clap => "Enable CLI argument parsing",
            Self::Argh => "Enable minimalist CLI parsing",
            Self::Dialoguer => "Enable interactive CLI prompts",
            Self::Indicatif => "Enable progress bar support",
            Self::Console => "Enable terminal styling features",
            Self::Crossterm => "Enable cross-platform terminal manipulation",
            Self::Inquire => "Enable interactive prompts",
            Self::OwoColors => "Enable lightweight terminal colors",

            // Tier 6
            Self::Hyper => "Enable low-level HTTP support",
            Self::Reqwest => "Enable high-level HTTP client",
            Self::Axum => "Enable modern web framework",
            Self::Warp => "Enable filter-based web framework",
            Self::Tower => "Enable service abstraction layer",
            Self::TowerHttp => "Enable HTTP middleware",
            Self::Httparse => "Enable zero-copy HTTP parsing",
            Self::Http => "Enable core HTTP types",
            Self::Url => "Enable URL parsing and manipulation",
            Self::Headers => "Enable typed HTTP headers",

            // Tier 7
            Self::Chrono => "Enable date/time support (chrono)",
            Self::Time => "Enable date/time support (time)",
            Self::Humantime => "Enable human-readable duration parsing",
            Self::Rand => "Enable random number generation",
            Self::RandChacha => "Enable ChaCha PRNG",
            Self::Uuid => "Enable UUID support",
            Self::Ulid => "Enable sortable unique IDs",
            Self::Nanoid => "Enable short URL-safe IDs",
            Self::Snowflake => "Enable distributed ID generation",
            Self::Getrandom => "Enable OS random number source",

            // Tier 8
            Self::Indexmap => "Enable ordered hash maps",
            Self::Smallvec => "Enable stack-allocated vectors",
            Self::Bytes => "Enable efficient byte buffers",
            Self::Bitflags => "Enable bitmask types",
            Self::Num => "Enable numeric trait support",
            Self::NumTraits => "Enable numeric traits",
            Self::NumBigint => "Enable big integer support",
            Self::Itertools => "Enable iterator extensions",
            Self::Either => "Enable Either<L,R> type",
            Self::Petgraph => "Enable graph data structures",

            // Tier 9
            Self::Ndarray => "Enable N-dimensional array support",
            Self::Nalgebra => "Enable linear algebra support",
            Self::Statrs => "Enable statistics support",
            Self::Approx => "Enable approximate equality for floats",

            // Tier 10
            Self::SimdJson => "Enable SIMD-accelerated JSON parsing",
            Self::Csv => "Enable CSV reading/writing",
            Self::FormUrlencoded => "Enable URL encoding",
            Self::QuickXml => "Enable XML parsing",
            Self::Image => "Enable image type conversions",

            // Tier 11
            Self::Sqlx => "Enable async SQL database support",
            Self::Diesel => "Enable synchronous ORM",
            Self::SeaOrm => "Enable async ORM",
            Self::Mongodb => "Enable MongoDB driver",
            Self::Redis => "Enable Redis client",
            Self::Rusqlite => "Enable SQLite bindings",
            Self::Sled => "Enable embedded KV store",
            Self::Rocksdb => "Enable RocksDB bindings",
            Self::Bb8 => "Enable connection pooling (bb8)",
            Self::Deadpool => "Enable connection pooling (deadpool)",

            // Tier 12
            Self::Prost => "Enable Protocol Buffers support",
            Self::Tonic => "Enable gRPC support",
            Self::Bytemuck => "Enable zero-copy type casting",
            Self::Zerocopy => "Enable safe byte casting",
            Self::ParkingLotCore => "Enable parking_lot internals",
            Self::Mio => "Enable non-blocking I/O",
            Self::EnvLogger => "Enable simple logging",
        }
    }

    /// Get the tier/priority for this crate (1 = highest)
    pub fn tier(&self) -> u8 {
        match self {
            // Tier 1: Universal - almost always include these
            Self::Serde | Self::Thiserror | Self::Tracing | Self::Log | Self::OnceCell => 1,

            // Tier 2: Async/Concurrency - include if library has any async or threading
            Self::Tokio
            | Self::AsyncStd
            | Self::Futures
            | Self::Rayon
            | Self::Crossbeam
            | Self::ParkingLot
            | Self::Dashmap
            | Self::ArcSwap
            | Self::TokioUtil
            | Self::AsyncTrait
            | Self::PinProject
            | Self::TokioStream => 2,

            // Tier 3: Serialization - include if types need serialization
            Self::SerdeJson
            | Self::SerdeYaml
            | Self::SerdeCbor
            | Self::SerdeWith
            | Self::Bincode
            | Self::Ron
            | Self::Schemars
            | Self::RmpSerde
            | Self::Postcard
            | Self::SerdeQs
            | Self::SerdeUrlencoded
            | Self::SerdeRepr
            | Self::SerdeBytes => 3,

            // Tier 4: Observability - include for better debugging
            Self::Anyhow
            | Self::Eyre
            | Self::ColorEyre
            | Self::TracingSubscriber
            | Self::TracingError
            | Self::Sentry => 4,

            // Tier 5: CLI - include if library has CLI tool
            Self::Clap
            | Self::Argh
            | Self::Dialoguer
            | Self::Indicatif
            | Self::Console
            | Self::Crossterm
            | Self::Inquire
            | Self::OwoColors => 5,

            // Tier 6: HTTP - include if library does networking
            Self::Hyper
            | Self::Reqwest
            | Self::Axum
            | Self::Warp
            | Self::Tower
            | Self::TowerHttp
            | Self::Httparse
            | Self::Http
            | Self::Url
            | Self::Headers => 6,

            // Tier 7: Time/IDs - include if library deals with time or needs IDs
            Self::Chrono
            | Self::Time
            | Self::Humantime
            | Self::Rand
            | Self::RandChacha
            | Self::Uuid
            | Self::Ulid
            | Self::Nanoid
            | Self::Snowflake
            | Self::Getrandom => 7,

            // Tier 8: Data structures - include for better collections
            Self::Indexmap
            | Self::Smallvec
            | Self::Bytes
            | Self::Bitflags
            | Self::Num
            | Self::NumTraits
            | Self::NumBigint
            | Self::Itertools
            | Self::Either
            | Self::Petgraph => 8,

            // Tier 9: Math/Science - include for numerical libraries
            Self::Ndarray | Self::Nalgebra | Self::Statrs | Self::Approx => 9,

            // Tier 10: Formats - include for specific format support
            Self::SimdJson | Self::Csv | Self::FormUrlencoded | Self::QuickXml | Self::Image => 10,

            // Tier 11: Databases - include if library does database work
            Self::Sqlx
            | Self::Diesel
            | Self::SeaOrm
            | Self::Mongodb
            | Self::Redis
            | Self::Rusqlite
            | Self::Sled
            | Self::Rocksdb
            | Self::Bb8
            | Self::Deadpool => 11,

            // Tier 12: Low-level - include for specific protocol/FFI needs
            Self::Prost
            | Self::Tonic
            | Self::Bytemuck
            | Self::Zerocopy
            | Self::ParkingLotCore
            | Self::Mio
            | Self::EnvLogger => 12,
        }
    }

    /// Features to enable for this crate's Cargo.toml entry (if any)
    pub fn cargo_features(&self) -> Option<&'static [&'static str]> {
        match self {
            Self::Tokio => Some(&["full"]),
            Self::Serde => Some(&["derive"]),
            Self::Clap => Some(&["derive"]),
            Self::Tracing => Some(&["log"]),
            Self::Sqlx => Some(&["runtime-tokio", "tls-native-tls"]),
            Self::Uuid => Some(&["v4", "serde"]),
            Self::Chrono => Some(&["serde"]),
            Self::Rand => Some(&["std", "std_rng"]),
            _ => None,
        }
    }
}
/// Category of library being wrapped (helps determine relevant integrations)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LibraryCategory {
    /// Math/linear algebra
    Mathematics,

    /// Graphics/GPU/rendering
    Graphics,

    /// Machine learning/AI
    MachineLearning,

    /// Networking/protocols
    Networking,

    /// Cryptography/security
    Cryptography,

    /// Audio/video processing
    Multimedia,

    /// Database/storage
    Database,

    /// System/OS APIs
    System,

    /// Unknown/general purpose
    General,
}

impl LibraryCategory {
    /// Recommended integrations for this category
    pub fn recommended_integrations(&self) -> Vec<EcosystemCrate> {
        let mut crates = vec![
            // Tier 1: Universal - ALWAYS include these
            EcosystemCrate::Serde,
            EcosystemCrate::Thiserror,
            EcosystemCrate::Tracing,
            EcosystemCrate::Log,
        ];

        // Add category-specific crates
        match self {
            Self::Mathematics => {
                crates.extend_from_slice(&[
                    EcosystemCrate::Ndarray,
                    EcosystemCrate::Nalgebra,
                    EcosystemCrate::Num,
                    EcosystemCrate::NumTraits,
                    EcosystemCrate::Approx,
                    EcosystemCrate::Rayon,
                    EcosystemCrate::Itertools,
                    EcosystemCrate::SerdeJson,
                    EcosystemCrate::Bincode,
                ]);
            }
            Self::Graphics => {
                crates.extend_from_slice(&[
                    EcosystemCrate::Image,
                    EcosystemCrate::Bytes,
                    EcosystemCrate::Rayon,
                    EcosystemCrate::Nalgebra,
                    EcosystemCrate::Bytemuck,
                    EcosystemCrate::Zerocopy,
                    EcosystemCrate::Smallvec,
                ]);
            }
            Self::MachineLearning => {
                crates.extend_from_slice(&[
                    EcosystemCrate::Ndarray,
                    EcosystemCrate::Rayon,
                    EcosystemCrate::Tokio,
                    EcosystemCrate::SerdeJson,
                    EcosystemCrate::Bincode,
                    EcosystemCrate::Csv,
                    EcosystemCrate::Indicatif,
                    EcosystemCrate::Rand,
                ]);
            }
            Self::Networking => {
                crates.extend_from_slice(&[
                    EcosystemCrate::Tokio,
                    EcosystemCrate::Futures,
                    EcosystemCrate::Bytes,
                    EcosystemCrate::Hyper,
                    EcosystemCrate::Reqwest,
                    EcosystemCrate::Url,
                    EcosystemCrate::Http,
                    EcosystemCrate::SerdeJson,
                    EcosystemCrate::Prost,
                    EcosystemCrate::Tonic,
                ]);
            }
            Self::Cryptography => {
                crates.extend_from_slice(&[
                    EcosystemCrate::Bytes,
                    EcosystemCrate::Rand,
                    EcosystemCrate::Getrandom,
                    EcosystemCrate::Uuid,
                    EcosystemCrate::Bytemuck,
                    EcosystemCrate::Zerocopy,
                ]);
            }
            Self::Multimedia => {
                crates.extend_from_slice(&[
                    EcosystemCrate::Image,
                    EcosystemCrate::Bytes,
                    EcosystemCrate::Tokio,
                    EcosystemCrate::Rayon,
                    EcosystemCrate::Indicatif,
                ]);
            }
            Self::Database => {
                crates.extend_from_slice(&[
                    EcosystemCrate::Tokio,
                    EcosystemCrate::Sqlx,
                    EcosystemCrate::SerdeJson,
                    EcosystemCrate::Chrono,
                    EcosystemCrate::Time,
                    EcosystemCrate::Uuid,
                    EcosystemCrate::Bb8,
                    EcosystemCrate::Deadpool,
                ]);
            }
            Self::System => {
                crates.extend_from_slice(&[
                    EcosystemCrate::Tokio,
                    EcosystemCrate::Crossbeam,
                    EcosystemCrate::ParkingLot,
                    EcosystemCrate::Mio,
                    EcosystemCrate::OnceCell,
                    EcosystemCrate::Bitflags,
                ]);
            }
            Self::General => {
                crates.extend_from_slice(&[
                    EcosystemCrate::SerdeJson,
                    EcosystemCrate::Anyhow,
                    EcosystemCrate::Indexmap,
                    EcosystemCrate::OnceCell,
                ]);
            }
        }

        crates
    }
}
