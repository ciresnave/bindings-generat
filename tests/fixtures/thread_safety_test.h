#include <stddef.h>

// Thread-safe handle
typedef struct TSHandle* TSHandle_t;

/**
 * @brief Create a new thread-safe handle
 * 
 * This function is thread-safe and may be called concurrently from
 * multiple threads without external synchronization.
 *
 * @return TSHandle_t New handle or NULL on failure
 */
TSHandle_t tsHandleCreate(void);

/**
 * @brief Destroy a thread-safe handle
 * 
 * @param handle Handle to destroy
 */
void tsHandleDestroy(TSHandle_t handle);

// Not thread-safe handle
typedef struct NTSHandle* NTSHandle_t;

/**
 * @brief Create a new non-thread-safe handle
 * 
 * WARNING: Not thread-safe. Must be called from a single thread only.
 * Requires external synchronization for concurrent access.
 *
 * @return NTSHandle_t New handle or NULL on failure
 */
NTSHandle_t ntsHandleCreate(void);

/**
 * @brief Destroy a non-thread-safe handle
 * 
 * @param handle Handle to destroy
 */
void ntsHandleDestroy(NTSHandle_t handle);

// Reentrant handle
typedef struct ReentrantHandle* ReentrantHandle_t;

/**
 * @brief Create a reentrant handle
 * 
 * @reentrant This function is reentrant and safe for signal handlers.
 *
 * @return ReentrantHandle_t New handle or NULL on failure
 */
ReentrantHandle_t reentrantHandleCreate(void);

/**
 * @brief Destroy a reentrant handle
 * 
 * @param handle Handle to destroy
 */
void reentrantHandleDestroy(ReentrantHandle_t handle);

// Per-thread handle
typedef struct PerThreadHandle* PerThreadHandle_t;

/**
 * @brief Create a per-thread handle
 * 
 * One instance per thread is required for optimal performance.
 * Thread-local storage is recommended.
 *
 * @return PerThreadHandle_t New handle or NULL on failure
 */
PerThreadHandle_t perThreadHandleCreate(void);

/**
 * @brief Destroy a per-thread handle
 * 
 * @param handle Handle to destroy
 */
void perThreadHandleDestroy(PerThreadHandle_t handle);
