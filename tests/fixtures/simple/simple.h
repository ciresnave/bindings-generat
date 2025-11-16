#ifndef SIMPLE_H
#define SIMPLE_H

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle type
typedef struct SimpleHandle_t* SimpleHandle;

// Status codes (error handling pattern)
typedef enum {
    SIMPLE_STATUS_SUCCESS = 0,
    SIMPLE_STATUS_ERROR_INVALID_ARGUMENT = 1,
    SIMPLE_STATUS_ERROR_OUT_OF_MEMORY = 2,
    SIMPLE_STATUS_ERROR_NOT_INITIALIZED = 3,
} SimpleStatus;

// RAII pattern: create/destroy
SimpleStatus simple_create(SimpleHandle* handle);
void simple_destroy(SimpleHandle handle);

// Operations on handle
SimpleStatus simple_process(SimpleHandle handle, int value);
SimpleStatus simple_get_value(SimpleHandle handle, int* out_value);
SimpleStatus simple_set_name(SimpleHandle handle, const char* name);
SimpleStatus simple_get_name(SimpleHandle handle, char* buffer, int buffer_size);

#ifdef __cplusplus
}
#endif

#endif // SIMPLE_H
