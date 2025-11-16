// Simple C library for testing bindings-generat
#ifndef SIMPLE_LIB_H
#define SIMPLE_LIB_H

// Status codes
typedef enum {
    STATUS_OK = 0,
    STATUS_ERROR_INVALID = 1,
    STATUS_ERROR_NOT_FOUND = 2,
} SimpleStatus;

// Opaque context type
typedef struct simple_context_t simple_context_t;

// Create a new context
simple_context_t* simple_context_create(void);

// Set a value in the context
SimpleStatus simple_context_set_value(simple_context_t* ctx, int value);

// Get a value from the context
SimpleStatus simple_context_get_value(simple_context_t* ctx, int* out_value);

// Destroy the context
void simple_context_destroy(simple_context_t* ctx);

#endif // SIMPLE_LIB_H
