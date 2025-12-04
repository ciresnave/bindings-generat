#ifndef TEST_ERRORS_H
#define TEST_ERRORS_H

// Test error handling with success enum

// Error enum with success = 0
typedef enum {
    MY_STATUS_SUCCESS = 0,
    MY_STATUS_BAD_PARAM = 1,
    MY_STATUS_OUT_OF_MEMORY = 2,
    MY_STATUS_INTERNAL_ERROR = 3
} MyStatus_t;

// Opaque handle type
typedef struct MyContext_s* MyHandle_t;

// Create function that returns status
MyStatus_t myCreate(MyHandle_t* handle);

// Destroy function that returns status
MyStatus_t myDestroy(MyHandle_t handle);

// Operation that returns status
MyStatus_t myDoSomething(MyHandle_t handle, int value);

#endif // TEST_ERRORS_H
