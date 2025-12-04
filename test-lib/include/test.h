// Test type alias detection

typedef struct cudnnContext* cudnnHandle_t;
typedef struct cudnnTensorStruct* cudnnTensorDescriptor_t;
typedef struct cudnnFilterStruct* cudnnFilterDescriptor_t;

// Create/destroy functions
int cudnnCreate(cudnnHandle_t* handle);
int cudnnDestroy(cudnnHandle_t handle);

int cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* desc);
int cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t desc);

int cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t* desc);
int cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t desc);

// Some operations
int cudnnSetTensor4dDescriptor(
    cudnnTensorDescriptor_t desc,
    int format,
    int dataType,
    int n,
    int c,
    int h,
    int w
);
