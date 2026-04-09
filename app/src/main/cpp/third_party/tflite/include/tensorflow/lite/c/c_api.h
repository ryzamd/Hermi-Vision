/**
 * TFLite C API — Declarations for HermiVision C++ pipeline.
 *
 * Only the functions ACTUALLY USED by our code are declared here.
 * They match the symbols exported by libtensorflowlite_jni.so (TFLite 2.16.1).
 *
 * Types are imported from core/c/c_api_types.h (via c_api_types.h redirect).
 */
#ifndef TENSORFLOW_LITE_C_C_API_H_
#define TENSORFLOW_LITE_C_C_API_H_

#include "c_api_types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Opaque types (only declare if not already declared by core headers) ──
typedef struct TfLiteModel TfLiteModel;
typedef struct TfLiteInterpreter TfLiteInterpreter;
typedef struct TfLiteInterpreterOptions TfLiteInterpreterOptions;
typedef struct TfLiteTensor TfLiteTensor;
// Note: TfLiteDelegate is already declared in core/c/c_api_types.h

// ════════════════════════════════════════════════════════════════
// Model
// ════════════════════════════════════════════════════════════════

/// Load model from a file path on disk.
extern TfLiteModel* TfLiteModelCreateFromFile(const char* model_path);

/// Free model.
extern void TfLiteModelDelete(TfLiteModel* model);

// ════════════════════════════════════════════════════════════════
// Interpreter Options
// ════════════════════════════════════════════════════════════════

/// Create interpreter options.
extern TfLiteInterpreterOptions* TfLiteInterpreterOptionsCreate();

/// Free interpreter options.
extern void TfLiteInterpreterOptionsDelete(
    TfLiteInterpreterOptions* options);

/// Set the number of CPU threads.
extern void TfLiteInterpreterOptionsSetNumThreads(
    TfLiteInterpreterOptions* options,
    int32_t num_threads);

/// Add a delegate to interpreter options (NNAPI, GPU, etc.)
extern void TfLiteInterpreterOptionsAddDelegate(
    TfLiteInterpreterOptions* options,
    TfLiteDelegate* delegate);

// ════════════════════════════════════════════════════════════════
// Interpreter
// ════════════════════════════════════════════════════════════════

/// Create interpreter from model + options.
extern TfLiteInterpreter* TfLiteInterpreterCreate(
    const TfLiteModel* model,
    const TfLiteInterpreterOptions* optional_options);

/// Free interpreter.
extern void TfLiteInterpreterDelete(TfLiteInterpreter* interpreter);

/// Allocate tensors (must call before invoke).
extern TfLiteStatus TfLiteInterpreterAllocateTensors(
    TfLiteInterpreter* interpreter);

/// Run inference.
extern TfLiteStatus TfLiteInterpreterInvoke(
    TfLiteInterpreter* interpreter);

// ════════════════════════════════════════════════════════════════
// Tensors
// ════════════════════════════════════════════════════════════════

/// Get input tensor at index.
extern TfLiteTensor* TfLiteInterpreterGetInputTensor(
    const TfLiteInterpreter* interpreter,
    int32_t input_index);

/// Get output tensor at index (const — read-only).
extern const TfLiteTensor* TfLiteInterpreterGetOutputTensor(
    const TfLiteInterpreter* interpreter,
    int32_t output_index);

/// Copy data INTO input tensor.
extern TfLiteStatus TfLiteTensorCopyFromBuffer(
    TfLiteTensor* tensor,
    const void* input_data,
    size_t input_data_size);

/// Number of dimensions.
extern int32_t TfLiteTensorNumDims(const TfLiteTensor* tensor);

/// Size of dimension at index.
extern int32_t TfLiteTensorDim(const TfLiteTensor* tensor, int32_t dim_index);

/// Raw data pointer (for reading output tensor).
extern void* TfLiteTensorData(const TfLiteTensor* tensor);

/// Byte size of tensor data.
extern size_t TfLiteTensorByteSize(const TfLiteTensor* tensor);

/// Get tensor data type.
extern TfLiteType TfLiteTensorType(const TfLiteTensor* tensor);

#ifdef __cplusplus
}
#endif

#endif  // TENSORFLOW_LITE_C_C_API_H_
