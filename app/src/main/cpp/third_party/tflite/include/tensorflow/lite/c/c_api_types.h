/**
 * TFLite C API Types — Minimal declarations for HermiVision C++ pipeline.
 *
 * These are the stable, public C API types from TensorFlow Lite.
 * They match the symbols exported by libtensorflowlite_jni.so (TFLite 2.16.1).
 *
 * Source: tensorflow/lite/c/c_api_types.h
 */
#ifndef TENSORFLOW_LITE_C_C_API_TYPES_H_
#define TENSORFLOW_LITE_C_C_API_TYPES_H_

#ifdef __cplusplus
extern "C" {
#endif

/// Status codes returned by TFLite C API functions.
typedef enum TfLiteStatus {
    kTfLiteOk = 0,
    kTfLiteError = 1,
    kTfLiteDelegateError = 2,
    kTfLiteApplicationError = 3,
    kTfLiteDelegateDataNotFound = 4,
    kTfLiteDelegateDataWriteError = 5,
    kTfLiteDelegateDataReadError = 6,
    kTfLiteUnresolvedOps = 7,
    kTfLiteCancelled = 8,
} TfLiteStatus;

/// TFLite data types.
typedef enum TfLiteType {
    kTfLiteNoType = 0,
    kTfLiteFloat32 = 1,
    kTfLiteInt32 = 2,
    kTfLiteUInt8 = 3,
    kTfLiteInt64 = 4,
    kTfLiteString = 5,
    kTfLiteBool = 6,
    kTfLiteInt16 = 7,
    kTfLiteComplex64 = 8,
    kTfLiteInt8 = 9,
    kTfLiteFloat16 = 10,
    kTfLiteFloat64 = 11,
} TfLiteType;

#ifdef __cplusplus
}
#endif

#endif  // TENSORFLOW_LITE_C_C_API_TYPES_H_
