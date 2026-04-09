#pragma once

#include "types.h"
#include "tensorflow/lite/c/c_api.h"

namespace hermivision {

/**
 * Manages TFLite hardware delegate lifecycle with auto-fallback.
 *
 * Fallback chain: NNAPI (NPU/DSP) → GPU (OpenCL/GL) → CPU (XNNPACK auto-enabled)
 *
 * Usage:
 *   DelegateManager dm;
 *   TfLiteInterpreter* interp = dm.createInterpreter(model, DelegateType::NNAPI, 4);
 *   // ... run inference ...
 *   dm.release();  // Cleans up delegate + interpreter
 */
class DelegateManager {
public:
    DelegateManager() = default;
    ~DelegateManager();

    /**
     * Create a TFLite interpreter with the best available delegate.
     * Tries the preferred delegate first, falls back automatically.
     *
     * @param model      Loaded TFLite model (caller retains ownership)
     * @param preferred  Preferred delegate from DeviceProfiler benchmark
     * @param numThreads CPU thread count for fallback/CPU mode
     * @return Interpreter pointer, or nullptr on total failure
     */
    TfLiteInterpreter* createInterpreter(
        TfLiteModel* model,
        DelegateType preferred,
        int numThreads
    );

    /// Human-readable name of the delegate that was actually applied
    const std::string& getActiveDelegateName() const { return activeDelegateName_; }

    /// Get the interpreter pointer (for tensor access during inference)
    TfLiteInterpreter* getInterpreter() const { return interpreter_; }

    /// Release interpreter and delegate resources
    void release();

private:
    TfLiteInterpreter* interpreter_ = nullptr;
    TfLiteDelegate* activeDelegate_ = nullptr;
    DelegateType activeDelegateType_ = DelegateType::CPU;
    std::string activeDelegateName_ = "none";

    /// Try creating interpreter with a specific delegate. Returns nullptr on failure.
    TfLiteInterpreter* tryCreate(TfLiteModel* model, DelegateType type, int numThreads);

    /// Clean up a delegate pointer based on its type
    void deleteDelegate(TfLiteDelegate* delegate, DelegateType type);
};

} // namespace hermivision
