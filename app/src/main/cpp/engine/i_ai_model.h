#pragma once

#include "types.h"
#include <string>

namespace hermivision {

/**
 * Strategy Pattern: Abstract interface for all AI models.
 *
 * Concrete implementations:
 *   - YoloBallDetector (Phase 1)
 *   - CourtDetector    (Future)
 *   - BounceDetector   (Future)
 *
 * Each model:
 *   1. Loads its TFLite/ONNX model file
 *   2. Configures hardware delegate (NNAPI→GPU→CPU fallback)
 *   3. Reads from FrameContext, writes results back (zero-copy shared state)
 */
class IAIModel {
public:
    virtual ~IAIModel() = default;

    /// Load model file and configure hardware delegate with auto-fallback
    virtual bool loadModel(const std::string& modelPath, DelegateType delegate, int numThreads) = 0;

    /// Process a single frame — reads/writes FrameContext in-place
    virtual void process(FrameContext& ctx) = 0;

    /// Release all resources (model, interpreter, delegates, buffers)
    virtual void release() = 0;

    /// Query which delegate was actually applied ("NPU (NNAPI)", "GPU", "CPU (4 threads)")
    virtual std::string getActiveDelegate() const = 0;
};

} // namespace hermivision
