#pragma once

#include "types.h"
#include "frame_pool.h"
#include <string>

namespace hermivision {

/**
 * Strategy Pattern: Abstract interface for all AI models.
 *
 * Concrete implementations:
 *   - YoloBallDetector (Phase 1 — existing)
 *   - CourtDetector    (Phase 3)
 *   - BounceDetector   (Future)
 *
 * Each model:
 *   1. Loads its TFLite/ONNX model file
 *   2. Configures hardware delegate (NNAPI→GPU→CPU fallback)
 *   3. Acquires frame from FramePool, processes, releases
 *
 * Process pattern (enforced by interface):
 *   1. Acquire frame from pool (pool.acquireLatest())
 *   2. Resize to model-specific input size
 *   3. Preprocess (normalize, etc.)
 *   4. TFLite invoke
 *   5. Postprocess → write results to ctx
 *   6. Release frame (slot->release())
 */
class IAIModel {
public:
    virtual ~IAIModel() = default;

    /// Load model file and configure hardware delegate with auto-fallback
    virtual bool loadModel(const std::string& modelPath, DelegateType delegate, int numThreads) = 0;

    /// Process from FramePool — model acquires frame internally
    virtual void process(FrameContext& ctx, FramePool& pool) = 0;

    /// Release all resources (model, interpreter, delegates, buffers)
    virtual void release() = 0;

    /// Query which delegate was actually applied ("NPU (NNAPI)", "GPU", "CPU (4 threads)")
    virtual std::string getActiveDelegate() const = 0;
};

}