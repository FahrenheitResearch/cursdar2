#pragma once
#include "../nexrad/sweep_data.h"
#include <cuda_runtime.h>
#include <vector>

namespace gpu_preprocess {

class PreprocessWorkspace {
public:
    PreprocessWorkspace();
    ~PreprocessWorkspace();

    cudaStream_t stream();
    bool ensureGateCapacity(size_t gateCount);
    bool ensureSuppressCapacity(size_t gateCount);
    void reset();

    uint16_t* sourceGates() const { return m_d_source; }
    uint16_t* correctedGates() const { return m_d_corrected; }
    uint8_t* suppressMask() const { return m_d_suppress; }

private:
    cudaStream_t m_stream = nullptr;
    uint16_t* m_d_source = nullptr;
    uint16_t* m_d_corrected = nullptr;
    uint8_t* m_d_suppress = nullptr;
    size_t m_gateCapacity = 0;
    size_t m_suppressCapacity = 0;
};

bool dealiasVelocity(PrecomputedSweep::ProductData& velPd, int numRadials,
                     PreprocessWorkspace* workspace = nullptr);
bool suppressReflectivityRings(std::vector<PrecomputedSweep>& sweeps,
                               PreprocessWorkspace* workspace = nullptr);

} // namespace gpu_preprocess
