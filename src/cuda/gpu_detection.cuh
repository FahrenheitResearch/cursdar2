#pragma once
#include "gpu_tensor.cuh"
#include <cuda_runtime.h>
#include <vector>

namespace gpu_detection {

struct CompactCandidate {
    int radial_idx = 0;
    int gate_idx = 0;
    float score = 0.0f;
    float aux = 0.0f;
};

struct HostDetectionResults {
    int num_radials = 0;
    int num_gates = 0;
    float first_gate_km = 0.0f;
    float gate_spacing_km = 0.0f;
    std::vector<float> azimuths;
    std::vector<CompactCandidate> tds;
    std::vector<CompactCandidate> hail;
    std::vector<CompactCandidate> meso;
};

class CandidateWorkspace {
public:
    CandidateWorkspace();
    ~CandidateWorkspace();

    bool compute(const gpu_tensor::PolarTensor& tensor,
                 HostDetectionResults& out,
                 cudaStream_t stream = 0);
    void reset();

private:
    bool ensureCapacity(size_t cellCount);
    bool ensureOutputCapacity(size_t cellCount);
    bool ensureCountBuffer(int** ptr);

    size_t m_cellCapacity = 0;
    size_t m_outputCapacity = 0;
    uint8_t* m_d_tdsMask = nullptr;
    float* m_d_tdsScore = nullptr;
    uint8_t* m_d_tdsKeep = nullptr;
    CompactCandidate* m_d_tdsCandidates = nullptr;
    int* m_d_tdsCount = nullptr;

    uint8_t* m_d_hailMask = nullptr;
    float* m_d_hailScore = nullptr;
    uint8_t* m_d_hailKeep = nullptr;
    CompactCandidate* m_d_hailCandidates = nullptr;
    int* m_d_hailCount = nullptr;

    uint8_t* m_d_mesoMask = nullptr;
    float* m_d_mesoScore = nullptr;
    float* m_d_mesoDiameter = nullptr;
    uint8_t* m_d_mesoKeep = nullptr;
    CompactCandidate* m_d_mesoCandidates = nullptr;
    int* m_d_mesoCount = nullptr;
};

bool computeDetectionCandidates(const gpu_tensor::PolarTensor& tensor,
                                CandidateWorkspace& workspace,
                                HostDetectionResults& out,
                                cudaStream_t stream = 0);

} // namespace gpu_detection
