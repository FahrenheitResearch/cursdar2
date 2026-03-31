#include "preprocess.cuh"
#include "cuda_common.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

namespace gpu_preprocess {
namespace {

constexpr float kInvalidSample = -9999.0f;

__device__ float decodeVelocity(uint16_t raw, float scale, float offset) {
    if (raw <= 1 || scale == 0.0f)
        return kInvalidSample;
    return ((float)raw - offset) / scale;
}

__device__ uint16_t encodeVelocity(float velocity, float scale, float offset) {
    float rawValue = velocity * scale + offset;
    rawValue = fminf(fmaxf(rawValue, 2.0f), 65535.0f);
    return (uint16_t)lrintf(rawValue);
}

__device__ float bestUnfoldedVelocity(float velocity, float reference, float nyquist) {
    float best = velocity;
    float bestError = fabsf(velocity - reference);
    const float candidates[2] = {
        velocity - 2.0f * nyquist,
        velocity + 2.0f * nyquist
    };
    for (float candidate : candidates) {
        float error = fabsf(candidate - reference);
        if (error < bestError) {
            best = candidate;
            bestError = error;
        }
    }
    return best;
}

__global__ void dealiasVelocityKernel(const uint16_t* __restrict__ source,
                                      uint16_t* __restrict__ corrected,
                                      int nr, int ng,
                                      float scale, float offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nr * ng;
    if (idx >= total) return;

    int gi = idx / nr;
    int ri = idx - gi * nr;
    uint16_t raw = source[idx];
    corrected[idx] = raw;

    const float nyquist = 30.0f;
    const float maxNeighborSpread = nyquist * 0.75f;
    const float minImprovement = nyquist * 0.35f;
    const float maxResidual = nyquist * 0.45f;

    float velocity = decodeVelocity(raw, scale, offset);
    if (velocity <= -998.0f)
        return;

    int riPrev = (ri - 1 + nr) % nr;
    int riNext = (ri + 1) % nr;
    float prevVelocity = decodeVelocity(source[gi * nr + riPrev], scale, offset);
    float nextVelocity = decodeVelocity(source[gi * nr + riNext], scale, offset);
    if (prevVelocity <= -998.0f || nextVelocity <= -998.0f)
        return;
    if (fabsf(prevVelocity - nextVelocity) > maxNeighborSpread)
        return;

    float reference = 0.5f * (prevVelocity + nextVelocity);
    float baseError = fabsf(velocity - reference);
    float unfolded = bestUnfoldedVelocity(velocity, reference, nyquist);
    float unfoldedError = fabsf(unfolded - reference);
    if (unfoldedError >= baseError - minImprovement)
        return;
    if (unfoldedError > maxResidual)
        return;

    corrected[idx] = encodeVelocity(unfolded, scale, offset);
}

__global__ void ringStatsKernel(const uint16_t* __restrict__ gates,
                                int nr, int ng,
                                float scale, float offset,
                                float firstGateKm, float gateSpacingKm,
                                uint8_t* __restrict__ suppress) {
    int gi = blockIdx.x;
    if (gi >= ng) return;

    constexpr float kCoverageStrong = 0.95f;
    constexpr float kCoverageLoose = 0.88f;
    constexpr float kStdStrong = 12.0f;
    constexpr float kStdLoose = 6.0f;
    constexpr float kMaxRangeKm = 160.0f;

    __shared__ int validShared[256];
    __shared__ float sumShared[256];
    __shared__ float sum2Shared[256];

    int tid = threadIdx.x;
    int valid = 0;
    float sum = 0.0f;
    float sum2 = 0.0f;

    float rangeKm = firstGateKm + gi * gateSpacingKm;
    if (rangeKm <= kMaxRangeKm) {
        for (int ri = tid; ri < nr; ri += blockDim.x) {
            uint16_t raw = gates[(size_t)gi * nr + ri];
            if (raw <= 1) continue;
            float value = ((float)raw - offset) / scale;
            valid++;
            sum += value;
            sum2 += value * value;
        }
    }

    validShared[tid] = valid;
    sumShared[tid] = sum;
    sum2Shared[tid] = sum2;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            validShared[tid] += validShared[tid + stride];
            sumShared[tid] += sumShared[tid + stride];
            sum2Shared[tid] += sum2Shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid != 0)
        return;

    suppress[gi] = 0;
    if (rangeKm > kMaxRangeKm)
        return;

    int totalValid = validShared[0];
    if (totalValid < nr / 2)
        return;

    float coverage = (float)totalValid / (float)nr;
    float mean = sumShared[0] / (float)totalValid;
    float variance = fmaxf(sum2Shared[0] / (float)totalValid - mean * mean, 0.0f);
    float stddev = sqrtf(variance);
    bool strongRing = coverage >= kCoverageStrong && mean >= 10.0f && stddev <= kStdStrong;
    bool looseRing = coverage >= kCoverageLoose && mean >= 20.0f && stddev <= kStdLoose;
    suppress[gi] = (strongRing || looseRing) ? 1 : 0;
}

__global__ void zeroSuppressedGatesKernel(uint16_t* gates,
                                          int nr, int ng,
                                          const uint8_t* __restrict__ suppress) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nr * ng;
    if (idx >= total) return;
    int gi = idx / nr;
    if (suppress[gi])
        gates[idx] = 0;
}

template <typename T>
bool ensureCapacity(T** ptr, size_t* capacity, size_t requiredCount) {
    if (*capacity >= requiredCount)
        return true;

    size_t newCapacity = (*capacity > 0) ? *capacity : 1;
    while (newCapacity < requiredCount)
        newCapacity = std::max(newCapacity + newCapacity / 2, requiredCount);

    if (*ptr && cudaFree(*ptr) != cudaSuccess)
        return false;
    *ptr = nullptr;
    if (cudaMalloc(ptr, newCapacity * sizeof(T)) != cudaSuccess)
        return false;

    *capacity = newCapacity;
    return true;
}

} // namespace

PreprocessWorkspace::PreprocessWorkspace() = default;

PreprocessWorkspace::~PreprocessWorkspace() {
    reset();
}

cudaStream_t PreprocessWorkspace::stream() {
    if (!m_stream)
        cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking);
    return m_stream;
}

bool PreprocessWorkspace::ensureGateCapacity(size_t gateCount) {
    if (m_gateCapacity >= gateCount && m_d_source && m_d_corrected)
        return true;

    size_t newCapacity = (m_gateCapacity > 0) ? m_gateCapacity : 1;
    while (newCapacity < gateCount)
        newCapacity = std::max(newCapacity + newCapacity / 2, gateCount);

    if (m_d_source && cudaFree(m_d_source) != cudaSuccess)
        return false;
    if (m_d_corrected && cudaFree(m_d_corrected) != cudaSuccess)
        return false;

    m_d_source = nullptr;
    m_d_corrected = nullptr;
    if (cudaMalloc(&m_d_source, newCapacity * sizeof(uint16_t)) != cudaSuccess)
        return false;
    if (cudaMalloc(&m_d_corrected, newCapacity * sizeof(uint16_t)) != cudaSuccess) {
        cudaFree(m_d_source);
        m_d_source = nullptr;
        return false;
    }

    m_gateCapacity = newCapacity;
    return true;
}

bool PreprocessWorkspace::ensureSuppressCapacity(size_t gateCount) {
    return ensureCapacity(&m_d_suppress, &m_suppressCapacity, gateCount);
}

void PreprocessWorkspace::reset() {
    if (m_d_suppress) cudaFree(m_d_suppress);
    if (m_d_corrected) cudaFree(m_d_corrected);
    if (m_d_source) cudaFree(m_d_source);
    if (m_stream) cudaStreamDestroy(m_stream);
    m_d_suppress = nullptr;
    m_d_corrected = nullptr;
    m_d_source = nullptr;
    m_stream = nullptr;
    m_gateCapacity = 0;
    m_suppressCapacity = 0;
}

bool dealiasVelocity(PrecomputedSweep::ProductData& velPd, int numRadials,
                     PreprocessWorkspace* workspace) {
    if (!velPd.has_data || velPd.num_gates <= 0 || numRadials <= 2 || velPd.gates.empty())
        return true;

    PreprocessWorkspace ownedWorkspace;
    PreprocessWorkspace* activeWorkspace = workspace ? workspace : &ownedWorkspace;
    cudaStream_t stream = activeWorkspace->stream();
    if (!stream)
        return false;

    const size_t bytes = velPd.gates.size() * sizeof(uint16_t);
    bool ok = false;
    int total = numRadials * velPd.num_gates;
    int block = 256;
    int grid = (total + block - 1) / block;

    if (!activeWorkspace->ensureGateCapacity(velPd.gates.size()))
        return false;

    if (cudaMemcpyAsync(activeWorkspace->sourceGates(), velPd.gates.data(), bytes,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        goto cleanup;
    }

    dealiasVelocityKernel<<<grid, block, 0, stream>>>(
        activeWorkspace->sourceGates(), activeWorkspace->correctedGates(),
        numRadials, velPd.num_gates,
        velPd.scale, velPd.offset);
    if (cudaGetLastError() != cudaSuccess)
        goto cleanup;

    if (cudaMemcpyAsync(velPd.gates.data(), activeWorkspace->correctedGates(), bytes,
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        goto cleanup;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess)
        goto cleanup;

    ok = true;

cleanup:
    return ok;
}

bool suppressReflectivityRings(std::vector<PrecomputedSweep>& sweeps,
                               PreprocessWorkspace* workspace) {
    bool allOk = true;
    PreprocessWorkspace ownedWorkspace;
    PreprocessWorkspace* activeWorkspace = workspace ? workspace : &ownedWorkspace;
    cudaStream_t stream = activeWorkspace->stream();
    if (!stream)
        return false;

    for (auto& sweep : sweeps) {
        auto& pd = sweep.products[PROD_REF];
        if (!pd.has_data || pd.num_gates <= 0 || sweep.num_radials < 300 || pd.gates.empty())
            continue;

        const size_t gateBytes = pd.gates.size() * sizeof(uint16_t);
        const size_t gateCount = pd.gates.size();
        if (!activeWorkspace->ensureGateCapacity(gateCount) ||
            !activeWorkspace->ensureSuppressCapacity((size_t)pd.num_gates)) {
            allOk = false;
            continue;
        }

        bool ok = false;
        int total = sweep.num_radials * pd.num_gates;
        int block = 256;
        int grid = (total + block - 1) / block;

        if (cudaMemcpyAsync(activeWorkspace->sourceGates(), pd.gates.data(), gateBytes,
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            goto cleanup;
        }

        ringStatsKernel<<<pd.num_gates, 256, 0, stream>>>(
            activeWorkspace->sourceGates(), sweep.num_radials, pd.num_gates,
            pd.scale, pd.offset, pd.first_gate_km, pd.gate_spacing_km,
            activeWorkspace->suppressMask());
        if (cudaGetLastError() != cudaSuccess)
            goto cleanup;

        zeroSuppressedGatesKernel<<<grid, block, 0, stream>>>(
            activeWorkspace->sourceGates(), sweep.num_radials, pd.num_gates,
            activeWorkspace->suppressMask());
        if (cudaGetLastError() != cudaSuccess)
            goto cleanup;

        if (cudaMemcpyAsync(pd.gates.data(), activeWorkspace->sourceGates(), gateBytes,
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
            goto cleanup;
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess)
            goto cleanup;

        ok = true;

cleanup:
        if (!ok)
            allOk = false;
    }

    return allOk;
}

} // namespace gpu_preprocess
