#include "gpu_detection.cuh"
#include "cuda_common.cuh"

#include <cmath>

namespace gpu_detection {
namespace {

constexpr float kPosInf = 1.0e30f;
constexpr float kNegInf = -1.0e30f;

inline bool cudaSuccessOnly(cudaError_t err) {
    return err == cudaSuccess;
}

template <typename T>
bool reallocBuffer(T*& ptr, size_t count) {
    if (count == 0)
        return true;
    if (ptr && !cudaSuccessOnly(cudaFree(ptr)))
        return false;
    ptr = nullptr;
    return cudaSuccessOnly(cudaMalloc(&ptr, count * sizeof(T)));
}

template <typename T>
bool copyDeviceVector(std::vector<T>& out, const T* src, size_t count) {
    out.clear();
    if (count == 0)
        return true;
    out.resize(count);
    return cudaSuccessOnly(cudaMemcpy(out.data(), src, count * sizeof(T), cudaMemcpyDeviceToHost));
}

__global__ void tdsCandidateKernel(const float* __restrict__ ref,
                                   const float* __restrict__ ref_mask,
                                   const float* __restrict__ zdr,
                                   const float* __restrict__ zdr_mask,
                                   const float* __restrict__ cc,
                                   const float* __restrict__ cc_mask,
                                   int num_radials,
                                   int num_gates,
                                   float first_gate_km,
                                   float gate_spacing_km,
                                   uint8_t* __restrict__ mask,
                                   float* __restrict__ score) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_radials * num_gates;
    if (idx >= total)
        return;

    int gate_idx = idx / num_radials;
    float range_km = first_gate_km + gate_idx * gate_spacing_km;
    mask[idx] = 0;
    score[idx] = kPosInf;
    if ((gate_idx & 1) != 0 || range_km < 15.0f || range_km > 120.0f)
        return;

    if (cc_mask[idx] <= 0.5f || ref_mask[idx] <= 0.5f || zdr_mask[idx] <= 0.5f)
        return;
    float cc_v = cc[idx];
    if (cc_v < 0.55f || cc_v > 0.82f)
        return;
    if (fabsf(zdr[idx]) > 1.25f || ref[idx] < 40.0f)
        return;

    mask[idx] = 1;
    score[idx] = cc_v;
}

__global__ void hailCandidateKernel(const float* __restrict__ ref,
                                    const float* __restrict__ ref_mask,
                                    const float* __restrict__ zdr,
                                    const float* __restrict__ zdr_mask,
                                    int num_radials,
                                    int num_gates,
                                    float first_gate_km,
                                    float gate_spacing_km,
                                    uint8_t* __restrict__ mask,
                                    float* __restrict__ score) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_radials * num_gates;
    if (idx >= total)
        return;

    int gate_idx = idx / num_radials;
    float range_km = first_gate_km + gate_idx * gate_spacing_km;
    mask[idx] = 0;
    score[idx] = kNegInf;
    if ((gate_idx & 1) != 0 || range_km < 15.0f || range_km > 180.0f)
        return;

    if (ref_mask[idx] <= 0.5f || zdr_mask[idx] <= 0.5f)
        return;
    if (ref[idx] < 55.0f)
        return;

    float hdr = ref[idx] - (19.0f * fmaxf(zdr[idx], 0.0f) + 27.0f);
    if (hdr < 10.0f)
        return;

    mask[idx] = 1;
    score[idx] = hdr;
}

__device__ bool passesMesoGate(const float* vel,
                               const float* vel_mask,
                               int nr,
                               int ng,
                               int gate_idx,
                               int radial_idx,
                               float range_km,
                               float* shear_out,
                               float* span_out) {
    if (gate_idx < 0 || gate_idx >= ng)
        return false;

    constexpr int span = 2;
    int ri_lo = (radial_idx - span + nr) % nr;
    int ri_hi = (radial_idx + span) % nr;
    if (vel_mask[gate_idx * nr + ri_lo] <= 0.5f || vel_mask[gate_idx * nr + ri_hi] <= 0.5f)
        return false;
    float v_lo = vel[gate_idx * nr + ri_lo];
    float v_hi = vel[gate_idx * nr + ri_hi];
    if (fabsf(v_lo) < 12.0f || fabsf(v_hi) < 12.0f)
        return false;
    if (v_lo * v_hi >= 0.0f)
        return false;

    float shear_ms = fabsf(v_hi - v_lo);
    if (shear_ms < 40.0f)
        return false;

    float az_span_deg = span * 2.0f * (360.0f / nr);
    float az_span_km = range_km * az_span_deg * 0.01745329251994329577f;
    if (az_span_km < 1.0f || az_span_km > 10.0f)
        return false;

    *shear_out = shear_ms;
    *span_out = az_span_km;
    return true;
}

__global__ void mesoCandidateKernel(const float* __restrict__ ref,
                                    const float* __restrict__ ref_mask,
                                    const float* __restrict__ vel,
                                    const float* __restrict__ vel_mask,
                                    bool has_ref,
                                    int num_radials,
                                    int num_gates,
                                    float first_gate_km,
                                    float gate_spacing_km,
                                    uint8_t* __restrict__ mask,
                                    float* __restrict__ score,
                                    float* __restrict__ diameter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_radials * num_gates;
    if (idx >= total)
        return;

    int gate_idx = idx / num_radials;
    int radial_idx = idx - gate_idx * num_radials;
    float range_km = first_gate_km + gate_idx * gate_spacing_km;
    mask[idx] = 0;
    score[idx] = kNegInf;
    diameter[idx] = 0.0f;

    if (gate_idx < 12 || gate_idx >= num_gates - 12 || (gate_idx % 4) != 0 || (radial_idx & 1) != 0)
        return;
    if (range_km < 20.0f || range_km > 120.0f)
        return;

    if (has_ref) {
        if (ref_mask[idx] <= 0.5f || ref[idx] < 35.0f)
            return;
    }

    float shear_ms = 0.0f;
    float az_span_km = 0.0f;
    if (!passesMesoGate(vel, vel_mask, num_radials, num_gates, gate_idx, radial_idx, range_km,
                        &shear_ms, &az_span_km)) {
        return;
    }

    int gate_support = 0;
    for (int dgi = -2; dgi <= 2; ++dgi) {
        int ngi = gate_idx + dgi;
        if (ngi < 0 || ngi >= num_gates)
            continue;
        float neighbor_shear = 0.0f;
        float neighbor_span = 0.0f;
        float neighbor_range = first_gate_km + ngi * gate_spacing_km;
        if (passesMesoGate(vel, vel_mask, num_radials, num_gates, ngi, radial_idx, neighbor_range,
                           &neighbor_shear, &neighbor_span)) {
            ++gate_support;
        }
    }
    if (gate_support < 3)
        return;

    mask[idx] = 1;
    score[idx] = shear_ms;
    diameter[idx] = az_span_km;
}

__global__ void supportExtremumKernel(const uint8_t* __restrict__ mask,
                                      const float* __restrict__ score,
                                      int num_radials,
                                      int num_gates,
                                      int radial_radius,
                                      int gate_radius,
                                      int min_support,
                                      bool lower_is_better,
                                      uint8_t* __restrict__ keep) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_radials * num_gates;
    if (idx >= total)
        return;

    if (!mask[idx]) {
        keep[idx] = 0;
        return;
    }

    int gate_idx = idx / num_radials;
    int radial_idx = idx - gate_idx * num_radials;
    float center = score[idx];
    int support = 0;

    for (int dgi = -gate_radius; dgi <= gate_radius; ++dgi) {
        int ngi = gate_idx + dgi;
        if (ngi < 0 || ngi >= num_gates)
            continue;
        for (int dri = -radial_radius; dri <= radial_radius; ++dri) {
            int nri = (radial_idx + dri + num_radials) % num_radials;
            size_t nidx = (size_t)ngi * (size_t)num_radials + (size_t)nri;
            if (!mask[nidx])
                continue;
            ++support;
            if ((int)nidx == idx)
                continue;

            float neighbor = score[nidx];
            if (lower_is_better) {
                if (neighbor < center - 0.01f) {
                    keep[idx] = 0;
                    return;
                }
            } else if (neighbor > center + 0.01f) {
                keep[idx] = 0;
                return;
            }
        }
    }

    keep[idx] = support >= min_support ? 1 : 0;
}

__global__ void compactCandidatesKernel(const uint8_t* __restrict__ keep,
                                        const float* __restrict__ score,
                                        const float* __restrict__ aux,
                                        int num_radials,
                                        int num_gates,
                                        CompactCandidate* __restrict__ out,
                                        int* __restrict__ out_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_radials * num_gates;
    if (idx >= total || !keep[idx])
        return;

    int slot = atomicAdd(out_count, 1);
    int gate_idx = idx / num_radials;
    int radial_idx = idx - gate_idx * num_radials;
    out[slot].radial_idx = radial_idx;
    out[slot].gate_idx = gate_idx;
    out[slot].score = score[idx];
    out[slot].aux = aux ? aux[idx] : 0.0f;
}

} // namespace

CandidateWorkspace::CandidateWorkspace() = default;

CandidateWorkspace::~CandidateWorkspace() {
    reset();
}

bool CandidateWorkspace::ensureCountBuffer(int** ptr) {
    if (*ptr)
        return true;
    return cudaSuccessOnly(cudaMalloc(ptr, sizeof(int)));
}

bool CandidateWorkspace::ensureCapacity(size_t cellCount) {
    if (cellCount <= m_cellCapacity)
        return true;

    if (!reallocBuffer(m_d_tdsMask, cellCount) ||
        !reallocBuffer(m_d_tdsScore, cellCount) ||
        !reallocBuffer(m_d_tdsKeep, cellCount) ||
        !reallocBuffer(m_d_hailMask, cellCount) ||
        !reallocBuffer(m_d_hailScore, cellCount) ||
        !reallocBuffer(m_d_hailKeep, cellCount) ||
        !reallocBuffer(m_d_mesoMask, cellCount) ||
        !reallocBuffer(m_d_mesoScore, cellCount) ||
        !reallocBuffer(m_d_mesoDiameter, cellCount) ||
        !reallocBuffer(m_d_mesoKeep, cellCount)) {
        reset();
        return false;
    }

    m_cellCapacity = cellCount;
    return true;
}

bool CandidateWorkspace::ensureOutputCapacity(size_t cellCount) {
    if (cellCount <= m_outputCapacity)
        return true;

    if (!reallocBuffer(m_d_tdsCandidates, cellCount) ||
        !reallocBuffer(m_d_hailCandidates, cellCount) ||
        !reallocBuffer(m_d_mesoCandidates, cellCount)) {
        reset();
        return false;
    }

    m_outputCapacity = cellCount;
    return true;
}

void CandidateWorkspace::reset() {
    if (m_d_mesoCount) cudaFree(m_d_mesoCount);
    if (m_d_mesoCandidates) cudaFree(m_d_mesoCandidates);
    if (m_d_mesoKeep) cudaFree(m_d_mesoKeep);
    if (m_d_mesoDiameter) cudaFree(m_d_mesoDiameter);
    if (m_d_mesoScore) cudaFree(m_d_mesoScore);
    if (m_d_mesoMask) cudaFree(m_d_mesoMask);
    if (m_d_hailCount) cudaFree(m_d_hailCount);
    if (m_d_hailCandidates) cudaFree(m_d_hailCandidates);
    if (m_d_hailKeep) cudaFree(m_d_hailKeep);
    if (m_d_hailScore) cudaFree(m_d_hailScore);
    if (m_d_hailMask) cudaFree(m_d_hailMask);
    if (m_d_tdsCount) cudaFree(m_d_tdsCount);
    if (m_d_tdsCandidates) cudaFree(m_d_tdsCandidates);
    if (m_d_tdsKeep) cudaFree(m_d_tdsKeep);
    if (m_d_tdsScore) cudaFree(m_d_tdsScore);
    if (m_d_tdsMask) cudaFree(m_d_tdsMask);

    m_d_tdsMask = nullptr;
    m_d_tdsScore = nullptr;
    m_d_tdsKeep = nullptr;
    m_d_tdsCandidates = nullptr;
    m_d_tdsCount = nullptr;
    m_d_hailMask = nullptr;
    m_d_hailScore = nullptr;
    m_d_hailKeep = nullptr;
    m_d_hailCandidates = nullptr;
    m_d_hailCount = nullptr;
    m_d_mesoMask = nullptr;
    m_d_mesoScore = nullptr;
    m_d_mesoDiameter = nullptr;
    m_d_mesoKeep = nullptr;
    m_d_mesoCandidates = nullptr;
    m_d_mesoCount = nullptr;
    m_cellCapacity = 0;
    m_outputCapacity = 0;
}

bool CandidateWorkspace::compute(const gpu_tensor::PolarTensor& tensor,
                                 HostDetectionResults& out,
                                 cudaStream_t stream) {
    out = {};
    if (tensor.spec.num_radials <= 0 || tensor.spec.num_gates <= 0 || !tensor.d_target_azimuths)
        return false;

    const size_t cellCount = tensor.cell_count;
    if (!ensureCapacity(cellCount) ||
        !ensureOutputCapacity(cellCount) ||
        !ensureCountBuffer(&m_d_tdsCount) ||
        !ensureCountBuffer(&m_d_hailCount) ||
        !ensureCountBuffer(&m_d_mesoCount)) {
        return false;
    }

    out.num_radials = tensor.spec.num_radials;
    out.num_gates = tensor.spec.num_gates;
    out.first_gate_km = tensor.spec.first_gate_km;
    out.gate_spacing_km = tensor.spec.gate_spacing_km;

    const int zero = 0;
    if (!cudaSuccessOnly(cudaMemcpyAsync(m_d_tdsCount, &zero, sizeof(int), cudaMemcpyHostToDevice, stream)) ||
        !cudaSuccessOnly(cudaMemcpyAsync(m_d_hailCount, &zero, sizeof(int), cudaMemcpyHostToDevice, stream)) ||
        !cudaSuccessOnly(cudaMemcpyAsync(m_d_mesoCount, &zero, sizeof(int), cudaMemcpyHostToDevice, stream))) {
        return false;
    }

    const bool hasRef = (tensor.valid_product_mask & (1u << gpu_tensor::SLOT_REF)) != 0;
    const bool hasVel = (tensor.valid_product_mask & (1u << gpu_tensor::SLOT_VEL)) != 0;
    const bool hasZdr = (tensor.valid_product_mask & (1u << gpu_tensor::SLOT_ZDR)) != 0;
    const bool hasCc = (tensor.valid_product_mask & (1u << gpu_tensor::SLOT_CC)) != 0;
    const float* ref = tensor.channel(gpu_tensor::CHANNEL_REF);
    const float* vel = tensor.channel(gpu_tensor::CHANNEL_VEL);
    const float* zdr = tensor.channel(gpu_tensor::CHANNEL_ZDR);
    const float* cc = tensor.channel(gpu_tensor::CHANNEL_CC);
    const float* refMask = tensor.channel(gpu_tensor::CHANNEL_MASK_REF);
    const float* velMask = tensor.channel(gpu_tensor::CHANNEL_MASK_VEL);
    const float* zdrMask = tensor.channel(gpu_tensor::CHANNEL_MASK_ZDR);
    const float* ccMask = tensor.channel(gpu_tensor::CHANNEL_MASK_CC);

    const int block = 256;
    const int grid = (int)((cellCount + block - 1) / block);

    if (hasRef && hasZdr && hasCc) {
        tdsCandidateKernel<<<grid, block, 0, stream>>>(
            ref, refMask, zdr, zdrMask, cc, ccMask,
            tensor.spec.num_radials, tensor.spec.num_gates,
            tensor.spec.first_gate_km, tensor.spec.gate_spacing_km,
            m_d_tdsMask, m_d_tdsScore);
        supportExtremumKernel<<<grid, block, 0, stream>>>(
            m_d_tdsMask, m_d_tdsScore,
            tensor.spec.num_radials, tensor.spec.num_gates,
            2, 2, 6, true, m_d_tdsKeep);
        compactCandidatesKernel<<<grid, block, 0, stream>>>(
            m_d_tdsKeep, m_d_tdsScore, nullptr,
            tensor.spec.num_radials, tensor.spec.num_gates,
            m_d_tdsCandidates, m_d_tdsCount);
        if (cudaGetLastError() != cudaSuccess)
            return false;
    }

    if (hasRef && hasZdr) {
        hailCandidateKernel<<<grid, block, 0, stream>>>(
            ref, refMask, zdr, zdrMask,
            tensor.spec.num_radials, tensor.spec.num_gates,
            tensor.spec.first_gate_km, tensor.spec.gate_spacing_km,
            m_d_hailMask, m_d_hailScore);
        supportExtremumKernel<<<grid, block, 0, stream>>>(
            m_d_hailMask, m_d_hailScore,
            tensor.spec.num_radials, tensor.spec.num_gates,
            2, 2, 5, false, m_d_hailKeep);
        compactCandidatesKernel<<<grid, block, 0, stream>>>(
            m_d_hailKeep, m_d_hailScore, nullptr,
            tensor.spec.num_radials, tensor.spec.num_gates,
            m_d_hailCandidates, m_d_hailCount);
        if (cudaGetLastError() != cudaSuccess)
            return false;
    }

    if (hasVel) {
        mesoCandidateKernel<<<grid, block, 0, stream>>>(
            ref, refMask, vel, velMask, hasRef,
            tensor.spec.num_radials, tensor.spec.num_gates,
            tensor.spec.first_gate_km, tensor.spec.gate_spacing_km,
            m_d_mesoMask, m_d_mesoScore, m_d_mesoDiameter);
        supportExtremumKernel<<<grid, block, 0, stream>>>(
            m_d_mesoMask, m_d_mesoScore,
            tensor.spec.num_radials, tensor.spec.num_gates,
            2, 1, 3, false, m_d_mesoKeep);
        compactCandidatesKernel<<<grid, block, 0, stream>>>(
            m_d_mesoKeep, m_d_mesoScore, m_d_mesoDiameter,
            tensor.spec.num_radials, tensor.spec.num_gates,
            m_d_mesoCandidates, m_d_mesoCount);
        if (cudaGetLastError() != cudaSuccess)
            return false;
    }

    if (!cudaSuccessOnly(cudaStreamSynchronize(stream)))
        return false;

    int tdsCount = 0;
    int hailCount = 0;
    int mesoCount = 0;
    if (!cudaSuccessOnly(cudaMemcpy(&tdsCount, m_d_tdsCount, sizeof(int), cudaMemcpyDeviceToHost)) ||
        !cudaSuccessOnly(cudaMemcpy(&hailCount, m_d_hailCount, sizeof(int), cudaMemcpyDeviceToHost)) ||
        !cudaSuccessOnly(cudaMemcpy(&mesoCount, m_d_mesoCount, sizeof(int), cudaMemcpyDeviceToHost)) ||
        !copyDeviceVector(out.azimuths, tensor.d_target_azimuths, (size_t)tensor.spec.num_radials) ||
        !copyDeviceVector(out.tds, m_d_tdsCandidates, (size_t)tdsCount) ||
        !copyDeviceVector(out.hail, m_d_hailCandidates, (size_t)hailCount) ||
        !copyDeviceVector(out.meso, m_d_mesoCandidates, (size_t)mesoCount)) {
        return false;
    }

    return true;
}

bool computeDetectionCandidates(const gpu_tensor::PolarTensor& tensor,
                                CandidateWorkspace& workspace,
                                HostDetectionResults& out,
                                cudaStream_t stream) {
    return workspace.compute(tensor, out, stream);
}

} // namespace gpu_detection
