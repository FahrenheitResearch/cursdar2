#include "gpu_tensor.cuh"
#include "cuda_common.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace gpu_tensor {
namespace {

struct KernelSweepInput {
    const float* azimuths = nullptr;
    const uint16_t* gates = nullptr;
    int num_radials = 0;
    int num_gates = 0;
    float first_gate_km = 0.0f;
    float gate_spacing_km = 0.0f;
    float scale = 0.0f;
    float offset = 0.0f;
    int valid = 0;
};

constexpr float kInvalidSample = -9999.0f;

inline bool cudaSuccessOnly(cudaError_t err) {
    return err == cudaSuccess;
}

__device__ float wrapAzimuth(float azimuth) {
    float wrapped = fmodf(azimuth, 360.0f);
    if (wrapped < 0.0f)
        wrapped += 360.0f;
    return wrapped;
}

__device__ float decodeGate(uint16_t raw, float scale, float offset) {
    if (raw <= GATE_RANGE_FOLDED || scale == 0.0f)
        return kInvalidSample;
    return (((float)raw) - offset) / scale;
}

__device__ bool sampleRange(const KernelSweepInput& input,
                            int radialIndex,
                            int gateLo, int gateHi, float gateWeight,
                            bool linearRange,
                            float* outValue) {
    if (radialIndex < 0 || radialIndex >= input.num_radials ||
        gateLo < 0 || gateLo >= input.num_gates) {
        return false;
    }

    const size_t baseLo = (size_t)gateLo * (size_t)input.num_radials + (size_t)radialIndex;
    const float valueLo = decodeGate(input.gates[baseLo], input.scale, input.offset);
    const bool validLo = valueLo > kInvalidSample * 0.5f;

    if (!linearRange || gateHi < 0 || gateHi >= input.num_gates || gateHi == gateLo) {
        if (!validLo)
            return false;
        *outValue = valueLo;
        return true;
    }

    const size_t baseHi = (size_t)gateHi * (size_t)input.num_radials + (size_t)radialIndex;
    const float valueHi = decodeGate(input.gates[baseHi], input.scale, input.offset);
    const bool validHi = valueHi > kInvalidSample * 0.5f;

    if (validLo && validHi) {
        *outValue = valueLo + (valueHi - valueLo) * gateWeight;
        return true;
    }

    if (validLo && (!validHi || gateWeight <= 0.5f)) {
        *outValue = valueLo;
        return true;
    }

    if (validHi) {
        *outValue = valueHi;
        return true;
    }

    return false;
}

__global__ void generateUniformAzimuthsKernel(float* outAzimuths,
                                              int count,
                                              float offsetDeg) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    const float step = (count > 0) ? (360.0f / (float)count) : 0.0f;
    outAzimuths[idx] = wrapAzimuth(offsetDeg + step * (float)idx);
}

__global__ void buildRadialMapKernel(const float* sourceAzimuths,
                                     int numSourceRadials,
                                     const float* targetAzimuths,
                                     int numTargetRadials,
                                     int* outLo,
                                     int* outHi,
                                     float* outWeight,
                                     bool linearAzimuth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTargetRadials)
        return;

    if (numSourceRadials <= 0 || !sourceAzimuths) {
        outLo[idx] = -1;
        outHi[idx] = -1;
        outWeight[idx] = 0.0f;
        return;
    }

    if (numSourceRadials == 1) {
        outLo[idx] = 0;
        outHi[idx] = 0;
        outWeight[idx] = 0.0f;
        return;
    }

    const float target = wrapAzimuth(targetAzimuths[idx]);
    const float first = wrapAzimuth(sourceAzimuths[0]);
    const float last = wrapAzimuth(sourceAzimuths[numSourceRadials - 1]);

    int lo = 0;
    int hi = 0;
    float weight = 0.0f;

    if (target < first || target > last) {
        lo = numSourceRadials - 1;
        hi = 0;
        const float loAz = last;
        float hiAz = first + 360.0f;
        float adjustedTarget = target;
        if (adjustedTarget < first)
            adjustedTarget += 360.0f;
        const float denom = hiAz - loAz;
        weight = (denom > 1.0e-5f) ? (adjustedTarget - loAz) / denom : 0.0f;
    } else {
        int left = 0;
        int right = numSourceRadials - 1;
        while (left + 1 < right) {
            int mid = left + (right - left) / 2;
            if (sourceAzimuths[mid] <= target) {
                left = mid;
            } else {
                right = mid;
            }
        }

        lo = left;
        hi = right;
        const float loAz = sourceAzimuths[lo];
        const float hiAz = sourceAzimuths[hi];
        const float denom = hiAz - loAz;
        weight = (denom > 1.0e-5f) ? (target - loAz) / denom : 0.0f;
    }

    weight = fminf(fmaxf(weight, 0.0f), 1.0f);
    if (!linearAzimuth) {
        if (weight > 0.5f) {
            lo = hi;
        } else {
            hi = lo;
        }
        weight = 0.0f;
    }

    outLo[idx] = lo;
    outHi[idx] = hi;
    outWeight[idx] = weight;
}

__global__ void buildGateMapKernel(int numSourceGates,
                                   float sourceFirstGateKm,
                                   float sourceGateSpacingKm,
                                   int numTargetGates,
                                   float targetFirstGateKm,
                                   float targetGateSpacingKm,
                                   int* outLo,
                                   int* outHi,
                                   float* outWeight,
                                   bool linearRange) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTargetGates)
        return;

    if (numSourceGates <= 0 || sourceGateSpacingKm <= 0.0f) {
        outLo[idx] = -1;
        outHi[idx] = -1;
        outWeight[idx] = 0.0f;
        return;
    }

    const float targetRangeKm = targetFirstGateKm + targetGateSpacingKm * (float)idx;
    const float sourcePos = (targetRangeKm - sourceFirstGateKm) / sourceGateSpacingKm;
    if (sourcePos < 0.0f || sourcePos > (float)(numSourceGates - 1)) {
        outLo[idx] = -1;
        outHi[idx] = -1;
        outWeight[idx] = 0.0f;
        return;
    }

    if (!linearRange || numSourceGates == 1) {
        const int nearest = max(0, min(numSourceGates - 1, (int)lrintf(sourcePos)));
        outLo[idx] = nearest;
        outHi[idx] = nearest;
        outWeight[idx] = 0.0f;
        return;
    }

    int gateLo = (int)floorf(sourcePos);
    int gateHi = min(gateLo + 1, numSourceGates - 1);
    float weight = sourcePos - (float)gateLo;

    if (gateLo == gateHi)
        weight = 0.0f;

    outLo[idx] = gateLo;
    outHi[idx] = gateHi;
    outWeight[idx] = fminf(fmaxf(weight, 0.0f), 1.0f);
}

__global__ void buildTensorKernel(const KernelSweepInput* inputs,
                                  const int* const* radialLoPtrs,
                                  const int* const* radialHiPtrs,
                                  const float* const* radialWeightPtrs,
                                  const int* const* gateLoPtrs,
                                  const int* const* gateHiPtrs,
                                  const float* const* gateWeightPtrs,
                                  TensorSpec spec,
                                  float* outTensor,
                                  uint32_t validProductMask) {
    const int radialIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int gateIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (radialIndex >= spec.num_radials || gateIndex >= spec.num_gates)
        return;

    const size_t cellIndex = (size_t)gateIndex * (size_t)spec.num_radials + (size_t)radialIndex;
    const size_t channelStride = (size_t)spec.num_radials * (size_t)spec.num_gates;
    const bool linearAzimuth = (spec.flags & TENSOR_INTERP_AZIMUTH_LINEAR) != 0;
    const bool linearRange = (spec.flags & TENSOR_INTERP_RANGE_LINEAR) != 0;

    for (int slot = 0; slot < NUM_TENSOR_PRODUCTS; ++slot) {
        const size_t valueOffset = (size_t)slot * channelStride + cellIndex;
        const size_t maskOffset = (size_t)maskChannelForSlot(slot) * channelStride + cellIndex;
        outTensor[valueOffset] = 0.0f;
        outTensor[maskOffset] = 0.0f;

        if ((validProductMask & (1u << slot)) == 0)
            continue;

        const KernelSweepInput input = inputs[slot];
        const int gateLo = gateLoPtrs[slot][gateIndex];
        if (!input.valid || gateLo < 0)
            continue;

        const int gateHi = gateHiPtrs[slot][gateIndex];
        const float gateWeight = gateWeightPtrs[slot][gateIndex];
        const int radialLo = radialLoPtrs[slot][radialIndex];
        const int radialHi = radialHiPtrs[slot][radialIndex];
        const float radialWeight = radialWeightPtrs[slot][radialIndex];

        float loValue = 0.0f;
        float hiValue = 0.0f;
        const bool validLo = sampleRange(input, radialLo, gateLo, gateHi, gateWeight,
                                         linearRange, &loValue);
        const bool validHi = sampleRange(input, radialHi, gateLo, gateHi, gateWeight,
                                         linearRange, &hiValue);

        float outValue = 0.0f;
        bool valid = false;
        if (validLo && validHi) {
            if (linearAzimuth && radialLo != radialHi) {
                outValue = loValue + (hiValue - loValue) * radialWeight;
            } else if (radialWeight > 0.5f) {
                outValue = hiValue;
            } else {
                outValue = loValue;
            }
            valid = true;
        } else if (validLo && (!validHi || radialWeight <= 0.5f)) {
            outValue = loValue;
            valid = true;
        } else if (validHi) {
            outValue = hiValue;
            valid = true;
        }

        if (valid) {
            outTensor[valueOffset] = outValue;
            outTensor[maskOffset] = 1.0f;
        }
    }
}

bool chooseSpec(const SweepInput inputs[NUM_TENSOR_PRODUCTS],
                const TensorSpec* preferredSpec,
                TensorSpec* outSpec,
                uint32_t* outMask) {
    if (!outSpec || !outMask)
        return false;

    *outMask = 0;
    for (int slot = 0; slot < NUM_TENSOR_PRODUCTS; ++slot) {
        if (inputs[slot].valid && inputs[slot].h_azimuths && inputs[slot].h_gates &&
            inputs[slot].num_radials > 0 && inputs[slot].num_gates > 0 &&
            inputs[slot].gate_spacing_km > 0.0f) {
            *outMask |= (1u << slot);
        }
    }

    if (*outMask == 0)
        return false;

    TensorSpec spec = {};
    if (preferredSpec)
        spec = *preferredSpec;

    if (spec.base_product_slot < 0 || spec.base_product_slot >= NUM_TENSOR_PRODUCTS ||
        ((*outMask & (1u << spec.base_product_slot)) == 0)) {
        for (int slot = 0; slot < NUM_TENSOR_PRODUCTS; ++slot) {
            if ((*outMask & (1u << slot)) != 0) {
                spec.base_product_slot = slot;
                break;
            }
        }
    }

    if (spec.base_product_slot < 0)
        return false;

    const SweepInput& base = inputs[spec.base_product_slot];
    if (spec.num_radials <= 0)
        spec.num_radials = base.num_radials;
    if (spec.num_gates <= 0)
        spec.num_gates = base.num_gates;
    if (spec.first_gate_km <= 0.0f && base.first_gate_km >= 0.0f)
        spec.first_gate_km = base.first_gate_km;
    if (spec.gate_spacing_km <= 0.0f)
        spec.gate_spacing_km = base.gate_spacing_km;

    if (spec.num_radials <= 0 || spec.num_gates <= 0 || spec.gate_spacing_km <= 0.0f)
        return false;

    *outSpec = spec;
    return true;
}

} // namespace

SweepInput makeSweepInput(const PrecomputedSweep::ProductData& pd,
                          int numRadials,
                          const float* azimuths) {
    SweepInput input;
    if (!pd.has_data || numRadials <= 0 || pd.num_gates <= 0 ||
        azimuths == nullptr || pd.gates.empty()) {
        return input;
    }

    input.valid = true;
    input.num_radials = numRadials;
    input.num_gates = pd.num_gates;
    input.first_gate_km = pd.first_gate_km;
    input.gate_spacing_km = pd.gate_spacing_km;
    input.scale = pd.scale;
    input.offset = pd.offset;
    input.h_azimuths = azimuths;
    input.h_gates = pd.gates.data();
    return input;
}

SweepInput makeSweepInput(const PrecomputedSweep& sweep, int product) {
    if (product < 0 || product >= NUM_PRODUCTS)
        return {};
    return makeSweepInput(sweep.products[product], sweep.num_radials, sweep.azimuths.data());
}

int slotToProduct(int slot) {
    switch (slot) {
    case SLOT_REF: return PROD_REF;
    case SLOT_VEL: return PROD_VEL;
    case SLOT_ZDR: return PROD_ZDR;
    case SLOT_CC: return PROD_CC;
    default: return -1;
    }
}

TensorWorkspace::TensorWorkspace() = default;

TensorWorkspace::~TensorWorkspace() {
    reset();
}

bool TensorWorkspace::ensureStream() {
    if (m_stream)
        return true;
    return cudaSuccessOnly(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));
}

bool TensorWorkspace::ensureInputCapacity(int slot, int numRadials, size_t gateBytes) {
    if (slot < 0 || slot >= NUM_TENSOR_PRODUCTS)
        return false;

    DeviceInput& input = m_inputs[slot];
    if (numRadials > input.azimuth_capacity) {
        if (input.d_azimuths) {
            if (!cudaSuccessOnly(cudaFree(input.d_azimuths)))
                return false;
        }
        if (!cudaSuccessOnly(cudaMalloc(&input.d_azimuths, (size_t)numRadials * sizeof(float))))
            return false;
        input.azimuth_capacity = numRadials;
    }

    if (gateBytes > input.gate_capacity_bytes) {
        if (input.d_gates) {
            if (!cudaSuccessOnly(cudaFree(input.d_gates)))
                return false;
        }
        if (!cudaSuccessOnly(cudaMalloc(&input.d_gates, gateBytes)))
            return false;
        input.gate_capacity_bytes = gateBytes;
    }

    return true;
}

bool TensorWorkspace::ensureTensorCapacity(size_t tensorElements, int numRadials) {
    if (tensorElements > m_tensor_capacity_elements) {
        if (m_d_tensor) {
            if (!cudaSuccessOnly(cudaFree(m_d_tensor)))
                return false;
        }
        if (!cudaSuccessOnly(cudaMalloc(&m_d_tensor, tensorElements * sizeof(float))))
            return false;
        m_tensor_capacity_elements = tensorElements;
    }

    if (numRadials > m_target_azimuth_capacity) {
        if (m_d_targetAzimuths) {
            if (!cudaSuccessOnly(cudaFree(m_d_targetAzimuths)))
                return false;
        }
        if (!cudaSuccessOnly(cudaMalloc(&m_d_targetAzimuths, (size_t)numRadials * sizeof(float))))
            return false;
        m_target_azimuth_capacity = numRadials;
    }

    return true;
}

bool TensorWorkspace::ensureMapCapacity(DeviceMap& map, int count) {
    if (count <= map.capacity)
        return true;

    if (map.d_lo && !cudaSuccessOnly(cudaFree(map.d_lo)))
        return false;
    if (map.d_hi && !cudaSuccessOnly(cudaFree(map.d_hi)))
        return false;
    if (map.d_weight && !cudaSuccessOnly(cudaFree(map.d_weight)))
        return false;

    map.d_lo = nullptr;
    map.d_hi = nullptr;
    map.d_weight = nullptr;
    map.capacity = 0;

    if (!cudaSuccessOnly(cudaMalloc(&map.d_lo, (size_t)count * sizeof(int))) ||
        !cudaSuccessOnly(cudaMalloc(&map.d_hi, (size_t)count * sizeof(int))) ||
        !cudaSuccessOnly(cudaMalloc(&map.d_weight, (size_t)count * sizeof(float)))) {
        return false;
    }

    map.capacity = count;
    return true;
}

bool TensorWorkspace::ensurePointerTables() {
    if (m_d_radialLoPtrs && m_d_radialHiPtrs && m_d_radialWeightPtrs &&
        m_d_gateLoPtrs && m_d_gateHiPtrs && m_d_gateWeightPtrs) {
        return true;
    }

    if (!m_d_radialLoPtrs &&
        !cudaSuccessOnly(cudaMalloc(&m_d_radialLoPtrs,
                                    (size_t)NUM_TENSOR_PRODUCTS * sizeof(int*))))
        return false;
    if (!m_d_radialHiPtrs &&
        !cudaSuccessOnly(cudaMalloc(&m_d_radialHiPtrs,
                                    (size_t)NUM_TENSOR_PRODUCTS * sizeof(int*))))
        return false;
    if (!m_d_radialWeightPtrs &&
        !cudaSuccessOnly(cudaMalloc(&m_d_radialWeightPtrs,
                                    (size_t)NUM_TENSOR_PRODUCTS * sizeof(float*))))
        return false;
    if (!m_d_gateLoPtrs &&
        !cudaSuccessOnly(cudaMalloc(&m_d_gateLoPtrs,
                                    (size_t)NUM_TENSOR_PRODUCTS * sizeof(int*))))
        return false;
    if (!m_d_gateHiPtrs &&
        !cudaSuccessOnly(cudaMalloc(&m_d_gateHiPtrs,
                                    (size_t)NUM_TENSOR_PRODUCTS * sizeof(int*))))
        return false;
    if (!m_d_gateWeightPtrs &&
        !cudaSuccessOnly(cudaMalloc(&m_d_gateWeightPtrs,
                                    (size_t)NUM_TENSOR_PRODUCTS * sizeof(float*))))
        return false;
    return true;
}

bool TensorWorkspace::build(const SweepInput inputs[NUM_TENSOR_PRODUCTS],
                            const TensorSpec* preferredSpec) {
    if (!inputs || !ensureStream())
        return false;
    m_tensor = {};

    TensorSpec spec = {};
    uint32_t validMask = 0;
    if (!chooseSpec(inputs, preferredSpec, &spec, &validMask))
        return false;

    const size_t cellCount = (size_t)spec.num_radials * (size_t)spec.num_gates;
    const size_t tensorElements = cellCount * (size_t)NUM_TENSOR_CHANNELS;
    if (!ensureTensorCapacity(tensorElements, spec.num_radials))
        return false;
    if (!ensurePointerTables())
        return false;

    if (!m_d_inputs) {
        if (!cudaSuccessOnly(cudaMalloc(&m_d_inputs,
                                        (size_t)NUM_TENSOR_PRODUCTS * sizeof(KernelSweepInput)))) {
            return false;
        }
    }

    std::vector<KernelSweepInput> hostInputs(NUM_TENSOR_PRODUCTS);
    std::vector<const int*> hostRadialLo(NUM_TENSOR_PRODUCTS, nullptr);
    std::vector<const int*> hostRadialHi(NUM_TENSOR_PRODUCTS, nullptr);
    std::vector<const float*> hostRadialWeight(NUM_TENSOR_PRODUCTS, nullptr);
    std::vector<const int*> hostGateLo(NUM_TENSOR_PRODUCTS, nullptr);
    std::vector<const int*> hostGateHi(NUM_TENSOR_PRODUCTS, nullptr);
    std::vector<const float*> hostGateWeight(NUM_TENSOR_PRODUCTS, nullptr);

    for (int slot = 0; slot < NUM_TENSOR_PRODUCTS; ++slot) {
        if ((validMask & (1u << slot)) == 0) {
            hostInputs[slot].valid = 0;
            continue;
        }

        const SweepInput& input = inputs[slot];
        const size_t gateBytes = (size_t)input.num_radials * (size_t)input.num_gates * sizeof(uint16_t);
        if (!ensureInputCapacity(slot, input.num_radials, gateBytes))
            return false;
        if (!ensureMapCapacity(m_radialMaps[slot], spec.num_radials) ||
            !ensureMapCapacity(m_gateMaps[slot], spec.num_gates)) {
            return false;
        }

        if (!cudaSuccessOnly(cudaMemcpyAsync(m_inputs[slot].d_azimuths, input.h_azimuths,
                                             (size_t)input.num_radials * sizeof(float),
                                             cudaMemcpyHostToDevice, m_stream)) ||
            !cudaSuccessOnly(cudaMemcpyAsync(m_inputs[slot].d_gates, input.h_gates,
                                             gateBytes,
                                             cudaMemcpyHostToDevice, m_stream))) {
            return false;
        }

        hostInputs[slot].azimuths = m_inputs[slot].d_azimuths;
        hostInputs[slot].gates = m_inputs[slot].d_gates;
        hostInputs[slot].num_radials = input.num_radials;
        hostInputs[slot].num_gates = input.num_gates;
        hostInputs[slot].first_gate_km = input.first_gate_km;
        hostInputs[slot].gate_spacing_km = input.gate_spacing_km;
        hostInputs[slot].scale = input.scale;
        hostInputs[slot].offset = input.offset;
        hostInputs[slot].valid = 1;

        hostRadialLo[slot] = m_radialMaps[slot].d_lo;
        hostRadialHi[slot] = m_radialMaps[slot].d_hi;
        hostRadialWeight[slot] = m_radialMaps[slot].d_weight;
        hostGateLo[slot] = m_gateMaps[slot].d_lo;
        hostGateHi[slot] = m_gateMaps[slot].d_hi;
        hostGateWeight[slot] = m_gateMaps[slot].d_weight;
    }

    const SweepInput& base = inputs[spec.base_product_slot];
    if ((spec.flags & TENSOR_TARGET_UNIFORM_AZIMUTH) != 0 || spec.num_radials != base.num_radials) {
        const int threads = 256;
        const int blocks = (spec.num_radials + threads - 1) / threads;
        generateUniformAzimuthsKernel<<<blocks, threads, 0, m_stream>>>(
            m_d_targetAzimuths, spec.num_radials, spec.azimuth_offset_deg);
    } else {
        if (!cudaSuccessOnly(cudaMemcpyAsync(m_d_targetAzimuths, base.h_azimuths,
                                             (size_t)spec.num_radials * sizeof(float),
                                             cudaMemcpyHostToDevice, m_stream))) {
            return false;
        }
    }
    if (!cudaSuccessOnly(cudaGetLastError()))
        return false;

    for (int slot = 0; slot < NUM_TENSOR_PRODUCTS; ++slot) {
        if ((validMask & (1u << slot)) == 0)
            continue;

        const bool linearAzimuth = (spec.flags & TENSOR_INTERP_AZIMUTH_LINEAR) != 0;
        const bool linearRange = (spec.flags & TENSOR_INTERP_RANGE_LINEAR) != 0;

        {
            const int threads = 256;
            const int blocks = (spec.num_radials + threads - 1) / threads;
            buildRadialMapKernel<<<blocks, threads, 0, m_stream>>>(
                m_inputs[slot].d_azimuths, inputs[slot].num_radials,
                m_d_targetAzimuths, spec.num_radials,
                m_radialMaps[slot].d_lo, m_radialMaps[slot].d_hi,
                m_radialMaps[slot].d_weight, linearAzimuth);
        }

        {
            const int threads = 256;
            const int blocks = (spec.num_gates + threads - 1) / threads;
            buildGateMapKernel<<<blocks, threads, 0, m_stream>>>(
                inputs[slot].num_gates,
                inputs[slot].first_gate_km,
                inputs[slot].gate_spacing_km,
                spec.num_gates,
                spec.first_gate_km,
                spec.gate_spacing_km,
                m_gateMaps[slot].d_lo, m_gateMaps[slot].d_hi,
                m_gateMaps[slot].d_weight, linearRange);
        }
    }

    if (!cudaSuccessOnly(cudaGetLastError()))
        return false;

    if (!cudaSuccessOnly(cudaMemcpyAsync(m_d_inputs, hostInputs.data(),
                                         hostInputs.size() * sizeof(KernelSweepInput),
                                         cudaMemcpyHostToDevice, m_stream)) ||
        !cudaSuccessOnly(cudaMemcpyAsync(m_d_radialLoPtrs, hostRadialLo.data(),
                                         hostRadialLo.size() * sizeof(int*),
                                         cudaMemcpyHostToDevice, m_stream)) ||
        !cudaSuccessOnly(cudaMemcpyAsync(m_d_radialHiPtrs, hostRadialHi.data(),
                                         hostRadialHi.size() * sizeof(int*),
                                         cudaMemcpyHostToDevice, m_stream)) ||
        !cudaSuccessOnly(cudaMemcpyAsync(m_d_radialWeightPtrs, hostRadialWeight.data(),
                                         hostRadialWeight.size() * sizeof(float*),
                                         cudaMemcpyHostToDevice, m_stream)) ||
        !cudaSuccessOnly(cudaMemcpyAsync(m_d_gateLoPtrs, hostGateLo.data(),
                                         hostGateLo.size() * sizeof(int*),
                                         cudaMemcpyHostToDevice, m_stream)) ||
        !cudaSuccessOnly(cudaMemcpyAsync(m_d_gateHiPtrs, hostGateHi.data(),
                                         hostGateHi.size() * sizeof(int*),
                                         cudaMemcpyHostToDevice, m_stream)) ||
        !cudaSuccessOnly(cudaMemcpyAsync(m_d_gateWeightPtrs, hostGateWeight.data(),
                                         hostGateWeight.size() * sizeof(float*),
                                         cudaMemcpyHostToDevice, m_stream))) {
        return false;
    }

    {
        dim3 block(16, 16);
        dim3 grid((spec.num_radials + block.x - 1) / block.x,
                  (spec.num_gates + block.y - 1) / block.y);
        buildTensorKernel<<<grid, block, 0, m_stream>>>(
            static_cast<const KernelSweepInput*>(m_d_inputs),
            m_d_radialLoPtrs, m_d_radialHiPtrs, m_d_radialWeightPtrs,
            m_d_gateLoPtrs, m_d_gateHiPtrs, m_d_gateWeightPtrs,
            spec, m_d_tensor, validMask);
    }

    if (!cudaSuccessOnly(cudaGetLastError()) ||
        !cudaSuccessOnly(cudaStreamSynchronize(m_stream))) {
        return false;
    }

    m_tensor.spec = spec;
    m_tensor.num_channels = NUM_TENSOR_CHANNELS;
    m_tensor.cell_count = cellCount;
    m_tensor.channel_stride = cellCount;
    m_tensor.valid_product_mask = validMask;
    m_tensor.d_data = m_d_tensor;
    m_tensor.d_target_azimuths = m_d_targetAzimuths;
    return true;
}

bool TensorWorkspace::downloadTargetAzimuths(std::vector<float>& out) const {
    if (!m_tensor.d_target_azimuths || m_tensor.spec.num_radials <= 0)
        return false;

    out.resize((size_t)m_tensor.spec.num_radials);
    return cudaSuccessOnly(cudaMemcpy(out.data(), m_tensor.d_target_azimuths,
                                      out.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost));
}

void TensorWorkspace::reset() {
    for (auto& map : m_gateMaps) {
        if (map.d_weight) cudaFree(map.d_weight);
        if (map.d_hi) cudaFree(map.d_hi);
        if (map.d_lo) cudaFree(map.d_lo);
        map = {};
    }

    for (auto& map : m_radialMaps) {
        if (map.d_weight) cudaFree(map.d_weight);
        if (map.d_hi) cudaFree(map.d_hi);
        if (map.d_lo) cudaFree(map.d_lo);
        map = {};
    }

    for (auto& input : m_inputs) {
        if (input.d_gates) cudaFree(input.d_gates);
        if (input.d_azimuths) cudaFree(input.d_azimuths);
        input = {};
    }

    if (m_d_inputs) cudaFree(m_d_inputs);
    if (m_d_gateWeightPtrs) cudaFree(m_d_gateWeightPtrs);
    if (m_d_gateHiPtrs) cudaFree(m_d_gateHiPtrs);
    if (m_d_gateLoPtrs) cudaFree(m_d_gateLoPtrs);
    if (m_d_radialWeightPtrs) cudaFree(m_d_radialWeightPtrs);
    if (m_d_radialHiPtrs) cudaFree(m_d_radialHiPtrs);
    if (m_d_radialLoPtrs) cudaFree(m_d_radialLoPtrs);
    if (m_d_targetAzimuths) cudaFree(m_d_targetAzimuths);
    if (m_d_tensor) cudaFree(m_d_tensor);
    if (m_stream) cudaStreamDestroy(m_stream);

    m_d_inputs = nullptr;
    m_d_gateWeightPtrs = nullptr;
    m_d_gateHiPtrs = nullptr;
    m_d_gateLoPtrs = nullptr;
    m_d_radialWeightPtrs = nullptr;
    m_d_radialHiPtrs = nullptr;
    m_d_radialLoPtrs = nullptr;
    m_d_targetAzimuths = nullptr;
    m_d_tensor = nullptr;
    m_stream = nullptr;
    m_tensor_capacity_elements = 0;
    m_target_azimuth_capacity = 0;
    m_tensor = {};
}

} // namespace gpu_tensor
