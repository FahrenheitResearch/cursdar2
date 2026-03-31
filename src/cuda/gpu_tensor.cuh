#pragma once

#include "../nexrad/sweep_data.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gpu_tensor {

constexpr int NUM_TENSOR_PRODUCTS = 4;
constexpr int NUM_TENSOR_CHANNELS = 8;

enum TensorProductSlot {
    SLOT_REF = 0,
    SLOT_VEL = 1,
    SLOT_ZDR = 2,
    SLOT_CC = 3
};

enum TensorChannel {
    CHANNEL_REF = 0,
    CHANNEL_VEL = 1,
    CHANNEL_ZDR = 2,
    CHANNEL_CC = 3,
    CHANNEL_MASK_REF = 4,
    CHANNEL_MASK_VEL = 5,
    CHANNEL_MASK_ZDR = 6,
    CHANNEL_MASK_CC = 7
};

enum TensorBuildFlags : uint32_t {
    TENSOR_INTERP_AZIMUTH_LINEAR = 1u << 0,
    TENSOR_INTERP_RANGE_LINEAR = 1u << 1,
    TENSOR_TARGET_UNIFORM_AZIMUTH = 1u << 2
};

struct SweepInput {
    bool valid = false;
    int num_radials = 0;
    int num_gates = 0;
    float first_gate_km = 0.0f;
    float gate_spacing_km = 0.0f;
    float scale = 0.0f;
    float offset = 0.0f;
    const float* h_azimuths = nullptr;
    const uint16_t* h_gates = nullptr;
};

struct TensorSpec {
    int num_radials = 0;
    int num_gates = 0;
    float first_gate_km = 0.0f;
    float gate_spacing_km = 0.0f;
    int base_product_slot = -1;
    uint32_t flags = TENSOR_INTERP_AZIMUTH_LINEAR | TENSOR_INTERP_RANGE_LINEAR;
    float azimuth_offset_deg = 0.0f;
};

struct PolarTensor {
    TensorSpec spec = {};
    int num_channels = NUM_TENSOR_CHANNELS;
    size_t cell_count = 0;
    size_t channel_stride = 0;
    uint32_t valid_product_mask = 0;
    const float* d_data = nullptr;
    const float* d_target_azimuths = nullptr;

    const float* channel(int channelIndex) const {
        if (!d_data || channelIndex < 0 || channelIndex >= num_channels)
            return nullptr;
        return d_data + channel_stride * (size_t)channelIndex;
    }
};

SweepInput makeSweepInput(const PrecomputedSweep::ProductData& pd,
                          int numRadials,
                          const float* azimuths);
SweepInput makeSweepInput(const PrecomputedSweep& sweep, int product);
int slotToProduct(int slot);
__host__ __device__ inline int maskChannelForSlot(int slot) {
    return NUM_TENSOR_PRODUCTS + slot;
}

class TensorWorkspace {
public:
    TensorWorkspace();
    ~TensorWorkspace();

    bool build(const SweepInput inputs[NUM_TENSOR_PRODUCTS],
               const TensorSpec* preferredSpec = nullptr);
    const PolarTensor& tensor() const { return m_tensor; }
    cudaStream_t stream() const { return m_stream; }
    bool downloadTargetAzimuths(std::vector<float>& out) const;
    void reset();

private:
    struct DeviceInput {
        float* d_azimuths = nullptr;
        uint16_t* d_gates = nullptr;
        int azimuth_capacity = 0;
        size_t gate_capacity_bytes = 0;
    };

    struct DeviceMap {
        int* d_lo = nullptr;
        int* d_hi = nullptr;
        float* d_weight = nullptr;
        int capacity = 0;
    };

    bool ensureStream();
    bool ensureInputCapacity(int slot, int numRadials, size_t gateBytes);
    bool ensureTensorCapacity(size_t tensorElements, int numRadials);
    bool ensureMapCapacity(DeviceMap& map, int count);
    bool ensurePointerTables();

    DeviceInput m_inputs[NUM_TENSOR_PRODUCTS];
    DeviceMap m_radialMaps[NUM_TENSOR_PRODUCTS];
    DeviceMap m_gateMaps[NUM_TENSOR_PRODUCTS];
    void* m_d_inputs = nullptr;
    const int** m_d_radialLoPtrs = nullptr;
    const int** m_d_radialHiPtrs = nullptr;
    const float** m_d_radialWeightPtrs = nullptr;
    const int** m_d_gateLoPtrs = nullptr;
    const int** m_d_gateHiPtrs = nullptr;
    const float** m_d_gateWeightPtrs = nullptr;
    float* m_d_tensor = nullptr;
    float* m_d_targetAzimuths = nullptr;
    size_t m_tensor_capacity_elements = 0;
    int m_target_azimuth_capacity = 0;
    cudaStream_t m_stream = nullptr;
    PolarTensor m_tensor = {};
};

} // namespace gpu_tensor
