#include "gpu_pipeline.cuh"
#include "../nexrad/level2.h"
#include <cstdio>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

namespace gpu_pipeline {

namespace {

struct DeviceProductMeta {
    int has_product = 0;
    int num_gates = 0;
    int first_gate = 0;
    int gate_spacing = 0;
    float scale = 0.0f;
    float offset = 0.0f;
};

struct PipelineScratch {
    cudaStream_t owned_stream = nullptr;
    uint8_t* d_raw = nullptr;
    size_t raw_capacity = 0;
    int* d_offsets = nullptr;
    size_t offset_capacity = 0;
    int* d_count = nullptr;
    GpuParsedRadial* d_radials = nullptr;
    size_t radial_capacity = 0;
    int* d_indices = nullptr;
    size_t index_capacity = 0;
    uint32_t* d_lowest_key = nullptr;
    size_t lowest_key_capacity = 0;
    uint32_t* d_sweep_bits = nullptr;
    size_t sweep_bits_capacity = 0;
    int* d_selected_count = nullptr;
    size_t selected_count_capacity = 0;
    DeviceProductMeta* d_product_meta = nullptr;
    size_t product_meta_capacity = 0;

    ~PipelineScratch() {
        if (d_product_meta) cudaFree(d_product_meta);
        if (d_selected_count) cudaFree(d_selected_count);
        if (d_sweep_bits) cudaFree(d_sweep_bits);
        if (d_lowest_key) cudaFree(d_lowest_key);
        if (d_indices) cudaFree(d_indices);
        if (d_radials) cudaFree(d_radials);
        if (d_count) cudaFree(d_count);
        if (d_offsets) cudaFree(d_offsets);
        if (d_raw) cudaFree(d_raw);
        if (owned_stream) cudaStreamDestroy(owned_stream);
    }
};

thread_local PipelineScratch g_scratch;

cudaStream_t resolveStream(PipelineScratch& scratch, cudaStream_t requested) {
    if (requested)
        return requested;
    if (!scratch.owned_stream)
        CUDA_CHECK(cudaStreamCreateWithFlags(&scratch.owned_stream, cudaStreamNonBlocking));
    return scratch.owned_stream;
}

template <typename T>
void ensureCapacity(T*& ptr, size_t& capacity, size_t requiredCount) {
    if (capacity >= requiredCount)
        return;
    size_t newCapacity = capacity ? capacity : 1;
    while (newCapacity < requiredCount)
        newCapacity = std::max(newCapacity + newCapacity / 2, requiredCount);
    if (ptr)
        CUDA_CHECK(cudaFree(ptr));
    CUDA_CHECK(cudaMalloc(&ptr, newCapacity * sizeof(T)));
    capacity = newCapacity;
}

void ensureScalar(int*& ptr) {
    if (!ptr)
        CUDA_CHECK(cudaMalloc(&ptr, sizeof(int)));
}

int countSetBits(uint32_t value) {
    int count = 0;
    while (value) {
        value &= (value - 1);
        ++count;
    }
    return count;
}

int estimateMaxParsedRadials(size_t raw_size) {
    const size_t estimate = raw_size / 1216 + 1024;
    return (int)std::max<size_t>(4096, estimate);
}

int absoluteMaxParsedRadials(size_t raw_size) {
    // MSG31 records are variable length, but if the estimate still truncates we
    // keep growing until we can hold one slot per ~64 decoded bytes. That is
    // intentionally conservative and avoids silently clamping large volumes.
    return (int)std::max<size_t>(estimateMaxParsedRadials(raw_size), raw_size / 64 + 4096);
}

int parseWithResize(const uint8_t* d_raw_data, size_t raw_size,
                    GpuParsedRadial*& d_radials_out, size_t& radial_capacity,
                    PipelineScratch& scratch, cudaStream_t activeStream,
                    bool* out_truncated) {
    const int hardLimit = absoluteMaxParsedRadials(raw_size);
    int maxParsed = estimateMaxParsedRadials(raw_size);
    bool truncated = false;
    int num_parsed = 0;

    while (true) {
        ensureCapacity(d_radials_out, radial_capacity, (size_t)maxParsed);
        ensureCapacity(scratch.d_indices, scratch.index_capacity, (size_t)maxParsed);
        num_parsed = parseOnGpu(d_raw_data, raw_size, d_radials_out, maxParsed,
                                activeStream, &truncated);
        CUDA_CHECK(cudaStreamSynchronize(activeStream));
        if (!truncated || maxParsed >= hardLimit)
            break;
        maxParsed = std::min(hardLimit, std::max(maxParsed + maxParsed / 2, maxParsed + 4096));
    }

    if (out_truncated)
        *out_truncated = truncated;
    return num_parsed;
}

} // namespace

// ── GPU Parser Kernel ───────────────────────────────────────
// Each thread scans from a potential message start position.
// We check every 2432-byte boundary AND variable offsets.

__device__ bool isValidMsg31(const uint8_t* data, size_t pos, size_t size) {
    if (pos + 12 + 16 + 32 > size) return false;
    // CTM at pos, MessageHeader at pos+12
    uint8_t msg_type = data[pos + 12 + 3]; // message_type byte
    if (msg_type != 31) return false;
    // Check message size is sane
    uint16_t msize = ((uint16_t)data[pos + 12] << 8) | data[pos + 12 + 1];
    if (msize < 20 || msize > 30000) return false;
    return true;
}

// Byte-swap helpers (big-endian to little-endian)
__device__ uint16_t d_bswap16(uint16_t v) {
    return (v >> 8) | (v << 8);
}
__device__ uint32_t d_bswap32(uint32_t v) {
    return ((v >> 24) & 0xFF) | ((v >> 8) & 0xFF00) |
           ((v << 8) & 0xFF0000) | ((v << 24) & 0xFF000000);
}
__device__ float d_bswapf(float v) {
    uint32_t i;
    memcpy(&i, &v, 4);
    i = d_bswap32(i);
    float r;
    memcpy(&r, &i, 4);
    return r;
}

__device__ __forceinline__ uint16_t d_read_be16(const uint8_t* p) {
    return ((uint16_t)p[0] << 8) | p[1];
}

__device__ __forceinline__ uint32_t d_read_be32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) |
           ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) |
           (uint32_t)p[3];
}

__device__ __forceinline__ float d_read_bef(const uint8_t* p) {
    return __uint_as_float(d_read_be32(p));
}

// Product code matching on GPU
__device__ int d_productFromCode(const char* code) {
    // REF=0, VEL=1, SW=2, ZDR=3, RHO=4, KDP=5, PHI=6
    if (code[0] == 'R' && code[1] == 'E' && code[2] == 'F') return 0;
    if (code[0] == 'V' && code[1] == 'E' && code[2] == 'L') return 1;
    if (code[0] == 'S' && code[1] == 'W')                   return 2;
    if (code[0] == 'Z' && code[1] == 'D' && code[2] == 'R') return 3;
    if (code[0] == 'R' && code[1] == 'H' && code[2] == 'O') return 4;
    if (code[0] == 'K' && code[1] == 'D' && code[2] == 'P') return 5;
    if (code[0] == 'P' && code[1] == 'H' && code[2] == 'I') return 6;
    return -1;
}

__global__ void parseMsg31Kernel(
    const uint8_t* __restrict__ raw,
    size_t raw_size,
    const int* __restrict__ msg_offsets, // pre-found message positions
    int num_messages,
    GpuParsedRadial* __restrict__ radials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_messages) return;

    int pos = msg_offsets[idx];
    if (pos < 0 || (size_t)pos + 12 + 16 + 32 > raw_size) return;

    // Skip CTM(12) + MessageHeader(16) → Msg31Header starts at pos+28
    const uint8_t* msg = raw + pos + 12 + 16;
    size_t msg_remain = raw_size - (pos + 12 + 16);
    if (msg_remain < 32) return;

    GpuParsedRadial& r = radials[idx];
    memset(&r, 0, sizeof(r));
    for (int p = 0; p < NUM_PRODUCTS; p++) r.moment_offsets[p] = -1;

    // Parse Msg31 header
    // Bytes 12-15: azimuth angle (float32 BE)
    // Bytes 24-27: elevation angle (float32 BE)
    r.azimuth = d_read_bef(msg + 12);
    r.elevation = d_read_bef(msg + 24);
    r.radial_status = msg[21];
    r.elevation_number = msg[22];

    // Validate
    if (r.azimuth < 0.0f || r.azimuth >= 360.0f) { r.azimuth = -1; return; }
    if (r.elevation < -2.0f || r.elevation > 90.0f) { r.azimuth = -1; return; }

    // Data block count at bytes 30-31
    uint16_t block_count = d_read_be16(msg + 30);
    if (block_count < 1 || block_count > 20) return;
    if (32 + (size_t)block_count * sizeof(uint32_t) > msg_remain) return;

    for (int b = 0; b < block_count && b < 10; b++) {
        uint32_t bptr = d_read_be32(msg + 32 + b * sizeof(uint32_t));
        if (bptr == 0 || bptr >= msg_remain) continue;
        if (bptr + 4 > msg_remain) continue;

        const uint8_t* block = msg + bptr;
        char btype = (char)block[0];
        char bname[3] = {(char)block[1], (char)block[2], (char)block[3]};

        if (btype == 'D') {
            int p = d_productFromCode(bname);
            if (p < 0 || p >= NUM_PRODUCTS) continue;
            if (bptr + 28 > msg_remain) continue;

            r.num_gates[p] = d_read_be16(block + 8);
            r.first_gate[p] = d_read_be16(block + 10);
            r.gate_spacing[p] = d_read_be16(block + 12);
            r.data_word_size[p] = block[19];
            r.scale[p] = d_read_bef(block + 20);
            r.offset[p] = d_read_bef(block + 24);

            // Store the absolute byte offset to gate data within raw buffer
            r.moment_offsets[p] = (int)((msg + bptr + 28) - raw);

            const int gateBytesPerWord = (r.data_word_size[p] == 16) ? 2 : (r.data_word_size[p] == 8 ? 1 : 0);
            if (r.num_gates[p] == 0 || r.num_gates[p] > 2000) {
                r.moment_offsets[p] = -1;
                r.num_gates[p] = 0;
            }
            if (gateBytesPerWord == 0 || r.gate_spacing[p] <= 0) {
                r.moment_offsets[p] = -1;
            }
            if (r.scale[p] == 0.0f || !isfinite(r.scale[p]) || !isfinite(r.offset[p])) {
                r.moment_offsets[p] = -1;
            }
            const size_t gateBytes = (size_t)r.num_gates[p] * gateBytesPerWord;
            if (r.moment_offsets[p] >= 0 && bptr + 28 + gateBytes > msg_remain) {
                r.moment_offsets[p] = -1;
                r.num_gates[p] = 0;
            }
        }
    }
}

// ── Message offset finder kernel ────────────────────────────
// Scans raw data for valid MSG31 positions at 2432-byte boundaries
// and variable offsets.

__global__ void findMessageOffsetsKernel(
    const uint8_t* __restrict__ raw,
    size_t raw_size,
    int* __restrict__ offsets_out,
    int* __restrict__ count_out,
    int max_messages)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread checks one potential position
    // Try 2432-byte aligned positions AND positions based on message size
    size_t pos = (size_t)tid * 2432;
    if (pos + 28 > raw_size) return;

    if (isValidMsg31(raw, pos, raw_size)) {
        int slot = atomicAdd(count_out, 1);
        if (slot < max_messages) {
            offsets_out[slot] = (int)pos;
        }
    }
}

// Second pass: find messages at variable offsets (between 2432-byte boundaries)
__global__ void findVariableOffsetsKernel(
    const uint8_t* __restrict__ raw,
    size_t raw_size,
    const int* __restrict__ aligned_offsets,
    int num_aligned,
    int* __restrict__ offsets_out,
    int* __restrict__ count_out,
    int max_messages)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_aligned) return;

    int base = aligned_offsets[idx];
    if (base < 0 || (size_t)base + 28 > raw_size) return;

    // Read message size and compute next message position
    uint16_t msize = ((uint16_t)raw[base + 12] << 8) | raw[base + 12 + 1];
    if (msize < 20 || msize > 30000) return;

    size_t next_pos = (size_t)base + (size_t)msize * 2 + 12;
    // Check if there's a valid message at the variable-offset position
    // that we wouldn't find at a 2432-byte boundary
    if (next_pos % 2432 != 0 && next_pos + 28 < raw_size) {
        if (isValidMsg31(raw, next_pos, raw_size)) {
            int slot = atomicAdd(count_out, 1);
            if (slot < max_messages) {
                offsets_out[slot] = (int)next_pos;
            }
        }
    }
}

// ── Transposition Kernel ────────────────────────────────────
// One thread per (gate, radial) pair. Reads from raw buffer at
// the offset stored in the parsed radial info, writes to
// gate-major output buffer.

__global__ void scanSweepMetadataKernel(const GpuParsedRadial* __restrict__ radials,
                                        int num_radials,
                                        uint32_t* __restrict__ lowest_key,
                                        uint32_t* __restrict__ sweep_bits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_radials)
        return;

    const GpuParsedRadial& r = radials[idx];
    if (r.azimuth < 0.0f || r.azimuth >= 360.0f)
        return;

    const unsigned int elev_num = r.elevation_number;
    if (elev_num < 256u)
        atomicOr(&sweep_bits[elev_num >> 5], 1u << (elev_num & 31u));

    int quant = __float2int_rn((r.elevation + 5.0f) * 1000.0f);
    quant = max(0, quant);
    uint32_t key = ((uint32_t)quant << 8) | (elev_num & 0xFFu);
    atomicMin(lowest_key, key);
}

__global__ void collectLowestSweepIndicesKernel(const GpuParsedRadial* __restrict__ radials,
                                                int num_radials,
                                                int lowest_elev_num,
                                                int* __restrict__ indices_out,
                                                int* __restrict__ count_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_radials)
        return;

    const GpuParsedRadial& r = radials[idx];
    if (r.azimuth < 0.0f || r.azimuth >= 360.0f)
        return;
    if ((int)r.elevation_number != lowest_elev_num)
        return;

    int slot = atomicAdd(count_out, 1);
    indices_out[slot] = idx;
}

__global__ void collectSweepIndicesKernel(const GpuParsedRadial* __restrict__ radials,
                                          int num_radials,
                                          int elevation_number,
                                          int* __restrict__ indices_out,
                                          int* __restrict__ count_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_radials)
        return;

    const GpuParsedRadial& r = radials[idx];
    if (r.azimuth < 0.0f || r.azimuth >= 360.0f)
        return;
    if ((int)r.elevation_number != elevation_number)
        return;

    int slot = atomicAdd(count_out, 1);
    indices_out[slot] = idx;
}

__global__ void selectProductMetaKernel(const GpuParsedRadial* __restrict__ radials,
                                        const int* __restrict__ radial_indices,
                                        int num_radials,
                                        int product,
                                        DeviceProductMeta* __restrict__ meta_out) {
    DeviceProductMeta best = {};
    for (int i = 0; i < num_radials; ++i) {
        const GpuParsedRadial& r = radials[radial_indices[i]];
        if (r.moment_offsets[product] < 0 || r.num_gates[product] <= 0)
            continue;
        if (!best.has_product || r.num_gates[product] > best.num_gates) {
            best.has_product = 1;
            best.num_gates = r.num_gates[product];
            best.first_gate = r.first_gate[product];
            best.gate_spacing = r.gate_spacing[product];
            best.scale = r.scale[product];
            best.offset = r.offset[product];
        }
    }
    meta_out[product] = best;
}

struct RadialIndexByAzimuth {
    const GpuParsedRadial* radials = nullptr;

    __host__ __device__ bool operator()(int a, int b) const {
        return radials[a].azimuth < radials[b].azimuth;
    }
};

__global__ void transposeKernel(
    const uint8_t* __restrict__ raw_data,
    const GpuParsedRadial* __restrict__ radials,
    const int* __restrict__ radial_indices, // sorted indices by azimuth
    int num_radials,
    int product,
    uint16_t* __restrict__ output,          // [num_gates][num_radials]
    int out_num_gates)
{
    int gate = blockIdx.x * blockDim.x + threadIdx.x;
    int radial = blockIdx.y * blockDim.y + threadIdx.y;
    if (gate >= out_num_gates || radial >= num_radials) return;

    int sorted_idx = radial_indices[radial];
    const GpuParsedRadial& r = radials[sorted_idx];

    uint16_t value = 0;
    int offset = r.moment_offsets[product];
    if (offset >= 0 && gate < r.num_gates[product]) {
        if (r.data_word_size[product] == 16) {
            // 16-bit big-endian
            const uint8_t* p = raw_data + offset + gate * 2;
            value = ((uint16_t)p[0] << 8) | p[1];
        } else {
            // 8-bit
            value = raw_data[offset + gate];
        }
    }

    // gate-major layout: output[gate * num_radials + radial]
    output[gate * num_radials + radial] = value;
}

// Extract sorted azimuths kernel
__global__ void extractAzimuthsKernel(
    const GpuParsedRadial* __restrict__ radials,
    const int* __restrict__ radial_indices,
    int num_radials,
    float* __restrict__ azimuths_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_radials) return;
    azimuths_out[i] = radials[radial_indices[i]].azimuth;
}

std::vector<int> extractSweepNumbers(const std::array<uint32_t, 8>& sweepBits) {
    std::vector<int> sweepNumbers;
    for (int word = 0; word < (int)sweepBits.size(); ++word) {
        uint32_t bits = sweepBits[word];
        while (bits) {
            int bitIndex = 0;
            while (((bits >> bitIndex) & 1u) == 0u)
                ++bitIndex;
            sweepNumbers.push_back(word * 32 + bitIndex);
            bits &= (bits - 1);
        }
    }
    return sweepNumbers;
}

bool buildSweepResult(const uint8_t* d_raw,
                      const GpuParsedRadial* d_radials,
                      PipelineScratch& scratch,
                      cudaStream_t activeStream,
                      int num_parsed,
                      int elevation_number,
                      GpuIngestResult& result) {
    result = {};

    const int summaryBlocks = (num_parsed + 255) / 256;
    CUDA_CHECK(cudaMemsetAsync(scratch.d_selected_count, 0, sizeof(int), activeStream));
    collectSweepIndicesKernel<<<summaryBlocks, 256, 0, activeStream>>>(
        d_radials, num_parsed, elevation_number, scratch.d_indices, scratch.d_selected_count);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(&result.num_radials, scratch.d_selected_count, sizeof(int),
                               cudaMemcpyDeviceToHost, activeStream));
    CUDA_CHECK(cudaStreamSynchronize(activeStream));
    if (result.num_radials <= 0)
        return false;

    result.elevation_number = elevation_number;
    thrust::device_ptr<int> d_idx_ptr(scratch.d_indices);
    thrust::sort(thrust::cuda::par.on(activeStream), d_idx_ptr, d_idx_ptr + result.num_radials,
                 RadialIndexByAzimuth{d_radials});

    int firstIndex = -1;
    GpuParsedRadial firstRadial = {};
    CUDA_CHECK(cudaMemcpyAsync(&firstIndex, scratch.d_indices, sizeof(int),
                               cudaMemcpyDeviceToHost, activeStream));
    CUDA_CHECK(cudaStreamSynchronize(activeStream));
    if (firstIndex < 0)
        return false;
    CUDA_CHECK(cudaMemcpyAsync(&firstRadial, d_radials + firstIndex, sizeof(GpuParsedRadial),
                               cudaMemcpyDeviceToHost, activeStream));
    CUDA_CHECK(cudaStreamSynchronize(activeStream));
    result.elevation_angle = firstRadial.elevation;

    CUDA_CHECK(cudaMemsetAsync(scratch.d_product_meta, 0,
                               NUM_PRODUCTS * sizeof(DeviceProductMeta), activeStream));
    for (int p = 0; p < NUM_PRODUCTS; ++p) {
        selectProductMetaKernel<<<1, 1, 0, activeStream>>>(
            d_radials, scratch.d_indices, result.num_radials, p, scratch.d_product_meta);
    }
    CUDA_CHECK(cudaGetLastError());

    std::array<DeviceProductMeta, NUM_PRODUCTS> hostMeta = {};
    CUDA_CHECK(cudaMemcpyAsync(hostMeta.data(), scratch.d_product_meta,
                               hostMeta.size() * sizeof(DeviceProductMeta),
                               cudaMemcpyDeviceToHost, activeStream));
    CUDA_CHECK(cudaStreamSynchronize(activeStream));

    bool hasAnyProduct = false;
    for (int p = 0; p < NUM_PRODUCTS; ++p) {
        const auto& meta = hostMeta[p];
        if (!meta.has_product || meta.num_gates <= 0)
            continue;
        hasAnyProduct = true;
        result.has_product[p] = true;
        result.num_gates[p] = meta.num_gates;
        result.first_gate_km[p] = meta.first_gate / 1000.0f;
        result.gate_spacing_km[p] = meta.gate_spacing / 1000.0f;
        result.scale[p] = meta.scale;
        result.offset[p] = meta.offset;
    }
    if (!hasAnyProduct)
        return false;

    CUDA_CHECK(cudaMalloc(&result.d_azimuths, (size_t)result.num_radials * sizeof(float)));
    for (int p = 0; p < NUM_PRODUCTS; ++p) {
        if (!result.has_product[p])
            continue;
        const size_t sz = (size_t)result.num_gates[p] * (size_t)result.num_radials * sizeof(uint16_t);
        CUDA_CHECK(cudaMalloc(&result.d_gates[p], sz));
        transposeGatesGpu(d_raw, d_radials, scratch.d_indices, result.num_radials, p,
                          result.d_gates[p], result.num_gates[p], result.d_azimuths, activeStream);
    }
    CUDA_CHECK(cudaStreamSynchronize(activeStream));
    return true;
}

// ── API Implementation ──────────────────────────────────────

int parseOnGpu(const uint8_t* d_raw_data, size_t raw_size,
               GpuParsedRadial* d_radials_out, int max_radials,
               cudaStream_t stream,
               bool* out_truncated) {
    PipelineScratch& scratch = g_scratch;
    cudaStream_t activeStream = resolveStream(scratch, stream);
    if (out_truncated)
        *out_truncated = false;
    ensureCapacity(scratch.d_offsets, scratch.offset_capacity, (size_t)max_radials);
    ensureScalar(scratch.d_count);
    CUDA_CHECK(cudaMemsetAsync(scratch.d_count, 0, sizeof(int), activeStream));

    // Pass 1: find messages at 2432-byte boundaries
    int num_potential = (int)(raw_size / 2432) + 1;
    int threads = 256;
    int blocks = (num_potential + threads - 1) / threads;
    findMessageOffsetsKernel<<<blocks, threads, 0, activeStream>>>(
        d_raw_data, raw_size, scratch.d_offsets, scratch.d_count, max_radials);

    // Get count from pass 1
    int h_count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_count, scratch.d_count, sizeof(int),
                                cudaMemcpyDeviceToHost, activeStream));
    CUDA_CHECK(cudaStreamSynchronize(activeStream));

    if (h_count > 0) {
        const int aligned_count = (h_count > max_radials) ? max_radials : h_count;
        // Pass 2: find variable-offset messages
        findVariableOffsetsKernel<<<(aligned_count + 255) / 256, 256, 0, activeStream>>>(
            d_raw_data, raw_size, scratch.d_offsets, aligned_count,
            scratch.d_offsets, scratch.d_count, max_radials);

        CUDA_CHECK(cudaMemcpyAsync(&h_count, scratch.d_count, sizeof(int),
                                    cudaMemcpyDeviceToHost, activeStream));
        CUDA_CHECK(cudaStreamSynchronize(activeStream));
    }

    if (h_count <= 0)
        return 0;

    if (out_truncated && h_count > max_radials)
        *out_truncated = true;
    h_count = (h_count > max_radials) ? max_radials : h_count;

    // Sort offsets
    thrust::device_ptr<int> d_off_ptr(scratch.d_offsets);
    thrust::sort(thrust::cuda::par.on(activeStream), d_off_ptr, d_off_ptr + h_count);

    // Parse all found messages
    parseMsg31Kernel<<<(h_count + 255) / 256, 256, 0, activeStream>>>(
        d_raw_data, raw_size, scratch.d_offsets, h_count, d_radials_out);
    return h_count;
}

// Sort helper for radials by azimuth
struct AzimuthSortKey {
    float azimuth;
    int   index;
};

void transposeGatesGpu(
    const uint8_t* d_raw_data,
    const GpuParsedRadial* d_radials,
    const int* d_radial_indices,
    int num_output_radials,
    int product,
    uint16_t* d_output,
    int out_num_gates,
    float* d_azimuths_out,
    cudaStream_t stream)
{
    if (num_output_radials <= 0) return;

    // Clear output
    CUDA_CHECK(cudaMemsetAsync(d_output, 0,
                               (size_t)out_num_gates * num_output_radials * sizeof(uint16_t),
                               stream));

    // Launch transpose kernel
    dim3 block(32, 8);
    dim3 grid((out_num_gates + 31) / 32, (num_output_radials + 7) / 8);
    transposeKernel<<<grid, block, 0, stream>>>(
        d_raw_data, d_radials, d_radial_indices, num_output_radials,
        product, d_output, out_num_gates);

    // Extract sorted azimuths
    extractAzimuthsKernel<<<(num_output_radials + 255) / 256, 256, 0, stream>>>(
        d_radials, d_radial_indices, num_output_radials, d_azimuths_out);
}

// ── Full ingest pipeline ────────────────────────────────────

GpuIngestResult ingestSweepGpu(const uint8_t* h_raw_data, size_t raw_size,
                                cudaStream_t stream) {
    GpuIngestResult result = {};
    PipelineScratch& scratch = g_scratch;
    cudaStream_t activeStream = resolveStream(scratch, stream);
    ensureCapacity(scratch.d_raw, scratch.raw_capacity, raw_size);
    CUDA_CHECK(cudaMemcpyAsync(scratch.d_raw, h_raw_data, raw_size,
                                cudaMemcpyHostToDevice, activeStream));

    // Parse on GPU
    bool truncated = false;
    int num_parsed = parseWithResize(scratch.d_raw, raw_size,
                                     scratch.d_radials, scratch.radial_capacity,
                                     scratch, activeStream, &truncated);
    result.truncated = truncated;
    result.parsed = num_parsed > 0 && !truncated;

    if (num_parsed <= 0 || truncated)
        return result;

    ensureCapacity(scratch.d_lowest_key, scratch.lowest_key_capacity, (size_t)1);
    ensureCapacity(scratch.d_sweep_bits, scratch.sweep_bits_capacity, (size_t)8);
    ensureCapacity(scratch.d_selected_count, scratch.selected_count_capacity, (size_t)1);
    ensureCapacity(scratch.d_product_meta, scratch.product_meta_capacity, (size_t)NUM_PRODUCTS);

    const uint32_t invalidLowestKey = 0xFFFFFFFFu;
    CUDA_CHECK(cudaMemcpyAsync(scratch.d_lowest_key, &invalidLowestKey, sizeof(uint32_t),
                               cudaMemcpyHostToDevice, activeStream));
    CUDA_CHECK(cudaMemsetAsync(scratch.d_sweep_bits, 0, 8 * sizeof(uint32_t), activeStream));

    const int summaryBlocks = (num_parsed + 255) / 256;
    scanSweepMetadataKernel<<<summaryBlocks, 256, 0, activeStream>>>(
        scratch.d_radials, num_parsed, scratch.d_lowest_key, scratch.d_sweep_bits);
    CUDA_CHECK(cudaGetLastError());

    uint32_t lowestKey = invalidLowestKey;
    std::array<uint32_t, 8> sweepBits = {};
    CUDA_CHECK(cudaMemcpyAsync(&lowestKey, scratch.d_lowest_key, sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, activeStream));
    CUDA_CHECK(cudaMemcpyAsync(sweepBits.data(), scratch.d_sweep_bits, sweepBits.size() * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, activeStream));
    CUDA_CHECK(cudaStreamSynchronize(activeStream));

    if (lowestKey == invalidLowestKey)
        return result;

    const int lowest_elev_num = (int)(lowestKey & 0xFFu);
    result.elevation_number = lowest_elev_num;
    const int lowest_quant = (int)(lowestKey >> 8);
    result.elevation_angle = (lowest_quant / 1000.0f) - 5.0f;

    for (uint32_t bits : sweepBits)
        result.total_sweeps += countSetBits(bits);

    CUDA_CHECK(cudaMemsetAsync(scratch.d_selected_count, 0, sizeof(int), activeStream));
    collectLowestSweepIndicesKernel<<<summaryBlocks, 256, 0, activeStream>>>(
        scratch.d_radials, num_parsed, lowest_elev_num, scratch.d_indices, scratch.d_selected_count);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(&result.num_radials, scratch.d_selected_count, sizeof(int),
                               cudaMemcpyDeviceToHost, activeStream));
    CUDA_CHECK(cudaStreamSynchronize(activeStream));
    if (result.num_radials <= 0)
        return result;

    thrust::device_ptr<int> d_idx_ptr(scratch.d_indices);
    thrust::sort(thrust::cuda::par.on(activeStream), d_idx_ptr, d_idx_ptr + result.num_radials,
                 RadialIndexByAzimuth{scratch.d_radials});

    CUDA_CHECK(cudaMemsetAsync(scratch.d_product_meta, 0,
                               NUM_PRODUCTS * sizeof(DeviceProductMeta), activeStream));
    for (int p = 0; p < NUM_PRODUCTS; ++p) {
        selectProductMetaKernel<<<1, 1, 0, activeStream>>>(
            scratch.d_radials, scratch.d_indices, result.num_radials, p, scratch.d_product_meta);
    }
    CUDA_CHECK(cudaGetLastError());

    std::array<DeviceProductMeta, NUM_PRODUCTS> hostMeta = {};
    CUDA_CHECK(cudaMemcpyAsync(hostMeta.data(), scratch.d_product_meta,
                               hostMeta.size() * sizeof(DeviceProductMeta),
                               cudaMemcpyDeviceToHost, activeStream));
    CUDA_CHECK(cudaStreamSynchronize(activeStream));

    for (int p = 0; p < NUM_PRODUCTS; ++p) {
        const auto& meta = hostMeta[p];
        if (!meta.has_product || meta.num_gates <= 0)
            continue;
        result.has_product[p] = true;
        result.num_gates[p] = meta.num_gates;
        result.first_gate_km[p] = meta.first_gate / 1000.0f;
        result.gate_spacing_km[p] = meta.gate_spacing / 1000.0f;
        result.scale[p] = meta.scale;
        result.offset[p] = meta.offset;
    }

    // Allocate output buffers
    CUDA_CHECK(cudaMalloc(&result.d_azimuths, result.num_radials * sizeof(float)));

    for (int p = 0; p < NUM_PRODUCTS; p++) {
        if (result.has_product[p]) {
            size_t sz = (size_t)result.num_gates[p] * result.num_radials * sizeof(uint16_t);
            CUDA_CHECK(cudaMalloc(&result.d_gates[p], sz));

            transposeGatesGpu(scratch.d_raw, scratch.d_radials, scratch.d_indices, result.num_radials, p,
                              result.d_gates[p], result.num_gates[p],
                              result.d_azimuths, activeStream);
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(activeStream));

    return result;
}

GpuVolumeIngestResult ingestVolumeGpu(const uint8_t* h_raw_data, size_t raw_size,
                                      float min_elevation_angle,
                                      float max_elevation_angle,
                                      cudaStream_t stream) {
    GpuVolumeIngestResult volume = {};
    PipelineScratch& scratch = g_scratch;
    cudaStream_t activeStream = resolveStream(scratch, stream);
    ensureCapacity(scratch.d_raw, scratch.raw_capacity, raw_size);
    CUDA_CHECK(cudaMemcpyAsync(scratch.d_raw, h_raw_data, raw_size,
                               cudaMemcpyHostToDevice, activeStream));

    ensureCapacity(scratch.d_lowest_key, scratch.lowest_key_capacity, (size_t)1);
    ensureCapacity(scratch.d_sweep_bits, scratch.sweep_bits_capacity, (size_t)8);
    ensureCapacity(scratch.d_selected_count, scratch.selected_count_capacity, (size_t)1);
    ensureCapacity(scratch.d_product_meta, scratch.product_meta_capacity, (size_t)NUM_PRODUCTS);

    bool truncated = false;
    int num_parsed = parseWithResize(scratch.d_raw, raw_size,
                                     scratch.d_radials, scratch.radial_capacity,
                                     scratch, activeStream, &truncated);
    volume.truncated = truncated;
    volume.parsed = num_parsed > 0 && !truncated;
    if (num_parsed <= 0 || truncated)
        return volume;

    const uint32_t invalidLowestKey = 0xFFFFFFFFu;
    CUDA_CHECK(cudaMemcpyAsync(scratch.d_lowest_key, &invalidLowestKey, sizeof(uint32_t),
                               cudaMemcpyHostToDevice, activeStream));
    CUDA_CHECK(cudaMemsetAsync(scratch.d_sweep_bits, 0, 8 * sizeof(uint32_t), activeStream));

    const int summaryBlocks = (num_parsed + 255) / 256;
    scanSweepMetadataKernel<<<summaryBlocks, 256, 0, activeStream>>>(
        scratch.d_radials, num_parsed, scratch.d_lowest_key, scratch.d_sweep_bits);
    CUDA_CHECK(cudaGetLastError());

    std::array<uint32_t, 8> sweepBits = {};
    CUDA_CHECK(cudaMemcpyAsync(sweepBits.data(), scratch.d_sweep_bits, sweepBits.size() * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, activeStream));
    CUDA_CHECK(cudaStreamSynchronize(activeStream));

    for (uint32_t bits : sweepBits)
        volume.total_sweeps += countSetBits(bits);

    auto sweepNumbers = extractSweepNumbers(sweepBits);
    volume.sweeps.reserve(sweepNumbers.size());
    for (int elevation_number : sweepNumbers) {
        GpuIngestResult sweep = {};
        if (!buildSweepResult(scratch.d_raw, scratch.d_radials, scratch, activeStream,
                              num_parsed, elevation_number, sweep)) {
            freeIngestResult(sweep);
            continue;
        }
        sweep.total_sweeps = volume.total_sweeps;
        if (min_elevation_angle >= 0.0f && sweep.elevation_angle + 0.001f < min_elevation_angle) {
            freeIngestResult(sweep);
            continue;
        }
        if (max_elevation_angle >= 0.0f && sweep.elevation_angle > max_elevation_angle + 0.001f) {
            freeIngestResult(sweep);
            continue;
        }
        volume.sweeps.push_back(std::move(sweep));
    }

    std::sort(volume.sweeps.begin(), volume.sweeps.end(),
              [](const GpuIngestResult& a, const GpuIngestResult& b) {
                  if (fabsf(a.elevation_angle - b.elevation_angle) > 0.001f)
                      return a.elevation_angle < b.elevation_angle;
                  return a.elevation_number < b.elevation_number;
              });
    return volume;
}

void freeIngestResult(GpuIngestResult& result) {
    if (result.d_azimuths) { cudaFree(result.d_azimuths); result.d_azimuths = nullptr; }
    for (int p = 0; p < NUM_PRODUCTS; p++) {
        if (result.d_gates[p]) { cudaFree(result.d_gates[p]); result.d_gates[p] = nullptr; }
    }
}

void freeVolumeIngestResult(GpuVolumeIngestResult& result) {
    for (auto& sweep : result.sweeps)
        freeIngestResult(sweep);
    result.sweeps.clear();
    result.total_sweeps = 0;
}

} // namespace gpu_pipeline
