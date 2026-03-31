#include "app.h"
#include "nexrad/stations.h"
#include "nexrad/level2_parser.h"
#include "cuda/gpu_pipeline.cuh"
#include "cuda/gpu_tensor.cuh"
#include "cuda/gpu_detection.cuh"
#include "cuda/preprocess.cuh"
#include "cuda/volume3d.cuh"
#include "net/aws_nexrad.h"
#include <nlohmann/json.hpp>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <regex>
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

namespace {

using json = nlohmann::json;

constexpr float kPi = 3.14159265f;
constexpr float kDegToRad = kPi / 180.0f;
constexpr float kInvalidSample = -9999.0f;
constexpr const char* kProbSevereHost = "mrms.ncep.noaa.gov";
constexpr const char* kProbSevereIndexPath = "/ProbSevere/PROBSEVERE/";
thread_local gpu_tensor::TensorWorkspace g_detectionTensorWorkspace;
thread_local gpu_detection::CandidateWorkspace g_detectionCandidateWorkspace;
thread_local gpu_preprocess::PreprocessWorkspace g_preprocessWorkspace;

using Clock = std::chrono::steady_clock;

float elapsedMs(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<float, std::milli>(end - start).count();
}

bool extractArchiveTime(const std::string& fname, int& hh, int& mm, int& ss) {
    size_t us = fname.find('_');
    if (us == std::string::npos || us + 7 > fname.size()) return false;
    const std::string timeStr = fname.substr(us + 1, 6);
    if (timeStr.size() != 6) return false;
    hh = std::stoi(timeStr.substr(0, 2));
    mm = std::stoi(timeStr.substr(2, 2));
    ss = std::stoi(timeStr.substr(4, 2));
    return true;
}

std::string filenameFromKey(const std::string& key) {
    size_t slash = key.rfind('/');
    return (slash != std::string::npos) ? key.substr(slash + 1) : key;
}

bool isDigitSpan(const std::string& text, size_t pos, size_t count) {
    if (pos + count > text.size()) return false;
    for (size_t i = 0; i < count; i++) {
        const unsigned char ch = (unsigned char)text[pos + i];
        if (ch < '0' || ch > '9')
            return false;
    }
    return true;
}

std::string formatVolumeKeyTimestamp(const std::string& key) {
    const std::string filename = filenameFromKey(key);
    if (filename.empty()) return {};

    for (size_t i = 0; i < filename.size(); i++) {
        if (!isDigitSpan(filename, i, 8))
            continue;

        size_t timePos = std::string::npos;
        if (i + 15 <= filename.size() && filename[i + 8] == '_' && isDigitSpan(filename, i + 9, 6))
            timePos = i + 9;
        else if (i + 14 <= filename.size() && isDigitSpan(filename, i + 8, 6))
            timePos = i + 8;

        if (timePos == std::string::npos)
            continue;

        return filename.substr(i, 4) + "-" +
               filename.substr(i + 4, 2) + "-" +
               filename.substr(i + 6, 2) + " " +
               filename.substr(timePos, 2) + ":" +
               filename.substr(timePos + 2, 2) + ":" +
               filename.substr(timePos + 4, 2) + " UTC";
    }

    return filename;
}

bool extractVolumeKeyDate(const std::string& key, int& year, int& month, int& day) {
    const std::string filename = filenameFromKey(key);
    if (filename.empty()) return false;

    for (size_t i = 0; i < filename.size(); i++) {
        if (!isDigitSpan(filename, i, 8))
            continue;

        year = std::stoi(filename.substr(i, 4));
        month = std::stoi(filename.substr(i + 4, 2));
        day = std::stoi(filename.substr(i + 6, 2));
        return true;
    }

    return false;
}

std::string makeIsoUtcTimestamp(int year, int month, int day, int hour, int minute, int second = 0) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%04d-%02d-%02dT%02d:%02d:%02dZ",
                  year, month, day, hour, minute, second);
    return buffer;
}

float stationPollJitter(int stationIdx) {
    return 0.85f + 0.03f * float((stationIdx * 7) % 11);
}

size_t queryProcessWorkingSetBytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX counters = {};
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                             sizeof(counters))) {
        return (size_t)counters.WorkingSetSize;
    }
#endif
    return 0;
}

std::string buildLiveListQuery(const std::string& station, int year, int month, int day,
                               const std::string& currentKey) {
    std::string listPath = buildListUrl(station, year, month, day);
    std::string query = "/?list-type=2&prefix=" + std::string(listPath.data() + 1);
    if (!currentKey.empty())
        query += "&start-after=" + currentKey;
    else
        query += "&max-keys=1000";
    return query;
}

std::string trimCopy(std::string text) {
    auto isSpace = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!text.empty() && isSpace((unsigned char)text.front()))
        text.erase(text.begin());
    while (!text.empty() && isSpace((unsigned char)text.back()))
        text.pop_back();
    return text;
}

std::string normalizeStationCode(std::string station) {
    station = trimCopy(std::move(station));
    std::transform(station.begin(), station.end(), station.begin(),
                   [](unsigned char c) { return (char)std::toupper(c); });
    return station;
}

int findStationIndexByCode(const std::string& station) {
    for (int i = 0; i < NUM_NEXRAD_STATIONS; i++) {
        if (station == NEXRAD_STATIONS[i].icao)
            return i;
    }
    return -1;
}

std::string makeArchiveRangeLabel(const std::string& station,
                                  int year, int month, int day,
                                  int startHour, int startMin,
                                  int endHour, int endMin) {
    char buffer[96];
    std::snprintf(buffer, sizeof(buffer),
                  "Archive %04d-%02d-%02d %02d:%02d-%02d:%02d UTC",
                  year, month, day,
                  startHour, startMin, endHour, endMin);
    return buffer;
}

int findLowestSweepIndex(const std::vector<PrecomputedSweep>& sweeps) {
    int bestIdx = -1;
    float bestElev = std::numeric_limits<float>::max();
    int bestProducts = -1;
    int bestRadials = -1;

    for (int i = 0; i < (int)sweeps.size(); i++) {
        const auto& sw = sweeps[i];
        if (sw.num_radials <= 0) continue;

        int productCount = 0;
        for (int p = 0; p < NUM_PRODUCTS; p++)
            productCount += sw.products[p].has_data ? 1 : 0;

        if (bestIdx < 0 ||
            sw.elevation_angle < bestElev - 0.05f ||
            (fabsf(sw.elevation_angle - bestElev) <= 0.05f && productCount > bestProducts) ||
            (fabsf(sw.elevation_angle - bestElev) <= 0.05f && productCount == bestProducts &&
             sw.num_radials > bestRadials)) {
            bestIdx = i;
            bestElev = sw.elevation_angle;
            bestProducts = productCount;
            bestRadials = sw.num_radials;
        }
    }

    return bestIdx;
}

float decodeGateValue(const PrecomputedSweep::ProductData& pd, int num_radials,
                      int gate_idx, int radial_idx) {
    if (!pd.has_data || num_radials <= 0 ||
        gate_idx < 0 || radial_idx < 0 ||
        gate_idx >= pd.num_gates || radial_idx >= num_radials ||
        pd.gates.empty()) {
        return kInvalidSample;
    }

    uint16_t raw = pd.gates[(size_t)gate_idx * num_radials + radial_idx];
    if (raw <= 1) return kInvalidSample;
    return ((float)raw - pd.offset) / pd.scale;
}

int gateIndexForRange(const PrecomputedSweep::ProductData& pd, float range_km) {
    if (!pd.has_data || pd.gate_spacing_km <= 0.0f) return -1;
    int gate_idx = (int)((range_km - pd.first_gate_km) / pd.gate_spacing_km);
    return (gate_idx >= 0 && gate_idx < pd.num_gates) ? gate_idx : -1;
}

float markerDistanceKm(float lat1, float lon1, float lat2, float lon2) {
    float mean_lat = 0.5f * (lat1 + lat2) * kDegToRad;
    float dlat_km = (lat1 - lat2) * 111.0f;
    float dlon_km = (lon1 - lon2) * 111.0f * cosf(mean_lat);
    return sqrtf(dlat_km * dlat_km + dlon_km * dlon_km);
}

float parseJsonFloat(const json& value) {
    if (value.is_number_float()) return value.get<float>();
    if (value.is_number_integer()) return (float)value.get<int>();
    if (value.is_string()) {
        const std::string text = value.get<std::string>();
        if (text.empty()) return 0.0f;
        try { return std::stof(text); } catch (...) { return 0.0f; }
    }
    return 0.0f;
}

float parseJsonFloatField(const json& object, const char* key) {
    if (!object.is_object()) return 0.0f;
    auto it = object.find(key);
    if (it == object.end()) return 0.0f;
    return parseJsonFloat(*it);
}

std::string parseJsonStringField(const json& object, const char* key) {
    if (!object.is_object()) return {};
    auto it = object.find(key);
    if (it == object.end()) return {};
    if (it->is_string()) return it->get<std::string>();
    if (it->is_number_integer()) return std::to_string(it->get<int>());
    if (it->is_number_unsigned()) return std::to_string(it->get<unsigned int>());
    return {};
}

std::string latestProbSevereJsonNameFromIndex(const std::string& html) {
    static const std::regex kProbFileRegex(R"(MRMS_PROBSEVERE_\d{8}_\d{6}\.json)");
    std::string latest;
    for (std::sregex_iterator it(html.begin(), html.end(), kProbFileRegex), end; it != end; ++it) {
        const std::string match = it->str();
        if (match > latest)
            latest = match;
    }
    return latest;
}

float decodeVelocityGate(const PrecomputedSweep::ProductData& pd, uint16_t raw) {
    if (raw <= 1)
        return kInvalidSample;
    return ((float)raw - pd.offset) / pd.scale;
}

uint16_t encodeVelocityGate(const PrecomputedSweep::ProductData& pd, float velocity) {
    const float rawValue = velocity * pd.scale + pd.offset;
    const float clamped = std::clamp(rawValue, 2.0f, 65535.0f);
    return (uint16_t)std::lround(clamped);
}

float bestUnfoldedVelocity(float velocity, float reference, float nyquist) {
    float best = velocity;
    float bestError = fabsf(velocity - reference);

    const float candidates[2] = {
        velocity - 2.0f * nyquist,
        velocity + 2.0f * nyquist
    };
    for (float candidate : candidates) {
        const float error = fabsf(candidate - reference);
        if (error < bestError) {
            best = candidate;
            bestError = error;
        }
    }
    return best;
}

void cpuDealiasVelocityProduct(PrecomputedSweep::ProductData& velPd, int numRadials) {
    if (!velPd.has_data || velPd.num_gates <= 0 || numRadials <= 2 || velPd.gates.empty())
        return;

    const int nr = numRadials;
    const int ng = velPd.num_gates;
    const float nyquist = 30.0f;
    const float maxNeighborSpread = nyquist * 0.75f;
    const float minImprovement = nyquist * 0.35f;
    const float maxResidual = nyquist * 0.45f;

    std::vector<uint16_t> source = velPd.gates;
    std::vector<uint16_t> corrected = source;

    for (int gi = 0; gi < ng; gi++) {
        for (int ri = 0; ri < nr; ri++) {
            const size_t idx = (size_t)gi * nr + ri;
            const float velocity = decodeVelocityGate(velPd, source[idx]);
            if (velocity <= -998.0f)
                continue;

            const int riPrev = (ri - 1 + nr) % nr;
            const int riNext = (ri + 1) % nr;
            const float prevVelocity = decodeVelocityGate(velPd, source[(size_t)gi * nr + riPrev]);
            const float nextVelocity = decodeVelocityGate(velPd, source[(size_t)gi * nr + riNext]);
            if (prevVelocity <= -998.0f || nextVelocity <= -998.0f)
                continue;

            if (fabsf(prevVelocity - nextVelocity) > maxNeighborSpread)
                continue;

            const float reference = 0.5f * (prevVelocity + nextVelocity);
            const float baseError = fabsf(velocity - reference);
            const float unfolded = bestUnfoldedVelocity(velocity, reference, nyquist);
            const float unfoldedError = fabsf(unfolded - reference);

            if (unfoldedError >= baseError - minImprovement)
                continue;
            if (unfoldedError > maxResidual)
                continue;

            corrected[idx] = encodeVelocityGate(velPd, unfolded);
        }
    }

    velPd.gates.swap(corrected);
}

int countCandidateSupport(const std::vector<uint8_t>& mask,
                         int nr, int ng,
                         int ri, int gi,
                         int radial_radius, int gate_radius) {
    int support = 0;
    for (int dgi = -gate_radius; dgi <= gate_radius; ++dgi) {
        int ngi = gi + dgi;
        if (ngi < 0 || ngi >= ng) continue;
        for (int dri = -radial_radius; dri <= radial_radius; ++dri) {
            int nri = (ri + dri + nr) % nr;
            support += mask[(size_t)ngi * nr + nri] ? 1 : 0;
        }
    }
    return support;
}

bool isLocalExtremum(const std::vector<float>& score,
                    const std::vector<uint8_t>& mask,
                    int nr, int ng,
                    int ri, int gi,
                    int radial_radius, int gate_radius,
                    bool lower_is_better) {
    const float center = score[(size_t)gi * nr + ri];
    for (int dgi = -gate_radius; dgi <= gate_radius; ++dgi) {
        int ngi = gi + dgi;
        if (ngi < 0 || ngi >= ng) continue;
        for (int dri = -radial_radius; dri <= radial_radius; ++dri) {
            int nri = (ri + dri + nr) % nr;
            if (ngi == gi && nri == ri) continue;
            if (!mask[(size_t)ngi * nr + nri]) continue;
            float neighbor = score[(size_t)ngi * nr + nri];
            if (lower_is_better) {
                if (neighbor < center - 0.01f) return false;
            } else {
                if (neighbor > center + 0.01f) return false;
            }
        }
    }
    return true;
}

void clusterMarkers(std::vector<Detection::Marker>& markers,
                    float merge_km, size_t max_markers,
                    bool lower_value_is_stronger = false) {
    if (markers.empty()) return;

    std::sort(markers.begin(), markers.end(),
              [lower_value_is_stronger](const Detection::Marker& a,
                                        const Detection::Marker& b) {
                  return lower_value_is_stronger ? (a.value < b.value)
                                                 : (a.value > b.value);
              });

    std::vector<Detection::Marker> clustered;
    clustered.reserve(std::min(markers.size(), max_markers));
    for (const auto& marker : markers) {
        bool keep = true;
        for (const auto& existing : clustered) {
            if (markerDistanceKm(marker.lat, marker.lon, existing.lat, existing.lon) < merge_km) {
                keep = false;
                break;
            }
        }
        if (keep) {
            clustered.push_back(marker);
            if (clustered.size() >= max_markers) break;
        }
    }
    markers.swap(clustered);
}

void clusterMesoMarkers(std::vector<Detection::MesoMarker>& markers,
                        float merge_km, size_t max_markers) {
    if (markers.empty()) return;

    std::sort(markers.begin(), markers.end(),
              [](const Detection::MesoMarker& a, const Detection::MesoMarker& b) {
                  return a.shear > b.shear;
              });

    std::vector<Detection::MesoMarker> clustered;
    clustered.reserve(std::min(markers.size(), max_markers));
    for (const auto& marker : markers) {
        bool keep = true;
        for (const auto& existing : clustered) {
            if (markerDistanceKm(marker.lat, marker.lon, existing.lat, existing.lon) < merge_km) {
                keep = false;
                break;
            }
        }
        if (keep) {
            clustered.push_back(marker);
            if (clustered.size() >= max_markers) break;
        }
    }
    markers.swap(clustered);
}

void cpuSuppressReflectivityRingArtifacts(std::vector<PrecomputedSweep>& sweeps) {
    constexpr float kCoverageStrong = 0.95f;
    constexpr float kCoverageLoose = 0.88f;
    constexpr float kStdStrong = 12.0f;
    constexpr float kStdLoose = 6.0f;
    constexpr float kMaxRangeKm = 160.0f;

    for (auto& sweep : sweeps) {
        auto& pd = sweep.products[PROD_REF];
        if (!pd.has_data || pd.num_gates <= 0 || sweep.num_radials < 300 || pd.gates.empty())
            continue;

        const int nr = sweep.num_radials;
        const int ng = pd.num_gates;
        std::vector<uint8_t> suppressGate((size_t)ng, 0);

        for (int gi = 0; gi < ng; ++gi) {
            const float range_km = pd.first_gate_km + gi * pd.gate_spacing_km;
            if (range_km > kMaxRangeKm) break;

            int valid = 0;
            float sum = 0.0f;
            float sum2 = 0.0f;
            for (int ri = 0; ri < nr; ++ri) {
                uint16_t raw = pd.gates[(size_t)gi * nr + ri];
                if (raw <= 1) continue;
                float value = ((float)raw - pd.offset) / pd.scale;
                valid++;
                sum += value;
                sum2 += value * value;
            }

            if (valid < nr / 2) continue;

            const float coverage = (float)valid / (float)nr;
            const float mean = sum / (float)valid;
            const float variance = fmaxf(sum2 / (float)valid - mean * mean, 0.0f);
            const float stddev = sqrtf(variance);

            const bool strongRing = coverage >= kCoverageStrong && mean >= 10.0f && stddev <= kStdStrong;
            const bool looseRing = coverage >= kCoverageLoose && mean >= 20.0f && stddev <= kStdLoose;
            if (strongRing || looseRing)
                suppressGate[gi] = 1;
        }

        for (int gi = 0; gi < ng; ++gi) {
            if (!suppressGate[(size_t)gi]) continue;
            for (int ri = 0; ri < nr; ++ri)
                pd.gates[(size_t)gi * nr + ri] = 0;
        }
    }
}

void dealiasVelocityProduct(PrecomputedSweep::ProductData& velPd, int numRadials) {
    if (!gpu_preprocess::dealiasVelocity(velPd, numRadials, &g_preprocessWorkspace))
        cpuDealiasVelocityProduct(velPd, numRadials);
}

void suppressReflectivityRingArtifacts(std::vector<PrecomputedSweep>& sweeps) {
    if (!gpu_preprocess::suppressReflectivityRings(sweeps, &g_preprocessWorkspace))
        cpuSuppressReflectivityRingArtifacts(sweeps);
}

bool parsedSweepHasProduct(const ParsedSweep& sweep, int product) {
    for (const auto& radial : sweep.radials) {
        for (const auto& moment : radial.moments) {
            if (moment.product_index == product)
                return true;
        }
    }
    return false;
}

int findLowestParsedSweepIndex(const ParsedRadarData& parsed) {
    int bestIdx = -1;
    float bestElev = std::numeric_limits<float>::max();
    int bestProducts = -1;
    int bestRadials = -1;

    for (int i = 0; i < (int)parsed.sweeps.size(); i++) {
        const auto& sweep = parsed.sweeps[i];
        if (sweep.radials.empty()) continue;

        int productCount = 0;
        for (int p = 0; p < NUM_PRODUCTS; p++)
            productCount += parsedSweepHasProduct(sweep, p) ? 1 : 0;

        if (bestIdx < 0 ||
            sweep.elevation_angle < bestElev - 0.05f ||
            (fabsf(sweep.elevation_angle - bestElev) <= 0.05f && productCount > bestProducts) ||
            (fabsf(sweep.elevation_angle - bestElev) <= 0.05f && productCount == bestProducts &&
             (int)sweep.radials.size() > bestRadials)) {
            bestIdx = i;
            bestElev = sweep.elevation_angle;
            bestProducts = productCount;
            bestRadials = (int)sweep.radials.size();
        }
    }

    return bestIdx;
}

PrecomputedSweep buildPrecomputedSweep(const ParsedSweep& sweep) {
    PrecomputedSweep pc;
    pc.elevation_angle = sweep.elevation_angle;
    pc.num_radials = (int)sweep.radials.size();
    if (pc.num_radials == 0)
        return pc;

    pc.azimuths.resize(pc.num_radials);
    for (int r = 0; r < pc.num_radials; r++)
        pc.azimuths[r] = sweep.radials[r].azimuth;

    for (const auto& radial : sweep.radials) {
        for (const auto& moment : radial.moments) {
            int p = moment.product_index;
            if (p < 0 || p >= NUM_PRODUCTS) continue;
            auto& pd = pc.products[p];
            if (!pd.has_data || moment.num_gates > pd.num_gates) {
                pd.has_data = true;
                pd.num_gates = moment.num_gates;
                pd.first_gate_km = moment.first_gate_m / 1000.0f;
                pd.gate_spacing_km = moment.gate_spacing_m / 1000.0f;
                pd.scale = moment.scale;
                pd.offset = moment.offset;
            }
        }
    }

    for (int p = 0; p < NUM_PRODUCTS; p++) {
        auto& pd = pc.products[p];
        if (!pd.has_data || pd.num_gates <= 0) continue;

        int ng = pd.num_gates;
        int nr = pc.num_radials;
        pd.gates.assign((size_t)ng * nr, 0);

        for (int r = 0; r < nr; r++) {
            for (const auto& mom : sweep.radials[r].moments) {
                if (mom.product_index != p) continue;
                int gc = std::min((int)mom.gates.size(), ng);
                for (int g = 0; g < gc; g++)
                    pd.gates[(size_t)g * nr + r] = mom.gates[g];
                break;
            }
        }
    }

    return pc;
}

std::vector<PrecomputedSweep> buildPrecomputedSweeps(const ParsedRadarData& parsed) {
    std::vector<PrecomputedSweep> precomp;
    precomp.resize(parsed.sweeps.size());

    for (int si = 0; si < (int)parsed.sweeps.size(); si++) {
        precomp[si] = buildPrecomputedSweep(parsed.sweeps[si]);
    }

    return precomp;
}

std::vector<PrecomputedSweep> buildReducedWorkingSetSweeps(const ParsedRadarData& parsed) {
    std::vector<int> selected;
    const int lowestIdx = findLowestParsedSweepIndex(parsed);
    if (lowestIdx >= 0)
        selected.push_back(lowestIdx);

    for (int p = 0; p < NUM_PRODUCTS; ++p) {
        for (int i = 0; i < (int)parsed.sweeps.size(); ++i) {
            const auto& sweep = parsed.sweeps[i];
            if (sweep.elevation_angle > 1.5f)
                continue;
            if (!parsedSweepHasProduct(sweep, p))
                continue;
            if (std::find(selected.begin(), selected.end(), i) == selected.end())
                selected.push_back(i);
            break;
        }
    }

    std::sort(selected.begin(), selected.end(), [&](int a, int b) {
        const auto& sa = parsed.sweeps[a];
        const auto& sb = parsed.sweeps[b];
        if (fabsf(sa.elevation_angle - sb.elevation_angle) > 0.05f)
            return sa.elevation_angle < sb.elevation_angle;
        return sa.sweep_number < sb.sweep_number;
    });

    std::vector<PrecomputedSweep> sweeps;
    sweeps.reserve(selected.size());
    for (int idx : selected)
        sweeps.push_back(buildPrecomputedSweep(parsed.sweeps[idx]));
    return sweeps;
}

std::vector<PrecomputedSweep> buildReducedWorkingSetSweeps(std::vector<PrecomputedSweep> sweeps) {
    if (sweeps.empty())
        return sweeps;

    std::vector<int> selected;
    int lowestIdx = findLowestSweepIndex(sweeps);
    if (lowestIdx >= 0)
        selected.push_back(lowestIdx);

    for (int p = 0; p < NUM_PRODUCTS; ++p) {
        for (int i = 0; i < (int)sweeps.size(); ++i) {
            const auto& sweep = sweeps[i];
            if (sweep.elevation_angle > 1.5f)
                continue;
            if (!sweep.products[p].has_data || sweep.num_radials <= 0)
                continue;
            if (std::find(selected.begin(), selected.end(), i) == selected.end())
                selected.push_back(i);
            break;
        }
    }

    std::sort(selected.begin(), selected.end(), [&](int a, int b) {
        const auto& sa = sweeps[a];
        const auto& sb = sweeps[b];
        if (fabsf(sa.elevation_angle - sb.elevation_angle) > 0.05f)
            return sa.elevation_angle < sb.elevation_angle;
        return a < b;
    });

    std::vector<PrecomputedSweep> reduced;
    reduced.reserve(selected.size());
    for (int idx : selected)
        reduced.push_back(std::move(sweeps[idx]));
    return reduced;
}

PrecomputedSweep buildPrecomputedSweep(const gpu_pipeline::GpuIngestResult& ingest) {
    PrecomputedSweep sweep;
    sweep.elevation_angle = ingest.elevation_angle;
    sweep.num_radials = ingest.num_radials;
    if (ingest.num_radials <= 0 || !ingest.d_azimuths)
        return sweep;

    sweep.azimuths.resize(ingest.num_radials);
    CUDA_CHECK(cudaMemcpy(sweep.azimuths.data(), ingest.d_azimuths,
                          (size_t)ingest.num_radials * sizeof(float), cudaMemcpyDeviceToHost));

    for (int p = 0; p < NUM_PRODUCTS; ++p) {
        if (!ingest.has_product[p] || !ingest.d_gates[p] || ingest.num_gates[p] <= 0)
            continue;

        auto& pd = sweep.products[p];
        pd.has_data = true;
        pd.num_gates = ingest.num_gates[p];
        pd.first_gate_km = ingest.first_gate_km[p];
        pd.gate_spacing_km = ingest.gate_spacing_km[p];
        pd.scale = ingest.scale[p];
        pd.offset = ingest.offset[p];
        pd.gates.resize((size_t)pd.num_gates * (size_t)ingest.num_radials);
        CUDA_CHECK(cudaMemcpy(pd.gates.data(), ingest.d_gates[p],
                              pd.gates.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    }

    return sweep;
}

enum class GpuWorkingSetMode {
    LowestSweep,
    LowTilts,
    AllSweeps
};

struct FastGpuWorkingSet {
    std::vector<PrecomputedSweep> sweeps;
    int total_sweeps = 0;
    bool success = false;
};

FastGpuWorkingSet buildGpuWorkingSetFromDecoded(const std::vector<uint8_t>& decodedBytes,
                                                GpuWorkingSetMode mode) {
    FastGpuWorkingSet workingSet;
    if (decodedBytes.empty())
        return workingSet;

    if (mode == GpuWorkingSetMode::LowestSweep) {
        auto ingest = gpu_pipeline::ingestSweepGpu(decodedBytes.data(), decodedBytes.size());
        if (!ingest.d_azimuths || ingest.num_radials <= 0) {
            gpu_pipeline::freeIngestResult(ingest);
            return workingSet;
        }
        workingSet.total_sweeps = std::max(ingest.total_sweeps, 1);
        workingSet.sweeps.push_back(buildPrecomputedSweep(ingest));
        gpu_pipeline::freeIngestResult(ingest);
        workingSet.success = !workingSet.sweeps.empty();
        return workingSet;
    }

    const float maxElevation = (mode == GpuWorkingSetMode::LowTilts) ? 1.5f : -1.0f;
    auto volume = gpu_pipeline::ingestVolumeGpu(decodedBytes.data(), decodedBytes.size(), maxElevation);
    if (volume.sweeps.empty()) {
        gpu_pipeline::freeVolumeIngestResult(volume);
        return workingSet;
    }

    workingSet.total_sweeps = std::max(volume.total_sweeps, (int)volume.sweeps.size());
    workingSet.sweeps.reserve(volume.sweeps.size());
    for (const auto& ingest : volume.sweeps)
        workingSet.sweeps.push_back(buildPrecomputedSweep(ingest));
    gpu_pipeline::freeVolumeIngestResult(volume);
    if (mode == GpuWorkingSetMode::LowTilts)
        workingSet.sweeps = buildReducedWorkingSetSweeps(std::move(workingSet.sweeps));
    workingSet.success = !workingSet.sweeps.empty();
    return workingSet;
}

void dealiasPrecomputedSweeps(std::vector<PrecomputedSweep>& sweeps) {
    for (auto& pc : sweeps) {
        auto& velPd = pc.products[PROD_VEL];
        dealiasVelocityProduct(velPd, pc.num_radials);
    }
}

Detection computeDetectionForSweepsCpu(const std::vector<PrecomputedSweep>& sweeps,
                                       float slat, float slon,
                                       const std::string& stationName) {
    Detection det;
    if (sweeps.empty())
        return det;

    det.computed = true;
    float cos_lat = std::max(cosf(slat * kDegToRad), 0.1f);

    int refSweep = -1, ccSweep = -1, zdrSweep = -1, velSweep = -1;
    for (int s = 0; s < (int)sweeps.size(); s++) {
        const auto& pc = sweeps[s];
        if (pc.elevation_angle > 1.5f) continue;
        if (pc.products[PROD_REF].has_data && refSweep < 0) refSweep = s;
        if (pc.products[PROD_CC].has_data && ccSweep < 0) ccSweep = s;
        if (pc.products[PROD_ZDR].has_data && zdrSweep < 0) zdrSweep = s;
        if (pc.products[PROD_VEL].has_data && velSweep < 0) velSweep = s;
    }

    if (ccSweep >= 0 && zdrSweep >= 0 && refSweep >= 0) {
        const auto& ccPc = sweeps[ccSweep];
        const auto& zdrPc = sweeps[zdrSweep];
        const auto& refPc = sweeps[refSweep];
        const auto& ccPd = ccPc.products[PROD_CC];
        const auto& zdrPd = zdrPc.products[PROD_ZDR];
        const auto& refPd = refPc.products[PROD_REF];

        int nr = ccPc.num_radials;
        int ng = ccPd.num_gates;
        if (nr > 0 && ng > 0) {
            std::vector<uint8_t> candidate((size_t)nr * ng, 0);
            std::vector<float> score((size_t)nr * ng, std::numeric_limits<float>::infinity());

            for (int ri = 0; ri < nr; ++ri) {
                int zdr_ri = std::min((int)((int64_t)ri * zdrPc.num_radials / std::max(nr, 1)),
                                      std::max(zdrPc.num_radials - 1, 0));
                int ref_ri = std::min((int)((int64_t)ri * refPc.num_radials / std::max(nr, 1)),
                                      std::max(refPc.num_radials - 1, 0));
                for (int gi = 0; gi < ng; gi += 2) {
                    float range_km = ccPd.first_gate_km + gi * ccPd.gate_spacing_km;
                    if (range_km < 15.0f || range_km > 120.0f) continue;

                    float cc = decodeGateValue(ccPd, nr, gi, ri);
                    if (cc == kInvalidSample || cc < 0.55f || cc > 0.82f) continue;

                    int zdr_gi = gateIndexForRange(zdrPd, range_km);
                    int ref_gi = gateIndexForRange(refPd, range_km);
                    if (zdr_gi < 0 || ref_gi < 0) continue;

                    float zdr = decodeGateValue(zdrPd, zdrPc.num_radials, zdr_gi, zdr_ri);
                    float ref = decodeGateValue(refPd, refPc.num_radials, ref_gi, ref_ri);
                    if (zdr == kInvalidSample || ref == kInvalidSample) continue;
                    if (fabsf(zdr) > 1.25f || ref < 40.0f) continue;

                    candidate[(size_t)gi * nr + ri] = 1;
                    score[(size_t)gi * nr + ri] = cc;
                }
            }

            for (int ri = 0; ri < nr; ++ri) {
                float az_rad = ccPc.azimuths[ri] * kDegToRad;
                for (int gi = 0; gi < ng; gi += 2) {
                    if (!candidate[(size_t)gi * nr + ri]) continue;
                    if (countCandidateSupport(candidate, nr, ng, ri, gi, 2, 2) < 6) continue;
                    if (!isLocalExtremum(score, candidate, nr, ng, ri, gi, 2, 2, true)) continue;

                    float range_km = ccPd.first_gate_km + gi * ccPd.gate_spacing_km;
                    float east_km = range_km * sinf(az_rad);
                    float north_km = range_km * cosf(az_rad);
                    det.tds.push_back({
                        slat + north_km / 111.0f,
                        slon + east_km / (111.0f * cos_lat),
                        score[(size_t)gi * nr + ri]
                    });
                }
            }
            clusterMarkers(det.tds, 8.0f, 12, true);
        }
    }

    if (refSweep >= 0 && zdrSweep >= 0) {
        const auto& refPc = sweeps[refSweep];
        const auto& zdrPc = sweeps[zdrSweep];
        const auto& refPd = refPc.products[PROD_REF];
        const auto& zdrPd = zdrPc.products[PROD_ZDR];

        int nr = refPc.num_radials;
        int ng = refPd.num_gates;
        if (nr > 0 && ng > 0) {
            std::vector<uint8_t> candidate((size_t)nr * ng, 0);
            std::vector<float> score((size_t)nr * ng, -std::numeric_limits<float>::infinity());

            for (int ri = 0; ri < nr; ++ri) {
                int zdr_ri = std::min((int)((int64_t)ri * zdrPc.num_radials / std::max(nr, 1)),
                                      std::max(zdrPc.num_radials - 1, 0));
                for (int gi = 0; gi < ng; gi += 2) {
                    float range_km = refPd.first_gate_km + gi * refPd.gate_spacing_km;
                    if (range_km < 15.0f || range_km > 180.0f) continue;

                    float ref = decodeGateValue(refPd, nr, gi, ri);
                    if (ref == kInvalidSample || ref < 55.0f) continue;

                    int zdr_gi = gateIndexForRange(zdrPd, range_km);
                    if (zdr_gi < 0) continue;
                    float zdr = decodeGateValue(zdrPd, zdrPc.num_radials, zdr_gi, zdr_ri);
                    if (zdr == kInvalidSample) continue;

                    float hdr = ref - (19.0f * std::max(zdr, 0.0f) + 27.0f);
                    if (hdr < 10.0f) continue;

                    candidate[(size_t)gi * nr + ri] = 1;
                    score[(size_t)gi * nr + ri] = hdr;
                }
            }

            for (int ri = 0; ri < nr; ++ri) {
                float az_rad = refPc.azimuths[ri] * kDegToRad;
                for (int gi = 0; gi < ng; gi += 2) {
                    if (!candidate[(size_t)gi * nr + ri]) continue;
                    if (countCandidateSupport(candidate, nr, ng, ri, gi, 2, 2) < 5) continue;
                    if (!isLocalExtremum(score, candidate, nr, ng, ri, gi, 2, 2, false)) continue;

                    float range_km = refPd.first_gate_km + gi * refPd.gate_spacing_km;
                    float east_km = range_km * sinf(az_rad);
                    float north_km = range_km * cosf(az_rad);
                    det.hail.push_back({
                        slat + north_km / 111.0f,
                        slon + east_km / (111.0f * cos_lat),
                        score[(size_t)gi * nr + ri]
                    });
                }
            }
            clusterMarkers(det.hail, 10.0f, 16, false);
        }
    }

    if (velSweep >= 0) {
        const auto& velPc = sweeps[velSweep];
        const auto& velPd = velPc.products[PROD_VEL];
        int nr = velPc.num_radials;
        int ng = velPd.num_gates;

        if (nr >= 10 && ng >= 10) {
            std::vector<uint8_t> candidate((size_t)nr * ng, 0);
            std::vector<float> score((size_t)nr * ng, -std::numeric_limits<float>::infinity());
            std::vector<float> diameter((size_t)nr * ng, 0.0f);

            auto passesMesoGate = [&](int gate_idx, int radial_idx,
                                      float range_km, float* shear_out,
                                      float* span_out) -> bool {
                int span = 2;
                int ri_lo = (radial_idx - span + nr) % nr;
                int ri_hi = (radial_idx + span) % nr;

                float v_lo = decodeGateValue(velPd, nr, gate_idx, ri_lo);
                float v_hi = decodeGateValue(velPd, nr, gate_idx, ri_hi);
                if (v_lo == kInvalidSample || v_hi == kInvalidSample) return false;
                if (fabsf(v_lo) < 12.0f || fabsf(v_hi) < 12.0f) return false;
                if (v_lo * v_hi >= 0.0f) return false;

                float shear_ms = fabsf(v_hi - v_lo);
                if (shear_ms < 40.0f) return false;

                float az_span_deg = span * 2.0f * (360.0f / nr);
                float az_span_km = range_km * az_span_deg * kDegToRad;
                if (az_span_km < 1.0f || az_span_km > 10.0f) return false;

                *shear_out = shear_ms;
                *span_out = az_span_km;
                return true;
            };

            for (int gi = 12; gi < ng - 12; gi += 4) {
                float range_km = velPd.first_gate_km + gi * velPd.gate_spacing_km;
                if (range_km < 20.0f || range_km > 120.0f) continue;

                for (int ri = 0; ri < nr; ri += 2) {
                    if (refSweep >= 0) {
                        const auto& refPc = sweeps[refSweep];
                        const auto& refPd = refPc.products[PROD_REF];
                        int ref_ri = std::min((int)((int64_t)ri * refPc.num_radials / std::max(nr, 1)),
                                              std::max(refPc.num_radials - 1, 0));
                        int ref_gi = gateIndexForRange(refPd, range_km);
                        float ref = decodeGateValue(refPd, refPc.num_radials, ref_gi, ref_ri);
                        if (ref == kInvalidSample || ref < 35.0f) continue;
                    }

                    float shear_ms = 0.0f;
                    float az_span_km = 0.0f;
                    if (!passesMesoGate(gi, ri, range_km, &shear_ms, &az_span_km)) continue;

                    int gate_support = 0;
                    for (int dgi = -2; dgi <= 2; ++dgi) {
                        int ngi = gi + dgi;
                        if (ngi < 0 || ngi >= ng) continue;
                        float neighbor_shear = 0.0f;
                        float neighbor_span = 0.0f;
                        float neighbor_range = velPd.first_gate_km + ngi * velPd.gate_spacing_km;
                        if (passesMesoGate(ngi, ri, neighbor_range, &neighbor_shear, &neighbor_span))
                            ++gate_support;
                    }
                    if (gate_support < 3) continue;

                    candidate[(size_t)gi * nr + ri] = 1;
                    score[(size_t)gi * nr + ri] = shear_ms;
                    diameter[(size_t)gi * nr + ri] = az_span_km;
                }
            }

            for (int gi = 12; gi < ng - 12; gi += 4) {
                for (int ri = 0; ri < nr; ri += 2) {
                    if (!candidate[(size_t)gi * nr + ri]) continue;
                    if (countCandidateSupport(candidate, nr, ng, ri, gi, 2, 1) < 3) continue;
                    if (!isLocalExtremum(score, candidate, nr, ng, ri, gi, 2, 1, false)) continue;

                    float range_km = velPd.first_gate_km + gi * velPd.gate_spacing_km;
                    float az_rad = velPc.azimuths[ri] * kDegToRad;
                    float east_km = range_km * sinf(az_rad);
                    float north_km = range_km * cosf(az_rad);
                    det.meso.push_back({
                        slat + north_km / 111.0f,
                        slon + east_km / (111.0f * cos_lat),
                        score[(size_t)gi * nr + ri],
                        diameter[(size_t)gi * nr + ri]
                    });
                }
            }
            clusterMesoMarkers(det.meso, 12.0f, 12);
        }
    }

    printf("Detection [%s]: %d TDS, %d hail, %d meso\n",
           stationName.c_str(), (int)det.tds.size(), (int)det.hail.size(), (int)det.meso.size());
    return det;
}

Detection buildDetectionFromCandidates(const gpu_detection::HostDetectionResults& fields,
                                       float slat, float slon,
                                       const std::string& stationName) {
    Detection det;
    if (fields.num_radials <= 0 || fields.num_gates <= 0 || fields.azimuths.empty())
        return det;

    det.computed = true;
    const float cos_lat = std::max(cosf(slat * kDegToRad), 0.1f);

    for (const auto& candidate : fields.tds) {
        if (candidate.radial_idx < 0 || candidate.radial_idx >= (int)fields.azimuths.size())
            continue;
        const float az_rad = fields.azimuths[candidate.radial_idx] * kDegToRad;
        const float range_km = fields.first_gate_km + candidate.gate_idx * fields.gate_spacing_km;
        const float east_km = range_km * sinf(az_rad);
        const float north_km = range_km * cosf(az_rad);
        det.tds.push_back({
            slat + north_km / 111.0f,
            slon + east_km / (111.0f * cos_lat),
            candidate.score
        });
    }

    for (const auto& candidate : fields.hail) {
        if (candidate.radial_idx < 0 || candidate.radial_idx >= (int)fields.azimuths.size())
            continue;
        const float az_rad = fields.azimuths[candidate.radial_idx] * kDegToRad;
        const float range_km = fields.first_gate_km + candidate.gate_idx * fields.gate_spacing_km;
        const float east_km = range_km * sinf(az_rad);
        const float north_km = range_km * cosf(az_rad);
        det.hail.push_back({
            slat + north_km / 111.0f,
            slon + east_km / (111.0f * cos_lat),
            candidate.score
        });
    }

    for (const auto& candidate : fields.meso) {
        if (candidate.radial_idx < 0 || candidate.radial_idx >= (int)fields.azimuths.size())
            continue;
        const float az_rad = fields.azimuths[candidate.radial_idx] * kDegToRad;
        const float range_km = fields.first_gate_km + candidate.gate_idx * fields.gate_spacing_km;
        const float east_km = range_km * sinf(az_rad);
        const float north_km = range_km * cosf(az_rad);
        det.meso.push_back({
            slat + north_km / 111.0f,
            slon + east_km / (111.0f * cos_lat),
            candidate.score,
            candidate.aux
        });
    }

    clusterMarkers(det.tds, 8.0f, 12, true);
    clusterMarkers(det.hail, 10.0f, 16, false);
    clusterMesoMarkers(det.meso, 12.0f, 12);
    printf("Detection [%s]: %d TDS, %d hail, %d meso\n",
           stationName.c_str(), (int)det.tds.size(), (int)det.hail.size(), (int)det.meso.size());
    return det;
}

Detection computeDetectionForSweeps(const std::vector<PrecomputedSweep>& sweeps,
                                    float slat, float slon,
                                    const std::string& stationName) {
    int refSweep = -1, ccSweep = -1, zdrSweep = -1, velSweep = -1;
    for (int s = 0; s < (int)sweeps.size(); s++) {
        const auto& pc = sweeps[s];
        if (pc.elevation_angle > 1.5f) continue;
        if (pc.products[PROD_REF].has_data && refSweep < 0) refSweep = s;
        if (pc.products[PROD_CC].has_data && ccSweep < 0) ccSweep = s;
        if (pc.products[PROD_ZDR].has_data && zdrSweep < 0) zdrSweep = s;
        if (pc.products[PROD_VEL].has_data && velSweep < 0) velSweep = s;
    }

    gpu_tensor::SweepInput inputs[gpu_tensor::NUM_TENSOR_PRODUCTS] = {};
    if (refSweep >= 0) inputs[gpu_tensor::SLOT_REF] = gpu_tensor::makeSweepInput(sweeps[refSweep], PROD_REF);
    if (velSweep >= 0) inputs[gpu_tensor::SLOT_VEL] = gpu_tensor::makeSweepInput(sweeps[velSweep], PROD_VEL);
    if (zdrSweep >= 0) inputs[gpu_tensor::SLOT_ZDR] = gpu_tensor::makeSweepInput(sweeps[zdrSweep], PROD_ZDR);
    if (ccSweep >= 0) inputs[gpu_tensor::SLOT_CC] = gpu_tensor::makeSweepInput(sweeps[ccSweep], PROD_CC);

    gpu_detection::HostDetectionResults fields;
    bool gpuOk = false;

    try {
        gpuOk = g_detectionTensorWorkspace.build(inputs);
        if (gpuOk)
            gpuOk = gpu_detection::computeDetectionCandidates(g_detectionTensorWorkspace.tensor(),
                                                              g_detectionCandidateWorkspace,
                                                              fields,
                                                              g_detectionTensorWorkspace.stream());
    } catch (...) {
        gpuOk = false;
    }

    if (gpuOk)
        return buildDetectionFromCandidates(fields, slat, slon, stationName);
    return computeDetectionForSweepsCpu(sweeps, slat, slon, stationName);
}

} // namespace

static int findProductSweep(const std::vector<PrecomputedSweep>& sweeps, int product, int tiltIdx);
static int countProductSweeps(const std::vector<PrecomputedSweep>& sweeps, int product);

App::App()
    : m_spatialGrid(std::make_unique<SpatialGrid>()) {
    m_priorityStatus = "Startup seed pending";
}

App::~App() {
    if (m_downloader) m_downloader->shutdown();
    m_historic.cancel();
    m_warnings.stop();
    invalidateFrameCache(true);
    if (m_d_xsOutput) cudaFree(m_d_xsOutput);
    if (m_d_compositeOutput) cudaFree(m_d_compositeOutput);
    gpu::freeVolume();
    m_xsTex.destroy();
    m_outputTex.destroy();
    gpu::shutdown();
}

bool App::init(int windowWidth, int windowHeight,
               int framebufferWidth, int framebufferHeight) {
    m_windowWidth = framebufferWidth;
    m_windowHeight = framebufferHeight;

    // Initialize CUDA renderer
    gpu::init();

    int device = 0;
    cudaDeviceProp prop = {};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    m_gpuName = prop.name;
    m_gpuTotalMemoryBytes = (size_t)prop.totalGlobalMem;
    m_lastMemorySample = std::chrono::steady_clock::now();
    applyPerformanceProfile(true);

    // Set up viewport centered on CONUS
    m_viewport.center_lat = 39.0;
    m_viewport.center_lon = -98.0;
    m_viewport.zoom = 28.0; // pixels per degree - shows full CONUS
    m_viewport.width = windowWidth;
    m_viewport.height = windowHeight;
    ensureRenderTargets();

    // Initialize station states
    m_stationsTotal = NUM_NEXRAD_STATIONS;
    m_stations.resize(m_stationsTotal);
    for (int i = 0; i < m_stationsTotal; i++) {
        auto& s = m_stations[i];
        s.index = i;
        s.icao = NEXRAD_STATIONS[i].icao;
        s.lat = NEXRAD_STATIONS[i].lat;
        s.lon = NEXRAD_STATIONS[i].lon;
    }

    // Create downloader with 48 concurrent threads
    m_downloader = std::make_unique<Downloader>(48);

    // Start downloading all stations
    startDownloads();

    gpu::initVolume();
    m_warnings.startPolling();
    m_lastRefresh = std::chrono::steady_clock::now();
    m_lastLivePollSweep = m_lastRefresh;
    updateMemoryTelemetry(true);
    resetMemoryPeaks();

    printf("App initialized: %d stations, viewport %dx%d\n",
           m_stationsTotal, windowWidth, windowHeight);
    return true;
}

PerformanceProfile App::recommendedPerformanceProfile() const {
    if (m_gpuTotalMemoryBytes > 0 && m_gpuTotalMemoryBytes <= (size_t(8) << 30))
        return PerformanceProfile::Performance;
    if (m_gpuTotalMemoryBytes > 0 && m_gpuTotalMemoryBytes <= (size_t(12) << 30))
        return PerformanceProfile::Balanced;
    return PerformanceProfile::Quality;
}

int App::renderWidth() const {
    return std::max(1, (int)std::lround((double)m_windowWidth * m_renderScale));
}

int App::renderHeight() const {
    return std::max(1, (int)std::lround((double)m_windowHeight * m_renderScale));
}

int App::historicFrameCacheLimit() const {
    switch (m_effectivePerformanceProfile) {
        case PerformanceProfile::Performance: return 0;
        case PerformanceProfile::Balanced: return 12;
        case PerformanceProfile::Auto:
        case PerformanceProfile::Quality:
        default: return MAX_CACHED_FRAMES;
    }
}

bool App::historicFrameCachingEnabled() const {
    return historicFrameCacheLimit() > 0;
}

bool App::shouldKeepStationFullVolumeLocked(int stationIdx, int focusIdx) const {
    if (stationIdx < 0 || stationIdx >= (int)m_stations.size())
        return false;
    if (m_snapshotMode || m_historicMode)
        return true;
    if (stationIdx == focusIdx || stationIdx == m_activeStationIdx)
        return true;
    if (!m_showAll && m_viewport.zoom >= 80.0 && stationLikelyVisible(stationIdx))
        return true;
    return false;
}

void App::trimStationToLowestSweepLocked(StationState& st) {
    if (st.precomputed.size() <= 1)
        return;

    int lowestIdx = findLowestSweepIndex(st.precomputed);
    if (lowestIdx < 0)
        return;

    std::vector<PrecomputedSweep> reduced;
    reduced.push_back(std::move(st.precomputed[lowestIdx]));
    st.precomputed.swap(reduced);
    st.full_volume_resident = false;
}

void App::trimStationWorkingSet(int focusIdx) {
    std::lock_guard<std::mutex> lock(m_stationMutex);
    for (int i = 0; i < (int)m_stations.size(); i++) {
        auto& st = m_stations[i];
        if (!st.full_volume_resident)
            continue;
        if (shouldKeepStationFullVolumeLocked(i, focusIdx))
            continue;
        trimStationToLowestSweepLocked(st);
    }
}

int App::stationLiveHistoryLimitLocked(int stationIdx) const {
    if (stationIdx < 0 || stationIdx >= (int)m_stations.size())
        return 0;

    int activeLimit = 20;
    int visibleLimit = 4;
    int backgroundLimit = 2;
    switch (m_effectivePerformanceProfile) {
        case PerformanceProfile::Performance:
            activeLimit = 8;
            visibleLimit = 2;
            backgroundLimit = 1;
            break;
        case PerformanceProfile::Balanced:
            activeLimit = 12;
            visibleLimit = 3;
            backgroundLimit = 1;
            break;
        case PerformanceProfile::Auto:
        case PerformanceProfile::Quality:
        default:
            break;
    }

    if (m_snapshotMode || m_historicMode)
        return 0;
    if (stationIdx == m_activeStationIdx)
        return activeLimit;
    if (stationLikelyVisible(stationIdx))
        return visibleLimit;
    return backgroundLimit;
}

void App::appendLiveHistoryLocked(StationState& st, const std::string& volumeKey,
                                  const std::vector<PrecomputedSweep>& sweeps,
                                  float stationLat, float stationLon) {
    if (volumeKey.empty() || sweeps.empty())
        return;

    LiveVolumeHistoryEntry entry;
    entry.volume_key = volumeKey;
    entry.label = formatVolumeKeyTimestamp(volumeKey);
    entry.sweeps = sweeps;
    entry.station_lat = stationLat;
    entry.station_lon = stationLon;

    auto insertPos = st.live_history.begin();
    while (insertPos != st.live_history.end() && insertPos->volume_key < volumeKey)
        ++insertPos;
    if (insertPos != st.live_history.end() && insertPos->volume_key == volumeKey)
        return;
    st.live_history.insert(insertPos, std::move(entry));
}

void App::trimLiveHistoryWorkingSet(int focusIdx) {
    (void)focusIdx;
    std::lock_guard<std::mutex> lock(m_stationMutex);
    for (int i = 0; i < (int)m_stations.size(); i++) {
        auto& st = m_stations[i];
        const int limit = stationLiveHistoryLimitLocked(i);
        while ((int)st.live_history.size() > limit)
            st.live_history.pop_front();
    }
}

bool App::ensureStationFullVolume(int stationIdx) {
    if (stationIdx < 0 || stationIdx >= (int)m_stations.size())
        return false;
    if (m_historicMode)
        return true;

    std::vector<uint8_t> rawCopy;
    std::string stationName;
    float fallbackLat = 0.0f;
    float fallbackLon = 0.0f;
    bool dealiasEnabled = m_dealias;
    {
        std::lock_guard<std::mutex> lock(m_stationMutex);
        auto& st = m_stations[stationIdx];
        if (st.full_volume_resident)
            return true;
        if (st.raw_volume_data.empty())
            return false;
        rawCopy = st.raw_volume_data;
        stationName = st.icao;
        fallbackLat = st.lat;
        fallbackLon = st.lon;
    }

    PipelineStageTimings timings = {};
    ParsedRadarData parsed = {};
    auto decodeStart = Clock::now();
    std::vector<uint8_t> decoded = Level2Parser::decodeArchiveBytes(rawCopy);
    timings.decode_ms = elapsedMs(decodeStart, Clock::now());
    if (decoded.empty())
        return false;

    auto gpuBuildStart = Clock::now();
    auto gpuWorkingSet = buildGpuWorkingSetFromDecoded(decoded, GpuWorkingSetMode::AllSweeps);
    timings.sweep_build_ms = elapsedMs(gpuBuildStart, Clock::now());

    std::vector<PrecomputedSweep> precomp;
    float detectionLat = fallbackLat;
    float detectionLon = fallbackLon;
    int totalSweeps = 0;

    if (gpuWorkingSet.success && !gpuWorkingSet.sweeps.empty()) {
        precomp = std::move(gpuWorkingSet.sweeps);
        totalSweeps = gpuWorkingSet.total_sweeps;
        timings.used_gpu_sweep_build = true;
    } else {
        auto parseStart = Clock::now();
        parsed = Level2Parser::parseDecodedMessages(decoded, stationName);
        timings.parse_ms = elapsedMs(parseStart, Clock::now());
        if (parsed.sweeps.empty())
            return false;

        detectionLat = parsed.station_lat != 0.0f ? parsed.station_lat : fallbackLat;
        detectionLon = parsed.station_lon != 0.0f ? parsed.station_lon : fallbackLon;
        auto buildStart = Clock::now();
        precomp = buildPrecomputedSweeps(parsed);
        timings.sweep_build_ms = elapsedMs(buildStart, Clock::now());
        totalSweeps = (int)parsed.sweeps.size();
    }

    auto preprocessStart = Clock::now();
    if (dealiasEnabled)
        dealiasPrecomputedSweeps(precomp);
    timings.preprocess_ms = elapsedMs(preprocessStart, Clock::now());

    auto detectStart = Clock::now();
    Detection detection = computeDetectionForSweeps(precomp, detectionLat, detectionLon, stationName);
    timings.detection_ms = elapsedMs(detectStart, Clock::now());
    {
        std::lock_guard<std::mutex> lock(m_stationMutex);
        auto& st = m_stations[stationIdx];
        if (st.raw_volume_data.empty())
            st.raw_volume_data = std::move(rawCopy);
        st.total_sweeps = totalSweeps > 0 ? totalSweeps : (int)precomp.size();
        st.lowest_sweep_elev = precomp.empty() ? 0.0f : precomp.front().elevation_angle;
        st.lowest_sweep_radials = precomp.empty() ? 0 : precomp.front().num_radials;
        st.data_lat = detectionLat;
        st.data_lon = detectionLon;
        st.precomputed = std::move(precomp);
        st.full_volume_resident = true;
        st.parsed = true;
        st.detection = std::move(detection);
        st.timings = timings;
        st.uploaded = false;
        st.uploaded_product = -1;
        st.uploaded_tilt = -1;
        st.uploaded_sweep = -1;
        st.uploaded_lowest_sweep = false;
        m_gridDirty = true;
    }

    trimStationWorkingSet(stationIdx);
    return true;
}

void App::ensureRenderTargets() {
    const int rw = renderWidth();
    const int rh = renderHeight();
    const bool sizeChanged =
        !m_d_compositeOutput ||
        m_memoryTelemetry.internal_render_width != rw ||
        m_memoryTelemetry.internal_render_height != rh;

    if (sizeChanged) {
        if (m_d_compositeOutput) {
            cudaFree(m_d_compositeOutput);
            m_d_compositeOutput = nullptr;
        }
        const size_t outSize = (size_t)rw * rh * sizeof(uint32_t);
        CUDA_CHECK(cudaMalloc(&m_d_compositeOutput, outSize));
        CUDA_CHECK(cudaMemset(m_d_compositeOutput, 0, outSize));
    }

    const bool textureReady = m_outputTex.textureId() != 0;
    const bool textureOk = textureReady ? m_outputTex.resize(rw, rh)
                                        : m_outputTex.init(rw, rh);
    if (!textureOk) {
        fprintf(stderr, "Failed to create output texture (%dx%d)\n", rw, rh);
    }

    m_memoryTelemetry.internal_render_width = rw;
    m_memoryTelemetry.internal_render_height = rh;
    m_memoryTelemetry.render_scale = m_renderScale;
}

void App::applyPerformanceProfile(bool force) {
    const PerformanceProfile resolved = (m_requestedPerformanceProfile == PerformanceProfile::Auto)
        ? recommendedPerformanceProfile()
        : m_requestedPerformanceProfile;
    const PerformanceProfile previous = m_effectivePerformanceProfile;
    const float previousScale = m_renderScale;

    VolumeQualitySettings volumeQuality = {};
    switch (resolved) {
        case PerformanceProfile::Balanced:
            m_renderScale = 1.0f;
            volumeQuality.smooth_passes = 1;
            volumeQuality.ray_step_km = 0.75f;
            volumeQuality.max_steps = 560;
            break;
        case PerformanceProfile::Performance:
            m_renderScale = 1.0f;
            volumeQuality.smooth_passes = 0;
            volumeQuality.ray_step_km = 1.05f;
            volumeQuality.max_steps = 360;
            break;
        case PerformanceProfile::Auto:
        case PerformanceProfile::Quality:
        default:
            m_renderScale = 1.0f;
            volumeQuality.smooth_passes = 2;
            volumeQuality.ray_step_km = 0.55f;
            volumeQuality.max_steps = 720;
            break;
    }

    m_effectivePerformanceProfile = resolved;
    gpu::setVolumeQuality(volumeQuality);

    if (!force &&
        previous == m_effectivePerformanceProfile &&
        fabsf(previousScale - m_renderScale) < 0.001f) {
        updateMemoryTelemetry(true);
        return;
    }

    ensureRenderTargets();
    ensureCrossSectionBuffer(renderWidth(), std::max(200, renderHeight() / 3));
    invalidateFrameCache(true);
    m_volumeBuilt = false;
    m_volumeStation = -1;
    m_needsComposite = true;
    m_needsRerender = true;

    if (m_mode3D || m_crossSection)
        rebuildVolumeForCurrentSelection();

    trimLiveHistoryWorkingSet(m_activeStationIdx);
    updateMemoryTelemetry(true);
}

void App::setPerformanceProfile(PerformanceProfile profile) {
    if (m_requestedPerformanceProfile == profile)
        return;
    m_requestedPerformanceProfile = profile;
    applyPerformanceProfile();
}

bool App::stationEnabled(int idx) const {
    std::lock_guard<std::mutex> lock(m_stationMutex);
    if (idx < 0 || idx >= (int)m_stations.size())
        return false;
    return m_stations[idx].enabled;
}

int App::enabledStationCount() const {
    std::lock_guard<std::mutex> lock(m_stationMutex);
    int count = 0;
    for (const auto& st : m_stations)
        count += st.enabled ? 1 : 0;
    return count;
}

void App::setStationEnabled(int idx, bool enabled) {
    if (idx < 0 || idx >= m_stationsTotal)
        return;

    bool changed = false;
    bool wasUploaded = false;
    bool wasDownloading = false;
    {
        std::lock_guard<std::mutex> lock(m_stationMutex);
        auto& st = m_stations[idx];
        if (st.enabled == enabled)
            return;
        changed = true;
        st.enabled = enabled;
        if (!enabled) {
            wasUploaded = st.uploaded;
            wasDownloading = st.downloading;
            st.downloading = false;
            st.parsed = false;
            st.uploaded = false;
            st.rendered = false;
            st.failed = false;
            st.error.clear();
            st.raw_volume_data.clear();
            st.gpuInfo = {};
            st.precomputed.clear();
            st.total_sweeps = 0;
            st.lowest_sweep_elev = 0.0f;
            st.lowest_sweep_radials = 0;
            st.data_lat = 0.0f;
            st.data_lon = 0.0f;
            st.full_volume_resident = false;
            st.lastUpdate = {};
            st.lastPollAttempt = {};
            st.latestVolumeKey.clear();
            st.detection = {};
            st.live_history.clear();
            st.uploaded_product = -1;
            st.uploaded_tilt = -1;
            st.uploaded_sweep = -1;
            st.uploaded_lowest_sweep = false;
        }
    }

    if (!changed)
        return;

    if (enabled) {
        queueLiveStationRefresh(idx, true);
    } else {
        gpu::freeStation(idx);
        if (wasUploaded && m_stationsLoaded.load() > 0)
            --m_stationsLoaded;
        if (wasDownloading && m_stationsDownloading.load() > 0)
            --m_stationsDownloading;
        m_gridDirty = true;
        if (m_activeStationIdx == idx) {
            invalidateLiveLoop(false);
            m_needsRerender = true;
        }
    }
}

void App::disableAllStations() {
    std::vector<int> enabledStations;
    {
        std::lock_guard<std::mutex> lock(m_stationMutex);
        enabledStations.reserve(m_stations.size());
        for (const auto& st : m_stations) {
            if (st.enabled)
                enabledStations.push_back(st.index);
        }
    }
    for (int idx : enabledStations)
        setStationEnabled(idx, false);
}

bool App::stationPinned(int idx) const {
    std::lock_guard<std::mutex> lock(m_priorityMutex);
    return m_pinnedStations.find(idx) != m_pinnedStations.end();
}

bool App::stationPriorityHot(int idx) const {
    std::lock_guard<std::mutex> lock(m_priorityMutex);
    if (m_pinnedStations.find(idx) != m_pinnedStations.end())
        return true;
    return std::find(m_dynamicPriorityStations.begin(),
                     m_dynamicPriorityStations.end(),
                     idx) != m_dynamicPriorityStations.end();
}

void App::setStationPinned(int idx, bool pinned) {
    if (idx < 0 || idx >= m_stationsTotal)
        return;

    {
        std::lock_guard<std::mutex> lock(m_priorityMutex);
        if (pinned)
            m_pinnedStations.insert(idx);
        else
            m_pinnedStations.erase(idx);

        const size_t hotCount = m_dynamicPriorityStations.size() + m_pinnedStations.size();
        m_priorityStatus = "Priority radars: " + std::to_string((int)hotCount) +
                           " hot, " + std::to_string((int)m_pinnedStations.size()) +
                           " pinned";
    }

    queueLiveStationRefresh(idx, true);
}

int App::pinnedStationCount() const {
    std::lock_guard<std::mutex> lock(m_priorityMutex);
    return (int)m_pinnedStations.size();
}

std::string App::priorityStatus() const {
    std::lock_guard<std::mutex> lock(m_priorityMutex);
    return m_priorityStatus;
}

float App::priorityScoreForStation(int stationIdx,
                                   const std::vector<ProbSevereObject>& objects) const {
    if (stationIdx < 0 || stationIdx >= m_stationsTotal)
        return 0.0f;

    const float radarLat = NEXRAD_STATIONS[stationIdx].lat;
    const float radarLon = NEXRAD_STATIONS[stationIdx].lon;
    float score = 0.0f;
    for (const auto& obj : objects) {
        const float cellScore =
            obj.prob_tor * 6.0f +
            obj.prob_severe * 2.0f +
            obj.prob_wind * 1.4f +
            obj.prob_hail * 1.2f;
        if (cellScore <= 0.0f)
            continue;

        const float distKm = markerDistanceKm(radarLat, radarLon, obj.lat, obj.lon);
        if (distKm > 325.0f)
            continue;

        score += cellScore / (1.0f + distKm / 90.0f);
    }
    return score;
}

std::vector<int> App::buildSpatialFallbackStations(int targetCount,
                                                   const std::vector<int>& seed) const {
    std::vector<int> selected = seed;
    selected.reserve(std::max(targetCount, (int)seed.size()));

    auto alreadySelected = [&selected](int idx) {
        return std::find(selected.begin(), selected.end(), idx) != selected.end();
    };

    if (selected.empty()) {
        int bestCenter = -1;
        float bestCenterDist = std::numeric_limits<float>::max();
        for (int i = 0; i < m_stationsTotal; i++) {
            const float d = markerDistanceKm(NEXRAD_STATIONS[i].lat, NEXRAD_STATIONS[i].lon,
                                             39.0f, -98.0f);
            if (d < bestCenterDist) {
                bestCenterDist = d;
                bestCenter = i;
            }
        }
        if (bestCenter >= 0)
            selected.push_back(bestCenter);
    }

    while ((int)selected.size() < targetCount) {
        int bestIdx = -1;
        float bestSpread = -1.0f;
        for (int i = 0; i < m_stationsTotal; i++) {
            if (alreadySelected(i))
                continue;

            float minDist = std::numeric_limits<float>::max();
            for (int chosen : selected) {
                const float d = markerDistanceKm(NEXRAD_STATIONS[i].lat, NEXRAD_STATIONS[i].lon,
                                                 NEXRAD_STATIONS[chosen].lat, NEXRAD_STATIONS[chosen].lon);
                minDist = std::min(minDist, d);
            }
            if (minDist > bestSpread) {
                bestSpread = minDist;
                bestIdx = i;
            }
        }
        if (bestIdx < 0)
            break;
        selected.push_back(bestIdx);
    }

    return selected;
}

std::vector<int> App::buildPrioritySeedStations(int targetCount) const {
    std::vector<int> seed;
    seed.reserve(targetCount);

    {
        std::lock_guard<std::mutex> lock(m_priorityMutex);
        if (m_activeStationIdx >= 0 && m_activeStationIdx < m_stationsTotal)
            seed.push_back(m_activeStationIdx);
        for (int idx : m_pinnedStations) {
            if (idx >= 0 && idx < m_stationsTotal &&
                std::find(seed.begin(), seed.end(), idx) == seed.end()) {
                seed.push_back(idx);
            }
        }
        for (int idx : m_dynamicPriorityStations) {
            if ((int)seed.size() >= targetCount)
                break;
            if (idx >= 0 && idx < m_stationsTotal &&
                std::find(seed.begin(), seed.end(), idx) == seed.end()) {
                seed.push_back(idx);
            }
        }
    }

    if ((int)seed.size() < targetCount)
        seed = buildSpatialFallbackStations(targetCount, seed);
    if ((int)seed.size() > targetCount)
        seed.resize(targetCount);
    return seed;
}

void App::updatePriorityStationsFromProbSevere(std::vector<ProbSevereObject> objects,
                                               std::string status) {
    std::vector<std::pair<float, int>> ranked;
    ranked.reserve(m_stationsTotal);
    for (int i = 0; i < m_stationsTotal; i++) {
        const float score = priorityScoreForStation(i, objects);
        if (score > 0.25f)
            ranked.emplace_back(score, i);
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<int> dynamic;
    dynamic.reserve(m_bootstrapStationTarget);
    for (const auto& entry : ranked) {
        if ((int)dynamic.size() >= m_bootstrapStationTarget)
            break;
        dynamic.push_back(entry.second);
    }
    if ((int)dynamic.size() < m_bootstrapStationTarget)
        dynamic = buildSpatialFallbackStations(m_bootstrapStationTarget, dynamic);

    {
        std::lock_guard<std::mutex> lock(m_priorityMutex);
        m_probSevereObjects = std::move(objects);
        m_dynamicPriorityStations = std::move(dynamic);
        m_priorityStatus = std::move(status);
        m_lastProbSevereRefresh = std::chrono::steady_clock::now();
    }

    const auto seed = buildPrioritySeedStations(m_bootstrapStationTarget);
    for (int idx : seed)
        queueLiveStationRefresh(idx, true);
}

void App::requestProbSevereRefresh(bool force) {
    if (!m_downloader)
        return;

    const auto now = std::chrono::steady_clock::now();
    if (!force && m_lastProbSevereRefresh.time_since_epoch().count() != 0) {
        const float elapsed = std::chrono::duration<float>(now - m_lastProbSevereRefresh).count();
        if (elapsed < 110.0f)
            return;
    }
    if (m_probSevereLoading.exchange(true))
        return;

    {
        std::lock_guard<std::mutex> lock(m_priorityMutex);
        if (m_priorityStatus.empty())
            m_priorityStatus = "Refreshing ProbSevere seed...";
    }

    m_downloader->queueDownload(
        "probsevere_index",
        kProbSevereHost,
        kProbSevereIndexPath,
        [this](const std::string& id, DownloadResult indexResult) {
            (void)id;

            auto finish = [this](std::vector<ProbSevereObject> objects, std::string status) {
                updatePriorityStationsFromProbSevere(std::move(objects), std::move(status));
                m_probSevereLoading = false;
            };

            if (!indexResult.success || indexResult.data.empty()) {
                finish({}, "ProbSevere unavailable; using spread startup seed.");
                return;
            }

            const std::string html(indexResult.data.begin(), indexResult.data.end());
            const std::string latestName = latestProbSevereJsonNameFromIndex(html);
            if (latestName.empty()) {
                finish({}, "ProbSevere index empty; using spread startup seed.");
                return;
            }

            auto jsonResult = Downloader::httpGet(
                kProbSevereHost,
                std::string(kProbSevereIndexPath) + latestName);
            if (!jsonResult.success || jsonResult.data.empty()) {
                finish({}, "ProbSevere fetch failed; using spread startup seed.");
                return;
            }

            const auto parsed = json::parse(jsonResult.data.begin(), jsonResult.data.end(), nullptr, false);
            if (parsed.is_discarded() || !parsed.is_object()) {
                finish({}, "ProbSevere parse failed; using spread startup seed.");
                return;
            }

            std::vector<ProbSevereObject> objects;
            auto featuresIt = parsed.find("features");
            if (featuresIt != parsed.end() && featuresIt->is_array()) {
                objects.reserve(featuresIt->size());
                for (const auto& feature : *featuresIt) {
                    const auto& props = feature.contains("properties") ? feature["properties"] : json();
                    ProbSevereObject obj;
                    obj.id = parseJsonStringField(props, "ID");
                    obj.lat = parseJsonFloatField(props, "MLAT");
                    obj.lon = parseJsonFloatField(props, "MLON");
                    obj.prob_severe = parseJsonFloatField(props, "ProbSevere");
                    obj.prob_tor = parseJsonFloatField(props, "ProbTor");
                    obj.prob_hail = parseJsonFloatField(props, "ProbHail");
                    obj.prob_wind = parseJsonFloatField(props, "ProbWind");
                    obj.motion_east = parseJsonFloatField(props, "MOTION_EAST");
                    obj.motion_south = parseJsonFloatField(props, "MOTION_SOUTH");
                    obj.avg_beam_height_km = parseJsonFloatField(props, "AVG_BEAM_HGT");
                    if (obj.lat == 0.0f && obj.lon == 0.0f)
                        continue;
                    objects.push_back(std::move(obj));
                }
            }

            std::string status;
            if (objects.empty()) {
                status = "ProbSevere quiet; using spread startup seed.";
            } else {
                status = "ProbSevere seeded " + std::to_string((int)objects.size()) +
                         " objects from " + latestName;
            }
            finish(std::move(objects), std::move(status));
        });
}

void App::invalidateLiveLoop(bool freeMemory) {
    m_liveLoopBackfillGeneration.fetch_add(1);
    m_liveLoopBackfillLoading = false;
    {
        std::lock_guard<std::mutex> lock(m_liveLoopBackfillMutex);
        m_liveLoopBackfillQueue.clear();
    }
    m_liveLoopBackfillReplaceExisting = false;
    resetLiveLoopFrameCache(freeMemory);
}

void App::resetLiveLoopFrameCache(bool freeMemory) {
    if (freeMemory) {
        for (auto& frame : m_liveLoopFrames) {
            if (frame) {
                cudaFree(frame);
                frame = nullptr;
            }
        }
        m_liveLoopFrameWidth = 0;
        m_liveLoopFrameHeight = 0;
    }

    for (auto& label : m_liveLoopLabels)
        label.clear();

    m_liveLoopPlaying = false;
    m_liveLoopCount = 0;
    m_liveLoopWriteIndex = 0;
    m_liveLoopPlaybackIndex = 0;
    m_liveLoopAccumulator = 0.0f;
    m_liveLoopCapturePending = false;
    m_memoryTelemetry.live_loop_bytes = 0;
}

void App::requestLiveLoopCapture() {
    if (!m_liveLoopEnabled || m_historicMode || m_snapshotMode || m_mode3D || m_crossSection)
        return;
    m_liveLoopCapturePending = true;
}

void App::requestLiveLoopBackfill() {
    if (!m_liveLoopEnabled || m_historicMode || m_snapshotMode || m_mode3D || m_crossSection) {
        requestLiveLoopCapture();
        return;
    }
    if (m_showAll || m_activeStationIdx < 0 || m_activeStationIdx >= (int)m_stations.size()) {
        requestLiveLoopCapture();
        return;
    }

    const int desiredFrames = std::max(1, std::min(m_liveLoopLength, MAX_LIVE_LOOP_FRAMES));
    const int requestStationIdx = m_activeStationIdx;
    std::string stationCode;
    std::string currentKey;
    float fallbackLat = 0.0f;
    float fallbackLon = 0.0f;
    std::vector<LiveLoopBackfillFrame> localFrames;
    const bool dealiasEnabled = m_dealias;
    {
        std::lock_guard<std::mutex> lock(m_stationMutex);
        const auto& st = m_stations[requestStationIdx];
        stationCode = st.icao;
        currentKey = st.latestVolumeKey;
        fallbackLat = (st.data_lat != 0.0f) ? st.data_lat : st.lat;
        fallbackLon = (st.data_lon != 0.0f) ? st.data_lon : st.lon;
        const int historyStart = std::max(0, (int)st.live_history.size() - desiredFrames);
        localFrames.reserve((int)st.live_history.size() - historyStart);
        for (int i = historyStart; i < (int)st.live_history.size(); i++) {
            const auto& entry = st.live_history[i];
            if (entry.sweeps.empty())
                continue;
            LiveLoopBackfillFrame frame;
            frame.label = entry.label.empty() ? formatVolumeKeyTimestamp(entry.volume_key) : entry.label;
            frame.sweeps = entry.sweeps;
            frame.station_lat = entry.station_lat != 0.0f ? entry.station_lat : fallbackLat;
            frame.station_lon = entry.station_lon != 0.0f ? entry.station_lon : fallbackLon;
            localFrames.push_back(std::move(frame));
        }
    }
    if (stationCode.empty()) {
        requestLiveLoopCapture();
        return;
    }

    const uint64_t generation = m_liveLoopBackfillGeneration.fetch_add(1) + 1;
    {
        std::lock_guard<std::mutex> lock(m_liveLoopBackfillMutex);
        m_liveLoopBackfillQueue.clear();
        for (auto& frame : localFrames)
            m_liveLoopBackfillQueue.push_back(std::move(frame));
        m_liveLoopBackfillReplaceExisting = !m_liveLoopBackfillQueue.empty();
    }
    if (localFrames.empty())
        resetLiveLoopFrameCache(false);

    if ((int)localFrames.size() >= desiredFrames || currentKey.empty()) {
        m_liveLoopBackfillLoading = false;
        if (localFrames.empty())
            requestLiveLoopCapture();
        return;
    }

    m_liveLoopBackfillLoading = true;

    int year = 0, month = 0, day = 0;
    if (!extractVolumeKeyDate(currentKey, year, month, day))
        getUtcDate(year, month, day);

    const std::string listQuery = buildLiveListQuery(stationCode, year, month, day, {});
    m_downloader->queueDownload(
        stationCode + "_live_loop_backfill",
        NEXRAD_HOST,
        listQuery,
        [this, generation, stationCode, currentKey, desiredFrames, dealiasEnabled,
         fallbackLat, fallbackLon, year, month, day, requestStationIdx](const std::string& id, DownloadResult listResult) {
            (void)id;
            if (generation != m_liveLoopBackfillGeneration.load())
                return;

            auto fetchList = [&](int y, int m, int d, std::vector<NexradFile>& outFiles) -> bool {
                DownloadResult result = Downloader::httpGet(NEXRAD_HOST, buildLiveListQuery(stationCode, y, m, d, {}));
                if (!result.success || result.data.empty())
                    return false;
                std::string xml(result.data.begin(), result.data.end());
                outFiles = parseS3ListResponse(xml);
                return !outFiles.empty();
            };

            std::vector<NexradFile> files;
            if (listResult.success && !listResult.data.empty()) {
                std::string xml(listResult.data.begin(), listResult.data.end());
                files = parseS3ListResponse(xml);
            }
            if (files.empty() && !fetchList(year, month, day, files)) {
                m_liveLoopBackfillLoading = false;
                return;
            }

            int endIndex = (int)files.size() - 1;
            if (!currentKey.empty()) {
                for (int i = (int)files.size() - 1; i >= 0; --i) {
                    if (files[i].key <= currentKey) {
                        endIndex = i;
                        break;
                    }
                }
            }
            if (endIndex < 0) {
                m_liveLoopBackfillLoading = false;
                return;
            }

            std::vector<std::string> selectedKeys;
            int need = desiredFrames;
            const int startIndex = std::max(0, endIndex - need + 1);
            for (int i = startIndex; i <= endIndex; i++)
                selectedKeys.push_back(files[i].key);
            need -= (int)selectedKeys.size();

            if (need > 0) {
                int prevYear = year;
                int prevMonth = month;
                int prevDay = day;
                shiftDate(prevYear, prevMonth, prevDay, -1);
                std::vector<NexradFile> previousFiles;
                if (fetchList(prevYear, prevMonth, prevDay, previousFiles) && !previousFiles.empty()) {
                    const int take = std::min(need, (int)previousFiles.size());
                    std::vector<std::string> priorKeys;
                    priorKeys.reserve(take);
                    for (int i = (int)previousFiles.size() - take; i < (int)previousFiles.size(); i++)
                        priorKeys.push_back(previousFiles[i].key);
                    selectedKeys.insert(selectedKeys.begin(), priorKeys.begin(), priorKeys.end());
                }
            }

            struct OrderedLoopFrame {
                std::string key;
                LiveLoopBackfillFrame frame;
            };
            std::vector<OrderedLoopFrame> orderedFrames;
            orderedFrames.reserve(desiredFrames);
            {
                std::lock_guard<std::mutex> stationLock(m_stationMutex);
                if (requestStationIdx >= 0 && requestStationIdx < (int)m_stations.size()) {
                    const auto& st = m_stations[requestStationIdx];
                    const int historyStart = std::max(0, (int)st.live_history.size() - desiredFrames);
                    for (int i = historyStart; i < (int)st.live_history.size(); i++) {
                        const auto& entry = st.live_history[i];
                        if (entry.sweeps.empty())
                            continue;
                        OrderedLoopFrame ordered;
                        ordered.key = entry.volume_key;
                        ordered.frame.label = entry.label.empty()
                            ? formatVolumeKeyTimestamp(entry.volume_key)
                            : entry.label;
                        ordered.frame.sweeps = entry.sweeps;
                        ordered.frame.station_lat =
                            entry.station_lat != 0.0f ? entry.station_lat : fallbackLat;
                        ordered.frame.station_lon =
                            entry.station_lon != 0.0f ? entry.station_lon : fallbackLon;
                        orderedFrames.push_back(std::move(ordered));
                    }
                }
            }

            for (const auto& key : selectedKeys) {
                if (generation != m_liveLoopBackfillGeneration.load()) {
                    m_liveLoopBackfillLoading = false;
                    return;
                }

                bool alreadyPresent = false;
                for (const auto& existing : orderedFrames) {
                    if (existing.key == key) {
                        alreadyPresent = true;
                        break;
                    }
                }
                if (alreadyPresent)
                    continue;

                DownloadResult fileResult = Downloader::httpGet(NEXRAD_HOST, "/" + key);
                if (!fileResult.success || fileResult.data.empty())
                    continue;

                auto parsed = Level2Parser::parse(fileResult.data);
                if (parsed.sweeps.empty())
                    continue;

                LiveLoopBackfillFrame frame;
                frame.label = formatVolumeKeyTimestamp(key);
                frame.sweeps = buildPrecomputedSweeps(parsed);
                if (dealiasEnabled)
                    dealiasPrecomputedSweeps(frame.sweeps);
                frame.station_lat = (parsed.station_lat != 0.0f) ? parsed.station_lat : fallbackLat;
                frame.station_lon = (parsed.station_lon != 0.0f) ? parsed.station_lon : fallbackLon;

                if (frame.sweeps.empty())
                    continue;

                OrderedLoopFrame ordered;
                ordered.key = key;
                ordered.frame = std::move(frame);
                orderedFrames.push_back(std::move(ordered));
            }

            std::sort(orderedFrames.begin(), orderedFrames.end(),
                      [](const OrderedLoopFrame& a, const OrderedLoopFrame& b) {
                          return a.key < b.key;
                      });
            orderedFrames.erase(
                std::unique(orderedFrames.begin(), orderedFrames.end(),
                            [](const OrderedLoopFrame& a, const OrderedLoopFrame& b) {
                                return a.key == b.key;
                            }),
                orderedFrames.end());
            if ((int)orderedFrames.size() > desiredFrames) {
                orderedFrames.erase(orderedFrames.begin(),
                                    orderedFrames.end() - desiredFrames);
            }

            {
                std::lock_guard<std::mutex> stationLock(m_stationMutex);
                if (requestStationIdx >= 0 && requestStationIdx < (int)m_stations.size()) {
                    auto& st = m_stations[requestStationIdx];
                    if (st.enabled) {
                        for (const auto& ordered : orderedFrames) {
                            appendLiveHistoryLocked(st,
                                                    ordered.key,
                                                    ordered.frame.sweeps,
                                                    ordered.frame.station_lat,
                                                    ordered.frame.station_lon);
                        }
                    }
                }
            }
            trimLiveHistoryWorkingSet(requestStationIdx);

            if (generation == m_liveLoopBackfillGeneration.load()) {
                std::lock_guard<std::mutex> lock(m_liveLoopBackfillMutex);
                m_liveLoopBackfillQueue.clear();
                for (auto& ordered : orderedFrames)
                    m_liveLoopBackfillQueue.push_back(std::move(ordered.frame));
                m_liveLoopBackfillReplaceExisting = !m_liveLoopBackfillQueue.empty();
                m_liveLoopBackfillLoading = false;
            }
        });
}

int App::liveLoopSlotForIndex(int index) const {
    if (m_liveLoopCount <= 0)
        return 0;
    index = std::max(0, std::min(index, m_liveLoopCount - 1));
    const int oldestSlot =
        (m_liveLoopWriteIndex - m_liveLoopCount + MAX_LIVE_LOOP_FRAMES) % MAX_LIVE_LOOP_FRAMES;
    return (oldestSlot + index) % MAX_LIVE_LOOP_FRAMES;
}

std::string App::currentLiveLoopCaptureLabel() const {
    if (m_showAll)
        return "Live Mosaic";

    std::lock_guard<std::mutex> lock(m_stationMutex);
    if (m_activeStationIdx >= 0 && m_activeStationIdx < (int)m_stations.size()) {
        const auto& st = m_stations[m_activeStationIdx];
        std::string label = st.icao.empty() ? std::string("Live") : st.icao;
        const std::string ts = formatVolumeKeyTimestamp(st.latestVolumeKey);
        if (!ts.empty())
            label += " | " + ts;
        return label;
    }
    return "Live";
}

void App::captureLiveLoopFrame(const uint32_t* d_src, int w, int h, const std::string& label) {
    if (!m_liveLoopEnabled || m_liveLoopLength <= 0 || !d_src || w <= 0 || h <= 0)
        return;

    if ((m_liveLoopFrameWidth > 0 || m_liveLoopFrameHeight > 0) &&
        (m_liveLoopFrameWidth != w || m_liveLoopFrameHeight != h)) {
        invalidateLiveLoop(true);
    }

    m_liveLoopFrameWidth = w;
    m_liveLoopFrameHeight = h;

    const int slot = m_liveLoopWriteIndex;
    const size_t sz = (size_t)w * h * sizeof(uint32_t);
    if (!m_liveLoopFrames[slot])
        CUDA_CHECK(cudaMalloc(&m_liveLoopFrames[slot], sz));
    CUDA_CHECK(cudaMemcpy(m_liveLoopFrames[slot], d_src, sz, cudaMemcpyDeviceToDevice));
    m_liveLoopLabels[slot] = label;

    if (m_liveLoopCount < m_liveLoopLength)
        m_liveLoopCount++;

    m_liveLoopWriteIndex = (m_liveLoopWriteIndex + 1) % MAX_LIVE_LOOP_FRAMES;
    m_liveLoopPlaybackIndex = std::max(0, m_liveLoopCount - 1);
    m_liveLoopCapturePending = false;
}

void App::updateLiveLoop(float dt) {
    if (!m_liveLoopEnabled || !m_liveLoopPlaying || m_liveLoopCount <= 1)
        return;

    const float fps = std::max(1.0f, m_liveLoopSpeed);
    m_liveLoopAccumulator += dt;
    const float frameInterval = 1.0f / fps;
    while (m_liveLoopAccumulator >= frameInterval) {
        m_liveLoopAccumulator -= frameInterval;
        m_liveLoopPlaybackIndex = (m_liveLoopPlaybackIndex + 1) % m_liveLoopCount;
    }
}

void App::processLiveLoopBackfill() {
    if (!m_liveLoopEnabled || m_historicMode || m_snapshotMode || m_mode3D || m_crossSection)
        return;

    LiveLoopBackfillFrame frame;
    bool haveFrame = false;
    bool replaceExisting = false;
    {
        std::lock_guard<std::mutex> lock(m_liveLoopBackfillMutex);
        if (!m_liveLoopBackfillQueue.empty() && m_liveLoopBackfillReplaceExisting) {
            replaceExisting = true;
            m_liveLoopBackfillReplaceExisting = false;
        }
        if (!m_liveLoopBackfillQueue.empty()) {
            frame = std::move(m_liveLoopBackfillQueue.front());
            m_liveLoopBackfillQueue.pop_front();
            haveFrame = true;
        }
    }

    if (!haveFrame) {
        if (!m_liveLoopBackfillLoading && m_liveLoopCount == 0)
            requestLiveLoopCapture();
        return;
    }

    if (replaceExisting)
        resetLiveLoopFrameCache(false);

    const int productTilts = countProductSweeps(frame.sweeps, m_activeProduct);
    if (productTilts <= 0)
        return;

    const int sweepIdx = findProductSweep(frame.sweeps, m_activeProduct,
                                          std::min(m_activeTilt, productTilts - 1));
    if (sweepIdx < 0 || sweepIdx >= (int)frame.sweeps.size())
        return;

    const auto& pc = frame.sweeps[sweepIdx];
    if (pc.num_radials <= 0)
        return;

    constexpr int kLiveLoopBackfillSlot = MAX_STATIONS - 1;
    GpuStationInfo info = {};
    info.lat = frame.station_lat;
    info.lon = frame.station_lon;
    info.elevation_angle = pc.elevation_angle;
    info.num_radials = pc.num_radials;

    const uint16_t* gatePtrs[NUM_PRODUCTS] = {};
    for (int p = 0; p < NUM_PRODUCTS; p++) {
        const auto& pd = pc.products[p];
        if (!pd.has_data)
            continue;
        info.has_product[p] = true;
        info.num_gates[p] = pd.num_gates;
        info.first_gate_km[p] = pd.first_gate_km;
        info.gate_spacing_km[p] = pd.gate_spacing_km;
        info.scale[p] = pd.scale;
        info.offset[p] = pd.offset;
        if (!pd.gates.empty())
            gatePtrs[p] = pd.gates.data();
    }

    gpu::allocateStation(kLiveLoopBackfillSlot, info);
    gpu::uploadStationData(kLiveLoopBackfillSlot, info, pc.azimuths.data(), gatePtrs);
    gpu::syncStation(kLiveLoopBackfillSlot);

    const int rw = renderWidth();
    const int rh = renderHeight();
    GpuViewport gpuVp;
    gpuVp.center_lat = (float)m_viewport.center_lat;
    gpuVp.center_lon = (float)m_viewport.center_lon;
    gpuVp.deg_per_pixel_x =
        (1.0f / (float)m_viewport.zoom) * ((float)m_viewport.width / (float)rw);
    gpuVp.deg_per_pixel_y =
        (1.0f / (float)m_viewport.zoom) * ((float)m_viewport.height / (float)rh);
    gpuVp.width = rw;
    gpuVp.height = rh;

    const float srvSpd = (m_srvMode && m_activeProduct == PROD_VEL) ? m_stormSpeed : 0.0f;
    const float activeThreshold = (m_activeProduct == PROD_VEL)
        ? m_velocityMinThreshold
        : m_dbzMinThreshold;
    gpu::forwardRenderStation(gpuVp, kLiveLoopBackfillSlot,
                              m_activeProduct, activeThreshold,
                              m_d_compositeOutput, srvSpd, m_stormDir);
    captureLiveLoopFrame(m_d_compositeOutput, gpuVp.width, gpuVp.height, frame.label);
    gpu::freeStation(kLiveLoopBackfillSlot);
}

void App::setLiveLoopEnabled(bool enabled) {
    if (m_liveLoopEnabled == enabled)
        return;

    m_liveLoopEnabled = enabled;
    if (!enabled) {
        invalidateLiveLoop(true);
        return;
    }

    requestLiveLoopBackfill();
}

void App::toggleLiveLoopPlayback() {
    if (!m_liveLoopEnabled)
        setLiveLoopEnabled(true);
    if (m_liveLoopCount <= 1) {
        m_liveLoopPlaying = false;
        requestLiveLoopCapture();
        return;
    }

    m_liveLoopPlaying = !m_liveLoopPlaying;
    m_liveLoopAccumulator = 0.0f;
}

void App::setLiveLoopLength(int frames) {
    frames = std::max(1, std::min(frames, MAX_LIVE_LOOP_FRAMES));
    if (m_liveLoopLength == frames)
        return;

    m_liveLoopLength = frames;
    if (m_liveLoopCount > m_liveLoopLength)
        m_liveLoopCount = m_liveLoopLength;
    if (m_liveLoopCount <= 0) {
        m_liveLoopPlaybackIndex = 0;
    } else if (m_liveLoopPlaybackIndex >= m_liveLoopCount) {
        m_liveLoopPlaybackIndex = m_liveLoopCount - 1;
    }

    if (m_liveLoopEnabled)
        requestLiveLoopBackfill();
}

void App::setLiveLoopPlaybackFrame(int index) {
    if (m_liveLoopCount <= 0) {
        m_liveLoopPlaybackIndex = 0;
        m_liveLoopPlaying = false;
        return;
    }

    m_liveLoopPlaybackIndex = std::max(0, std::min(index, m_liveLoopCount - 1));
    m_liveLoopPlaying = false;
    m_liveLoopAccumulator = 0.0f;
}

void App::setLiveLoopSpeed(float fps) {
    m_liveLoopSpeed = std::max(1.0f, std::min(fps, 30.0f));
}

bool App::liveLoopViewingHistory() const {
    return m_liveLoopEnabled &&
           m_liveLoopCount > 0 &&
           (m_liveLoopPlaying || m_liveLoopPlaybackIndex < (m_liveLoopCount - 1));
}

std::string App::liveLoopCurrentLabel() const {
    if (m_liveLoopCount <= 0)
        return {};
    const int slot = liveLoopSlotForIndex(m_liveLoopPlaybackIndex);
    return m_liveLoopLabels[slot];
}

void App::clearLiveLoop() {
    invalidateLiveLoop(true);
}

void App::updateMemoryTelemetry(bool force) {
    const auto now = std::chrono::steady_clock::now();
    if (!force && m_lastMemorySample.time_since_epoch().count() != 0) {
        const float elapsed = std::chrono::duration<float>(now - m_lastMemorySample).count();
        if (elapsed < 0.25f)
            return;
    }
    m_lastMemorySample = now;

    size_t freeBytes = 0;
    size_t totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess) {
        m_memoryTelemetry.gpu_total_bytes = totalBytes;
        m_memoryTelemetry.gpu_free_bytes = freeBytes;
        m_memoryTelemetry.gpu_used_bytes = totalBytes >= freeBytes ? (totalBytes - freeBytes) : 0;
        m_memoryTelemetry.gpu_peak_used_bytes =
            std::max(m_memoryTelemetry.gpu_peak_used_bytes, m_memoryTelemetry.gpu_used_bytes);
    }

    m_memoryTelemetry.process_working_set_bytes = queryProcessWorkingSetBytes();
    m_memoryTelemetry.process_peak_working_set_bytes =
        std::max(m_memoryTelemetry.process_peak_working_set_bytes,
                 m_memoryTelemetry.process_working_set_bytes);

    size_t cachedBytes = 0;
    if (m_cachedFrameWidth > 0 && m_cachedFrameHeight > 0) {
        const size_t frameBytes = (size_t)m_cachedFrameWidth * m_cachedFrameHeight * sizeof(uint32_t);
        for (const auto* frame : m_cachedFrames) {
            if (frame)
                cachedBytes += frameBytes;
        }
    }
    m_memoryTelemetry.historic_cache_bytes = cachedBytes;

    size_t liveLoopBytes = 0;
    if (m_liveLoopFrameWidth > 0 && m_liveLoopFrameHeight > 0) {
        const size_t frameBytes =
            (size_t)m_liveLoopFrameWidth * m_liveLoopFrameHeight * sizeof(uint32_t);
        for (const auto* frame : m_liveLoopFrames) {
            if (frame)
                liveLoopBytes += frameBytes;
        }
    }
    m_memoryTelemetry.live_loop_bytes = liveLoopBytes;
    m_memoryTelemetry.volume_working_set_bytes = gpu::volumeWorkingSetBytes();
    m_memoryTelemetry.internal_render_width = renderWidth();
    m_memoryTelemetry.internal_render_height = renderHeight();
    m_memoryTelemetry.render_scale = m_renderScale;
}

void App::resetMemoryPeaks() {
    updateMemoryTelemetry(true);
    m_memoryTelemetry.gpu_peak_used_bytes = m_memoryTelemetry.gpu_used_bytes;
    m_memoryTelemetry.process_peak_working_set_bytes = m_memoryTelemetry.process_working_set_bytes;
}

bool App::isCurrentDownloadGeneration(uint64_t generation) const {
    return generation == m_downloadGeneration.load();
}

void App::failDownload(int stationIdx, uint64_t generation, std::string error) {
    std::lock_guard<std::mutex> lock(m_stationMutex);
    if (!isCurrentDownloadGeneration(generation)) return;
    if (stationIdx < 0 || stationIdx >= (int)m_stations.size()) return;

    auto& st = m_stations[stationIdx];
    if (!st.enabled && !m_snapshotMode && !m_historicMode) {
        if (st.downloading) {
            st.downloading = false;
            if (m_stationsDownloading.load() > 0)
                m_stationsDownloading--;
        }
        return;
    }
    st.failed = true;
    st.error = std::move(error);
    if (st.downloading) {
        st.downloading = false;
        m_stationsDownloading--;
    }
}

void App::finishLivePollNoChange(int stationIdx, uint64_t generation) {
    std::lock_guard<std::mutex> lock(m_stationMutex);
    if (!isCurrentDownloadGeneration(generation)) return;
    if (stationIdx < 0 || stationIdx >= (int)m_stations.size()) return;

    auto& st = m_stations[stationIdx];
    if (!st.enabled && !m_snapshotMode && !m_historicMode) {
        if (st.downloading) {
            st.downloading = false;
            if (m_stationsDownloading.load() > 0)
                m_stationsDownloading--;
        }
        return;
    }
    st.failed = false;
    st.error.clear();
    if (st.downloading) {
        st.downloading = false;
        m_stationsDownloading--;
    }
}

bool App::stationLikelyVisible(int stationIdx) const {
    if (stationIdx < 0 || stationIdx >= m_stationsTotal || m_viewport.zoom <= 0.0)
        return false;
    if (m_viewport.zoom < 80.0)
        return false;

    const double halfLat = (double)m_viewport.height * 0.5 / m_viewport.zoom;
    const double halfLon = (double)m_viewport.width * 0.5 / m_viewport.zoom;
    const double marginLat = std::max(1.5, halfLat * 0.1);
    const double marginLon = std::max(1.5, halfLon * 0.1);
    const double lat = NEXRAD_STATIONS[stationIdx].lat;
    const double lon = NEXRAD_STATIONS[stationIdx].lon;
    return lat >= m_viewport.center_lat - halfLat - marginLat &&
           lat <= m_viewport.center_lat + halfLat + marginLat &&
           lon >= m_viewport.center_lon - halfLon - marginLon &&
           lon <= m_viewport.center_lon + halfLon + marginLon;
}

float App::livePollIntervalSecForStation(int stationIdx, const StationState& st) const {
    float base = st.parsed ? m_backgroundStationPollIntervalSec : m_coldStationPollIntervalSec;
    const bool active = stationIdx == m_activeStationIdx;
    const bool priority = stationPriorityHot(stationIdx);
    const bool visible = stationLikelyVisible(stationIdx);

    if (active) {
        base = m_activeStationPollIntervalSec;
    } else if (priority) {
        base = std::min(base, m_priorityStationPollIntervalSec);
    } else if (visible) {
        base = m_visibleStationPollIntervalSec;
    }

    if (!st.parsed || st.failed) {
        if (active || priority || visible)
            base = std::min(base, m_recoveryStationPollIntervalSec);
        else
            base = std::max(base, m_coldStationPollIntervalSec);
    }

    if (!m_showAll && !active && visible)
        base = std::min(base, 20.0f);

    return base * stationPollJitter(stationIdx);
}

void App::queueLiveStationRefresh(int stationIdx, bool force) {
    if (stationIdx < 0 || stationIdx >= m_stationsTotal || m_snapshotMode || m_historicMode)
        return;

    const uint64_t generation = m_downloadGeneration.load();
    const bool dealiasEnabled = m_dealias;
    const auto now = std::chrono::steady_clock::now();
    std::string station;
    std::string currentKey;

    {
        std::lock_guard<std::mutex> lock(m_stationMutex);
        if (!isCurrentDownloadGeneration(generation)) return;
        auto& st = m_stations[stationIdx];
        if (!st.enabled) return;
        if (st.downloading) return;

        if (!force) {
            const float elapsed = st.lastPollAttempt.time_since_epoch().count() == 0
                ? std::numeric_limits<float>::max()
                : std::chrono::duration<float>(now - st.lastPollAttempt).count();
            if (elapsed < livePollIntervalSecForStation(stationIdx, st))
                return;
        }

        st.downloading = true;
        st.failed = false;
        st.error.clear();
        st.lastPollAttempt = now;
        station = st.icao;
        currentKey = st.latestVolumeKey;
    }
    m_stationsDownloading++;

    int year, month, day;
    getUtcDate(year, month, day);
    const std::string listQuery = buildLiveListQuery(station, year, month, day, currentKey);

    m_downloader->queueDownload(
        station + "_live_list",
        NEXRAD_HOST,
        listQuery,
        [this, stationIdx, station, generation, dealiasEnabled, currentKey](const std::string& id, DownloadResult listResult) {
            if (!isCurrentDownloadGeneration(generation)) return;
            if (!listResult.success || listResult.data.empty()) {
                if (!currentKey.empty()) {
                    finishLivePollNoChange(stationIdx, generation);
                    return;
                }

                int y, m, d;
                getUtcDate(y, m, d);
                shiftDate(y, m, d, -1);

                std::string path2 = buildLiveListQuery(station, y, m, d, {});

                auto retry = Downloader::httpGet(NEXRAD_HOST, path2);
                if (!retry.success || retry.data.empty()) {
                    failDownload(stationIdx, generation, "No data available");
                    return;
                }
                listResult = std::move(retry);
            }

            if (!isCurrentDownloadGeneration(generation)) return;

            std::string xml(listResult.data.begin(), listResult.data.end());
            auto files = parseS3ListResponse(xml);
            if (files.empty()) {
                if (!currentKey.empty()) {
                    finishLivePollNoChange(stationIdx, generation);
                    return;
                }
                failDownload(stationIdx, generation, "No files found");
                return;
            }

            std::vector<std::string> candidates;
            const std::string& latestKey = files.back().key;
            if (currentKey.empty() || latestKey > currentKey)
                candidates.push_back(latestKey);

            if (files.size() >= 2) {
                const std::string& fallbackKey = files[files.size() - 2].key;
                if ((currentKey.empty() || fallbackKey > currentKey) &&
                    fallbackKey != latestKey) {
                    candidates.push_back(fallbackKey);
                }
            }

            if (candidates.empty()) {
                finishLivePollNoChange(stationIdx, generation);
                return;
            }

            std::string lastError = "Latest scan unavailable";
            for (const auto& fileKey : candidates) {
                if (!isCurrentDownloadGeneration(generation)) return;
                auto fileResult = Downloader::httpGet(NEXRAD_HOST, "/" + fileKey);
                if (!fileResult.success || fileResult.data.empty()) {
                    if (!fileResult.error.empty())
                        lastError = fileResult.error;
                    continue;
                }
                if (tryProcessDownload(stationIdx, std::move(fileResult.data), generation,
                                       false, false, dealiasEnabled, fileKey)) {
                    return;
                }
                lastError = "Parse failed: no sweeps";
            }

            if (!currentKey.empty()) {
                finishLivePollNoChange(stationIdx, generation);
                return;
            }
            failDownload(stationIdx, generation, lastError);
        }
    );
}

void App::startDownloads() {
    int year, month, day;
    getUtcDate(year, month, day);
    std::vector<int> enabledStations;
    {
        std::lock_guard<std::mutex> lock(m_stationMutex);
        enabledStations.reserve(m_stations.size());
        for (const auto& st : m_stations) {
            if (st.enabled)
                enabledStations.push_back(st.index);
        }
    }
    printf("Fetching latest data for %04d-%02d-%02d from %d enabled stations...\n",
           year, month, day, (int)enabledStations.size());
    for (int idx : enabledStations)
        queueLiveStationRefresh(idx, true);
}

void App::resetStationsForReload() {
    if (m_downloader) m_downloader->shutdown();
    m_downloader = std::make_unique<Downloader>(48);

    {
        std::lock_guard<std::mutex> lock(m_uploadMutex);
        m_uploadQueue.clear();
    }

    for (int i = 0; i < (int)m_stations.size(); i++) {
        gpu::freeStation(i);
        auto& st = m_stations[i];
        st.downloading = false;
        st.parsed = false;
        st.uploaded = false;
        st.rendered = false;
        st.failed = false;
        st.error.clear();
        st.raw_volume_data.clear();
        st.gpuInfo = {};
        st.precomputed.clear();
        st.total_sweeps = 0;
        st.lowest_sweep_elev = 0.0f;
        st.lowest_sweep_radials = 0;
        st.data_lat = 0.0f;
        st.data_lon = 0.0f;
        st.full_volume_resident = false;
        st.lastUpdate = {};
        st.lastPollAttempt = {};
        st.latestVolumeKey.clear();
        st.detection = {};
        st.live_history.clear();
        st.uploaded_product = -1;
        st.uploaded_tilt = -1;
        st.uploaded_sweep = -1;
        st.uploaded_lowest_sweep = false;
    }

    m_stationsLoaded = 0;
    m_stationsDownloading = 0;
    m_gridDirty = true;
    m_activeStationIdx = -1;
    m_allTiltsCached = false;
    m_activeTilt = 0;
    m_maxTilts = 1;
    m_activeTiltAngle = 0.5f;
    m_volumeBuilt = false;
    m_volumeStation = -1;
    m_lastHistoricFrame = -1;
    m_needsRerender = true;
    m_needsComposite = true;
}

void App::startDownloadsForTimestamp(int year, int month, int day, int hour, int minute) {
    const uint64_t generation = m_downloadGeneration.load();
    const bool snapshotMode = m_snapshotMode;
    const bool lowestSweepOnly = m_snapshotLowestSweepOnly;
    const bool dealiasEnabled = m_dealias;
    const int targetSeconds = hour * 3600 + minute * 60;
    if (snapshotMode) {
        m_snapshotTimestampIso = makeIsoUtcTimestamp(year, month, day, hour, minute);
        m_warnings.requestHistoricSnapshot(m_snapshotTimestampIso);
    } else {
        m_snapshotTimestampIso.clear();
    }
    printf("Fetching archive snapshot for %04d-%02d-%02d %02d:%02d UTC from %d stations...\n",
           year, month, day, hour, minute, m_stationsTotal);

    for (int i = 0; i < m_stationsTotal; i++) {
        {
            std::lock_guard<std::mutex> lock(m_stationMutex);
            auto& st = m_stations[i];
            if (st.downloading) continue;
            st.downloading = true;
            st.failed = false;
            st.error.clear();
        }
        m_stationsDownloading++;

        std::string station = m_stations[i].icao;
        int idx = i;
        std::string listPath = buildListUrl(station, year, month, day);

        m_downloader->queueDownload(
            station + "_archive_list",
            NEXRAD_HOST,
            "/?list-type=2&prefix=" + std::string(listPath.data() + 1) + "&max-keys=1000",
            [this, idx, station, targetSeconds, generation, snapshotMode, lowestSweepOnly, dealiasEnabled](const std::string& id, DownloadResult listResult) {
                if (!isCurrentDownloadGeneration(generation)) return;
                if (!listResult.success || listResult.data.empty()) {
                    failDownload(idx, generation, "Archive listing failed");
                    return;
                }

                std::string xml(listResult.data.begin(), listResult.data.end());
                auto files = parseS3ListResponse(xml);
                if (files.empty()) {
                    failDownload(idx, generation, "No archive files found");
                    return;
                }

                int bestIdx = -1;
                int bestDelta = std::numeric_limits<int>::max();
                for (int fi = 0; fi < (int)files.size(); fi++) {
                    int hh = 0, mm = 0, ss = 0;
                    if (!extractArchiveTime(filenameFromKey(files[fi].key), hh, mm, ss))
                        continue;
                    int delta = abs((hh * 3600 + mm * 60 + ss) - targetSeconds);
                    if (delta < bestDelta) {
                        bestDelta = delta;
                        bestIdx = fi;
                    }
                }

                if (bestIdx < 0) {
                    failDownload(idx, generation, "No timestamped archive volume");
                    return;
                }

                if (!isCurrentDownloadGeneration(generation)) return;
                auto fileResult = Downloader::httpGet(NEXRAD_HOST, "/" + files[bestIdx].key);
                if (fileResult.success && !fileResult.data.empty()) {
                    processDownload(idx, std::move(fileResult.data), generation,
                                    snapshotMode, lowestSweepOnly, dealiasEnabled,
                                    files[bestIdx].key);
                } else {
                    failDownload(idx, generation,
                                 fileResult.error.empty() ? "Archive download failed"
                                                           : fileResult.error);
                }
            }
        );
    }
}

bool App::tryProcessDownload(int stationIdx, std::vector<uint8_t> data, uint64_t generation,
                             bool snapshotMode, bool lowestSweepOnly, bool dealiasEnabled,
                             const std::string& volumeKey) {
    if (!isCurrentDownloadGeneration(generation)) return false;

    PipelineStageTimings timings = {};
    bool keepFull = snapshotMode || m_historicMode;
    if (!keepFull) {
        std::lock_guard<std::mutex> lock(m_stationMutex);
        keepFull = shouldKeepStationFullVolumeLocked(stationIdx, m_activeStationIdx);
    }

    const bool preferFastLowestSweep = (snapshotMode && lowestSweepOnly) || (!keepFull && !m_historicMode);
    const std::string stationName = NEXRAD_STATIONS[stationIdx].icao;
    const float fallbackLat = NEXRAD_STATIONS[stationIdx].lat;
    const float fallbackLon = NEXRAD_STATIONS[stationIdx].lon;

    auto decodeStart = Clock::now();
    std::vector<uint8_t> decoded = Level2Parser::decodeArchiveBytes(data);
    timings.decode_ms = elapsedMs(decodeStart, Clock::now());
    if (decoded.empty())
        return false;

    Detection detection = {};
    bool haveDetection = false;
    float detectionLat = fallbackLat;
    float detectionLon = fallbackLon;

    const bool runPreviewDetect = keepFull && !snapshotMode && !m_historicMode;
    FastGpuWorkingSet previewWorkingSet;
    if (runPreviewDetect) {
        auto previewBuildStart = Clock::now();
        previewWorkingSet = buildGpuWorkingSetFromDecoded(decoded, GpuWorkingSetMode::LowTilts);
        timings.gpu_detect_build_ms = elapsedMs(previewBuildStart, Clock::now());
        if (previewWorkingSet.success && !previewWorkingSet.sweeps.empty()) {
            if (dealiasEnabled) {
                auto previewPreprocessStart = Clock::now();
                dealiasPrecomputedSweeps(previewWorkingSet.sweeps);
                timings.gpu_detect_preprocess_ms = elapsedMs(previewPreprocessStart, Clock::now());
            }

            auto previewDetectStart = Clock::now();
            detection = computeDetectionForSweeps(previewWorkingSet.sweeps, detectionLat, detectionLon, stationName);
            timings.gpu_detect_ms = elapsedMs(previewDetectStart, Clock::now());
            timings.used_gpu_detect_stage = detection.computed;
            haveDetection = detection.computed;

            if (haveDetection) {
                std::lock_guard<std::mutex> lock(m_stationMutex);
                if (!isCurrentDownloadGeneration(generation)) return false;
                auto& st = m_stations[stationIdx];
                st.detection = detection;
                st.timings = timings;
            }
        }
    }

    std::vector<PrecomputedSweep> precomp;
    int totalSweeps = 0;
    bool finalAlreadyPreprocessed = false;

    const GpuWorkingSetMode finalMode = preferFastLowestSweep
        ? GpuWorkingSetMode::LowestSweep
        : (keepFull ? GpuWorkingSetMode::AllSweeps : GpuWorkingSetMode::LowTilts);

    if (runPreviewDetect && finalMode == GpuWorkingSetMode::LowTilts &&
        previewWorkingSet.success && !previewWorkingSet.sweeps.empty()) {
        precomp = std::move(previewWorkingSet.sweeps);
        totalSweeps = previewWorkingSet.total_sweeps;
        finalAlreadyPreprocessed = dealiasEnabled;
        timings.sweep_build_ms = timings.gpu_detect_build_ms;
        timings.used_gpu_sweep_build = true;
    }

    if (precomp.empty()) {
        auto buildStart = Clock::now();
        auto gpuWorkingSet = buildGpuWorkingSetFromDecoded(decoded, finalMode);
        timings.sweep_build_ms = elapsedMs(buildStart, Clock::now());
        if (gpuWorkingSet.success && !gpuWorkingSet.sweeps.empty()) {
            precomp = std::move(gpuWorkingSet.sweeps);
            totalSweeps = gpuWorkingSet.total_sweeps;
            timings.used_gpu_sweep_build = true;
        }
    }

    ParsedRadarData parsed = {};
    if (precomp.empty()) {
        auto parseStart = Clock::now();
        parsed = Level2Parser::parseDecodedMessages(decoded, stationName);
        timings.parse_ms = elapsedMs(parseStart, Clock::now());
        if (parsed.sweeps.empty())
            return false;

        detectionLat = parsed.station_lat != 0.0f ? parsed.station_lat : fallbackLat;
        detectionLon = parsed.station_lon != 0.0f ? parsed.station_lon : fallbackLon;

        auto buildStart = Clock::now();
        precomp = keepFull ? buildPrecomputedSweeps(parsed)
                           : buildReducedWorkingSetSweeps(parsed);
        timings.sweep_build_ms = elapsedMs(buildStart, Clock::now());
        totalSweeps = (int)parsed.sweeps.size();
    }

    if (!finalAlreadyPreprocessed) {
        auto preprocessStart = Clock::now();
        if (dealiasEnabled)
            dealiasPrecomputedSweeps(precomp);
        if (snapshotMode)
            suppressReflectivityRingArtifacts(precomp);
        timings.preprocess_ms = elapsedMs(preprocessStart, Clock::now());
    } else if (snapshotMode) {
        auto ringStart = Clock::now();
        suppressReflectivityRingArtifacts(precomp);
        timings.preprocess_ms += elapsedMs(ringStart, Clock::now());
    }

    if (!haveDetection) {
        auto detectStart = Clock::now();
        detection = computeDetectionForSweeps(precomp, detectionLat, detectionLon, stationName);
        timings.detection_ms = elapsedMs(detectStart, Clock::now());
        haveDetection = detection.computed;
    }

    if (snapshotMode && lowestSweepOnly) {
        int lowestIdx = findLowestSweepIndex(precomp);
        if (lowestIdx >= 0) {
            std::vector<PrecomputedSweep> reducedPrecomp;
            reducedPrecomp.push_back(std::move(precomp[lowestIdx]));
            precomp.swap(reducedPrecomp);
        }
    }

    {
        std::lock_guard<std::mutex> lock(m_stationMutex);
        if (!isCurrentDownloadGeneration(generation)) return false;
        auto& st = m_stations[stationIdx];
        if (!snapshotMode && !m_historicMode && !st.enabled) {
            if (st.downloading) {
                st.downloading = false;
                if (m_stationsDownloading.load() > 0)
                    m_stationsDownloading--;
            }
            return true;
        }
        st.raw_volume_data = std::move(data);
        st.total_sweeps = totalSweeps > 0 ? totalSweeps : (int)precomp.size();
        st.lowest_sweep_elev = precomp.empty() ? 0.0f : precomp.front().elevation_angle;
        st.lowest_sweep_radials = precomp.empty() ? 0 : precomp.front().num_radials;
        st.data_lat = detectionLat;
        st.data_lon = detectionLon;
        st.precomputed = std::move(precomp);
        st.full_volume_resident = keepFull && !(snapshotMode && lowestSweepOnly);
        st.parsed = true;
        st.failed = false;
        st.error.clear();
        st.detection = std::move(detection);
        st.timings = timings;
        if (!snapshotMode && !m_historicMode)
            appendLiveHistoryLocked(st, volumeKey, st.precomputed, detectionLat, detectionLon);
        if (st.downloading) {
            st.downloading = false;
            m_stationsDownloading--;
        }
        st.lastUpdate = std::chrono::steady_clock::now();
        st.latestVolumeKey = volumeKey;

        if (!keepFull)
            trimStationToLowestSweepLocked(st);
    }

    {
        std::lock_guard<std::mutex> lock(m_uploadMutex);
        if (isCurrentDownloadGeneration(generation))
            m_uploadQueue.push_back(stationIdx);
    }
    if (!snapshotMode && !m_historicMode)
        trimLiveHistoryWorkingSet(m_activeStationIdx);
    return true;
}

void App::processDownload(int stationIdx, std::vector<uint8_t> data, uint64_t generation,
                          bool snapshotMode, bool lowestSweepOnly, bool dealiasEnabled,
                          const std::string& volumeKey) {
    if (!tryProcessDownload(stationIdx, std::move(data), generation,
                            snapshotMode, lowestSweepOnly, dealiasEnabled, volumeKey)) {
        failDownload(stationIdx, generation, "Parse failed: no sweeps");
    }
}

// Build a list of "best" sweep indices for a product:
// At each unique elevation, keep only the sweep with the most gates.
// This deduplicates split-cut sweeps and removes junk tilts.
static std::vector<int> getBestSweeps(const std::vector<PrecomputedSweep>& sweeps, int product) {
    // Collect all sweeps that have this product, grouped by elevation
    struct ElevEntry { int sweepIdx; float elev; int gates; };
    std::vector<ElevEntry> candidates;
    for (int i = 0; i < (int)sweeps.size(); i++) {
        if (sweeps[i].products[product].has_data && sweeps[i].num_radials > 0) {
            candidates.push_back({i, sweeps[i].elevation_angle,
                                   sweeps[i].products[product].num_gates});
        }
    }

    // For each unique elevation (within 0.3°), keep the one with most gates
    std::vector<int> best;
    for (auto& c : candidates) {
        bool dominated = false;
        for (auto& b : best) {
            float de = fabsf(sweeps[b].elevation_angle - c.elev);
            if (de < 0.3f) {
                // Same elevation - keep the one with more gates
                if (c.gates > sweeps[b].products[product].num_gates) {
                    b = c.sweepIdx; // replace with better one
                }
                dominated = true;
                break;
            }
        }
        if (!dominated) {
            best.push_back(c.sweepIdx);
        }
    }

    std::sort(best.begin(), best.end(),
              [&sweeps](int a, int b) {
                  return sweeps[a].elevation_angle < sweeps[b].elevation_angle;
              });
    return best;
}

static int findProductSweep(const std::vector<PrecomputedSweep>& sweeps, int product, int tiltIdx) {
    auto best = getBestSweeps(sweeps, product);
    if (best.empty()) return 0;
    if (tiltIdx < 0) tiltIdx = 0;
    if (tiltIdx >= (int)best.size()) tiltIdx = (int)best.size() - 1;
    return best[tiltIdx];
}

static int findProductSweepNearestAngle(const std::vector<PrecomputedSweep>& sweeps,
                                        int product, float targetAngle) {
    auto best = getBestSweeps(sweeps, product);
    if (best.empty()) return 0;

    int bestSweep = best[0];
    float bestDelta = fabsf(sweeps[bestSweep].elevation_angle - targetAngle);
    for (int idx : best) {
        float delta = fabsf(sweeps[idx].elevation_angle - targetAngle);
        if (delta < bestDelta ||
            (fabsf(delta - bestDelta) < 0.001f &&
             sweeps[idx].products[product].num_gates > sweeps[bestSweep].products[product].num_gates)) {
            bestSweep = idx;
            bestDelta = delta;
        }
    }
    return bestSweep;
}

static int countProductSweeps(const std::vector<PrecomputedSweep>& sweeps, int product) {
    return (int)getBestSweeps(sweeps, product).size();
}

std::vector<StationUiState> App::stations() const {
    std::unordered_set<int> pinnedCopy;
    std::vector<int> priorityCopy;
    {
        std::lock_guard<std::mutex> priorityLock(m_priorityMutex);
        pinnedCopy = m_pinnedStations;
        priorityCopy = m_dynamicPriorityStations;
    }

    std::lock_guard<std::mutex> lock(m_stationMutex);
    std::vector<StationUiState> snapshot;
    snapshot.reserve(m_stations.size());
    for (const auto& st : m_stations) {
        StationUiState ui = {};
        ui.index = st.index;
        ui.icao = st.icao;
        ui.lat = st.lat;
        ui.lon = st.lon;
        ui.display_lat = st.gpuInfo.lat != 0.0f ? st.gpuInfo.lat : st.lat;
        ui.display_lon = st.gpuInfo.lon != 0.0f ? st.gpuInfo.lon : st.lon;
        ui.latest_scan_utc = formatVolumeKeyTimestamp(st.latestVolumeKey);
        ui.enabled = st.enabled;
        ui.pinned = pinnedCopy.find(st.index) != pinnedCopy.end();
        ui.priority_hot = ui.pinned ||
            std::find(priorityCopy.begin(), priorityCopy.end(), st.index) != priorityCopy.end();
        ui.downloading = st.downloading;
        ui.parsed = st.parsed;
        ui.uploaded = st.uploaded;
        ui.rendered = st.rendered;
        ui.failed = st.failed;
        ui.error = st.error;
        ui.sweep_count = st.total_sweeps;
        ui.lowest_elev = st.lowest_sweep_elev;
        ui.lowest_radials = st.lowest_sweep_radials;
        ui.detection = st.detection;
        ui.timings = st.timings;
        snapshot.push_back(std::move(ui));
    }
    return snapshot;
}

std::vector<WarningPolygon> App::currentWarnings() const {
    std::vector<WarningPolygon> source;
    if (m_historicMode) {
        const RadarFrame* fr = m_historic.frame(m_historic.currentFrame());
        if (fr && !fr->valid_time_iso.empty())
            source = m_warnings.getHistoricWarnings(fr->valid_time_iso);
    } else if (m_snapshotMode && !m_snapshotTimestampIso.empty()) {
        source = m_warnings.getHistoricWarnings(m_snapshotTimestampIso);
    } else {
        source = m_warnings.getWarnings();
    }

    std::vector<WarningPolygon> filtered;
    filtered.reserve(source.size());
    for (auto& warning : source) {
        if (!m_warningOptions.allows(warning)) continue;
        warning.color = m_warningOptions.resolvedColor(warning);
        warning.line_width = m_warningOptions.resolvedLineWidth(warning);
        filtered.push_back(std::move(warning));
    }
    return filtered;
}

bool App::loadColorTableFromFile(const std::string& path) {
    ParsedColorTable table;
    std::string error;
    if (!loadColorTableFile(path, table, error)) {
        m_colorTableStatus = "Palette load failed: " + error;
        return false;
    }

    gpu::setColorTable(table.product, table.colors.data());
    m_colorTableLabels[table.product] = table.label;
    m_colorTableStatus = table.format + " loaded for " + PRODUCT_INFO[table.product].name +
        " from " + table.label;
    invalidateFrameCache(true);
    m_needsRerender = true;
    return true;
}

void App::resetColorTable(int product) {
    if (product < 0 || product >= NUM_PRODUCTS)
        product = m_activeProduct;
    gpu::resetColorTable(product);
    m_colorTableLabels[product].clear();
    m_colorTableStatus = std::string("Reset ") + PRODUCT_INFO[product].name + " to built-in palette";
    invalidateFrameCache(true);
    m_needsRerender = true;
}

bool App::stationUploadMatchesSelection(const StationState& st) const {
    const bool lowestSweepUpload = m_showAll || m_snapshotMode;
    if (!st.uploaded ||
        st.uploaded_product != m_activeProduct ||
        st.uploaded_lowest_sweep != lowestSweepUpload) {
        return false;
    }

    return lowestSweepUpload || st.uploaded_tilt == m_activeTilt;
}

void App::invalidateFrameCache(bool freeMemory) {
    invalidateLiveLoop(freeMemory);
    if (freeMemory) {
        for (auto& frame : m_cachedFrames) {
            if (frame) {
                cudaFree(frame);
                frame = nullptr;
            }
        }
    }
    m_cachedFrameCount = 0;
    m_cachedFrameWidth = 0;
    m_cachedFrameHeight = 0;
    m_memoryTelemetry.historic_cache_bytes = 0;
}

void App::ensureCrossSectionBuffer(int width, int height) {
    if (width <= 0 || height <= 0) return;
    if (m_d_xsOutput &&
        width == m_xsAllocWidth &&
        height == m_xsAllocHeight) {
        return;
    }

    if (m_d_xsOutput) {
        cudaFree(m_d_xsOutput);
        m_d_xsOutput = nullptr;
    }

    CUDA_CHECK(cudaMalloc(&m_d_xsOutput, (size_t)width * height * sizeof(uint32_t)));
    m_xsAllocWidth = width;
    m_xsAllocHeight = height;
}

void App::rebuildVolumeForCurrentSelection() {
    m_volumeBuilt = false;
    m_volumeStation = -1;

    auto buildVolumeFromSweeps = [&](const std::vector<PrecomputedSweep>& sweeps,
                                     float slat, float slon, int stationSlot) {
        int ns = (int)sweeps.size();
        if (ns <= 0) return;

        std::vector<GpuStationInfo> sweepInfos(ns);
        std::vector<const float*> azPtrs(ns);
        std::vector<const uint16_t*> gatePtrs(ns);

        int builtSweeps = 0;
        for (int s = 0; s < ns && s < 32; s++) {
            const auto& pc = sweeps[s];
            int slot = 200 + s;
            if (slot >= MAX_STATIONS || pc.num_radials <= 0) break;

            GpuStationInfo info = {};
            info.lat = slat;
            info.lon = slon;
            info.elevation_angle = pc.elevation_angle;
            info.num_radials = pc.num_radials;
            for (int p = 0; p < NUM_PRODUCTS; p++) {
                const auto& pd = pc.products[p];
                if (!pd.has_data) continue;
                info.has_product[p] = true;
                info.num_gates[p] = pd.num_gates;
                info.first_gate_km[p] = pd.first_gate_km;
                info.gate_spacing_km[p] = pd.gate_spacing_km;
                info.scale[p] = pd.scale;
                info.offset[p] = pd.offset;
            }

            gpu::allocateStation(slot, info);
            const uint16_t* gp[NUM_PRODUCTS] = {};
            for (int p = 0; p < NUM_PRODUCTS; p++) {
                if (pc.products[p].has_data && !pc.products[p].gates.empty())
                    gp[p] = pc.products[p].gates.data();
            }
            gpu::uploadStationData(slot, info, pc.azimuths.data(), gp);
            gpu::syncStation(slot);

            sweepInfos[builtSweeps] = info;
            azPtrs[builtSweeps] = gpu::getStationAzimuths(slot);
            gatePtrs[builtSweeps] = gpu::getStationGates(slot, m_activeProduct);
            builtSweeps++;
        }

        if (builtSweeps <= 0) return;

        gpu::buildVolume(stationSlot, m_activeProduct,
                         sweepInfos.data(), builtSweeps,
                         azPtrs.data(), gatePtrs.data());
        m_volumeBuilt = true;
        m_volumeStation = stationSlot;
    };

    if (m_historicMode) {
        const RadarFrame* fr = m_historic.frame(m_historic.currentFrame());
        if (fr && fr->ready && !fr->sweeps.empty())
            buildVolumeFromSweeps(fr->sweeps, fr->station_lat, fr->station_lon, 0);
        return;
    }

    if (m_activeStationIdx < 0 || m_activeStationIdx >= (int)m_stations.size()) return;
    ensureStationFullVolume(m_activeStationIdx);

    bool needsUpload = false;
    {
        std::lock_guard<std::mutex> lock(m_stationMutex);
        needsUpload = !stationUploadMatchesSelection(m_stations[m_activeStationIdx]);
    }
    if (needsUpload)
        uploadStation(m_activeStationIdx);

    std::lock_guard<std::mutex> lock(m_stationMutex);
    const auto& st = m_stations[m_activeStationIdx];
    if (!stationUploadMatchesSelection(st) || st.precomputed.empty()) return;

    float slat = st.gpuInfo.lat != 0 ? st.gpuInfo.lat : st.lat;
    float slon = st.gpuInfo.lon != 0 ? st.gpuInfo.lon : st.lon;
    buildVolumeFromSweeps(st.precomputed, slat, slon, m_activeStationIdx);
}

void App::refreshActiveTiltMetadata() {
    if (m_historicMode) {
        const RadarFrame* fr = m_historic.frame(m_historic.currentFrame());
        if (!fr || !fr->ready || fr->sweeps.empty()) return;
        int productTilts = countProductSweeps(fr->sweeps, m_activeProduct);
        if (productTilts <= 0) return;
        if (m_activeTilt >= productTilts) m_activeTilt = productTilts - 1;
        m_maxTilts = productTilts;
        int sweepIdx = findProductSweep(fr->sweeps, m_activeProduct, m_activeTilt);
        m_activeTiltAngle = fr->sweeps[sweepIdx].elevation_angle;
        return;
    }

    if (m_activeStationIdx < 0 || m_activeStationIdx >= (int)m_stations.size()) return;
    std::lock_guard<std::mutex> lock(m_stationMutex);
    const auto& st = m_stations[m_activeStationIdx];
    if (st.precomputed.empty()) return;

    if (!st.full_volume_resident) {
        m_maxTilts = std::max(1, st.total_sweeps);
        m_activeTiltAngle = st.precomputed[0].elevation_angle;
        return;
    }

    int productTilts = countProductSweeps(st.precomputed, m_activeProduct);
    if (productTilts <= 0) return;
    if (m_activeTilt >= productTilts) m_activeTilt = productTilts - 1;
    m_maxTilts = productTilts;
    int sweepIdx = findProductSweep(st.precomputed, m_activeProduct, m_activeTilt);
    m_activeTiltAngle = st.precomputed[sweepIdx].elevation_angle;
}

int App::currentAvailableTilts() {
    if (m_historicMode) {
        const RadarFrame* fr = m_historic.frame(m_historic.currentFrame());
        if (!fr || !fr->ready || fr->sweeps.empty()) return 1;
        const int tilts = countProductSweeps(fr->sweeps, m_activeProduct);
        return std::max(1, tilts);
    }

    if (m_activeStationIdx < 0 || m_activeStationIdx >= (int)m_stations.size())
        return std::max(1, m_maxTilts);

    std::lock_guard<std::mutex> lock(m_stationMutex);
    const auto& st = m_stations[m_activeStationIdx];
    if (st.precomputed.empty())
        return std::max(1, m_maxTilts);
    if (!st.full_volume_resident)
        return std::max(1, st.total_sweeps);

    const int tilts = countProductSweeps(st.precomputed, m_activeProduct);
    return std::max(1, tilts);
}

void App::uploadStation(int stationIdx) {
    if (stationIdx < 0 || stationIdx >= (int)m_stations.size())
        return;

    const bool lowestSweepMosaic = m_showAll || m_snapshotMode;
    if (!lowestSweepMosaic && stationIdx == m_activeStationIdx) {
        bool needsPromotion = (m_activeTilt > 0) || m_crossSection || m_mode3D;
        if (!needsPromotion) {
            std::lock_guard<std::mutex> lock(m_stationMutex);
            const auto& st = m_stations[stationIdx];
            const bool hasCurrentProduct = countProductSweeps(st.precomputed, m_activeProduct) > 0;
            const bool hasTrimmedSweeps = st.total_sweeps > (int)st.precomputed.size();
            needsPromotion = !st.full_volume_resident && hasTrimmedSweeps && !hasCurrentProduct;
        }
        if (needsPromotion)
            ensureStationFullVolume(stationIdx);
    }

    std::lock_guard<std::mutex> lock(m_stationMutex);
    auto& st = m_stations[stationIdx];
    if (st.precomputed.empty()) return;

    // Filter sweeps by active product - only show tilts that have this product
    int productTilts = countProductSweeps(st.precomputed, m_activeProduct);
    if (productTilts <= 0) {
        gpu::freeStation(stationIdx);
        st.gpuInfo = {};
        st.uploaded = false;
        st.uploaded_product = -1;
        st.uploaded_tilt = -1;
        st.uploaded_sweep = -1;
        st.uploaded_lowest_sweep = false;
        m_gridDirty = true;
        return;
    }
    if (!lowestSweepMosaic && stationIdx == m_activeStationIdx)
        m_maxTilts = productTilts;
    int sweepIdx = lowestSweepMosaic
        ? findProductSweep(st.precomputed, m_activeProduct, 0)
        : findProductSweep(st.precomputed, m_activeProduct, m_activeTilt);
    auto& pc = st.precomputed[sweepIdx];
    if (pc.num_radials == 0) return;

    if (!lowestSweepMosaic && (stationIdx == m_activeStationIdx || m_activeStationIdx < 0))
        m_activeTiltAngle = pc.elevation_angle;

    // Build GpuStationInfo from precomputed data
    GpuStationInfo info = {};
    info.lat = st.lat;
    info.lon = st.lon;
    if (st.data_lat != 0.0f) info.lat = st.data_lat;
    if (st.data_lon != 0.0f) info.lon = st.data_lon;
    info.elevation_angle = pc.elevation_angle;
    info.num_radials = pc.num_radials;

    for (int p = 0; p < NUM_PRODUCTS; p++) {
        auto& pd = pc.products[p];
        if (!pd.has_data) continue;
        info.has_product[p] = true;
        info.num_gates[p] = pd.num_gates;
        info.first_gate_km[p] = pd.first_gate_km;
        info.gate_spacing_km[p] = pd.gate_spacing_km;
        info.scale[p] = pd.scale;
        info.offset[p] = pd.offset;
    }

    auto uploadStart = Clock::now();
    gpu::allocateStation(stationIdx, info);

    // Upload precomputed data (fast - just memcpy, no transposition)
    const uint16_t* gatePtrs[NUM_PRODUCTS] = {};
    for (int p = 0; p < NUM_PRODUCTS; p++) {
        if (pc.products[p].has_data && !pc.products[p].gates.empty())
            gatePtrs[p] = pc.products[p].gates.data();
    }

    gpu::uploadStationData(stationIdx, info, pc.azimuths.data(), gatePtrs);
    st.timings.upload_ms = elapsedMs(uploadStart, Clock::now());

    st.gpuInfo = info;
    st.uploaded_product = m_activeProduct;
    st.uploaded_tilt = m_activeTilt;
    st.uploaded_sweep = sweepIdx;
    st.uploaded_lowest_sweep = lowestSweepMosaic;
    if (!st.uploaded) {
        st.uploaded = true;
        int loaded = ++m_stationsLoaded;
        printf("GPU upload [%d/%d]: %s (%d radials, elev %.1f, %d sweeps)\n",
               loaded, m_stationsTotal, st.icao.c_str(),
               info.num_radials, info.elevation_angle, (int)st.precomputed.size());
    }
    m_gridDirty = true;
}

void App::buildSpatialGrid() {
    if (!m_spatialGrid) return;

    // GPU spatial grid construction
    std::vector<GpuStationInfo> infos(m_stations.size());
    {
        std::lock_guard<std::mutex> lock(m_stationMutex);
        for (int i = 0; i < (int)m_stations.size(); i++)
            infos[i] = m_stations[i].gpuInfo;
    }

    gpu::buildSpatialGridGpu(infos.data(), (int)infos.size(), m_spatialGrid.get());
    m_gridDirty = false;
}

void App::updateLivePolling(std::chrono::steady_clock::time_point now) {
    std::vector<int> activeDue;
    std::vector<std::pair<int, float>> priorityDue;
    std::vector<std::pair<int, float>> visibleDue;
    std::vector<std::pair<int, float>> backgroundDue;
    std::unordered_set<int> pinnedCopy;
    std::vector<int> dynamicPriorityCopy;
    std::vector<ProbSevereObject> probSevereCopy;

    {
        std::lock_guard<std::mutex> priorityLock(m_priorityMutex);
        pinnedCopy = m_pinnedStations;
        dynamicPriorityCopy = m_dynamicPriorityStations;
        probSevereCopy = m_probSevereObjects;
    }

    {
        std::lock_guard<std::mutex> lock(m_stationMutex);
        activeDue.reserve(1);
        priorityDue.reserve(m_stations.size());
        visibleDue.reserve(m_stations.size());
        backgroundDue.reserve(m_stations.size());

        for (int i = 0; i < (int)m_stations.size(); i++) {
            const auto& st = m_stations[i];
            if (!st.enabled) continue;
            if (st.downloading) continue;

            const bool active = (i == m_activeStationIdx);
            const bool pinned = pinnedCopy.find(i) != pinnedCopy.end();
            const bool dynamicPriority =
                std::find(dynamicPriorityCopy.begin(), dynamicPriorityCopy.end(), i) != dynamicPriorityCopy.end();
            const bool visible = stationLikelyVisible(i);
            const bool coldUnloaded =
                !st.parsed && !st.uploaded && st.latestVolumeKey.empty();

            const float elapsed = st.lastPollAttempt.time_since_epoch().count() == 0
                ? std::numeric_limits<float>::max()
                : std::chrono::duration<float>(now - st.lastPollAttempt).count();
            if (elapsed < livePollIntervalSecForStation(i, st))
                continue;

            // Keep cold sites completely idle unless the operator is looking at them
            // or the scheduler explicitly promoted them.
            if (!active && !pinned && !dynamicPriority && !visible && coldUnloaded)
                continue;

            if (active) {
                activeDue.push_back(i);
            } else if (pinned || dynamicPriority) {
                float rankScore = priorityScoreForStation(i, probSevereCopy);
                if (pinned)
                    rankScore += 100000.0f;
                priorityDue.emplace_back(i, -rankScore);
            } else if (visible) {
                const float dLat = NEXRAD_STATIONS[i].lat - m_viewport.center_lat;
                const float dLon = NEXRAD_STATIONS[i].lon - m_viewport.center_lon;
                visibleDue.emplace_back(i, dLat * dLat + dLon * dLon);
            } else if (st.parsed || st.uploaded || !st.latestVolumeKey.empty()) {
                backgroundDue.emplace_back(i, -elapsed);
            }
        }
    }

    std::sort(priorityDue.begin(), priorityDue.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    std::sort(visibleDue.begin(), visibleDue.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    std::sort(backgroundDue.begin(), backgroundDue.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    for (int idx : activeDue)
        queueLiveStationRefresh(idx, true);

    int priorityBudget = m_maxPriorityPollsPerSweep;
    for (const auto& entry : priorityDue) {
        if (priorityBudget-- <= 0) break;
        queueLiveStationRefresh(entry.first, true);
    }

    int visibleBudget = m_maxVisiblePollsPerSweep;
    for (const auto& entry : visibleDue) {
        if (visibleBudget-- <= 0) break;
        queueLiveStationRefresh(entry.first, true);
    }

    int backgroundBudget = m_maxBackgroundPollsPerSweep;
    for (const auto& entry : backgroundDue) {
        if (backgroundBudget-- <= 0) break;
        queueLiveStationRefresh(entry.first, true);
    }
}

void App::update(float dt) {
    // Historic mode: lock to event station, upload only on frame change
    if (m_historicMode) {
        if (m_historic.downloadedFrames() > 0) {
            m_historic.update(dt);
            int curFrame = m_historic.currentFrame();

            // If current frame isn't ready, find nearest ready one
            const RadarFrame* fr = m_historic.frame(curFrame);
            if (!fr || !fr->ready) {
                for (int i = 0; i < m_historic.numFrames(); i++) {
                    if (m_historic.frame(i) && m_historic.frame(i)->ready) {
                        curFrame = i;
                        m_historic.setFrame(i);
                        break;
                    }
                }
            }

            // Only upload when frame actually changes
            if (curFrame != m_lastHistoricFrame) {
                fr = m_historic.frame(curFrame);
                if (fr && fr->ready) {
                    uploadHistoricFrame(curFrame);
                    if (!fr->valid_time_iso.empty())
                        m_warnings.requestHistoricSnapshot(fr->valid_time_iso);
                    m_lastHistoricFrame = curFrame;
                    printf("Historic frame %d: %s\n", curFrame, fr->timestamp.c_str());
                }
            }
        }
        return;
    }

    // Process GPU upload queue
    {
        std::lock_guard<std::mutex> lock(m_uploadMutex);
        for (int idx : m_uploadQueue) {
            uploadStation(idx);
            if (m_liveLoopEnabled && !m_snapshotMode && !m_historicMode &&
                !m_mode3D && !m_crossSection &&
                (m_showAll || idx == m_activeStationIdx)) {
                requestLiveLoopCapture();
            }
        }
        m_uploadQueue.clear();
    }

    auto now = std::chrono::steady_clock::now();
    const float schedulerElapsed =
        std::chrono::duration<float>(now - m_lastLivePollSweep).count();
    if (!m_snapshotMode && schedulerElapsed >= m_livePollSweepIntervalSec) {
        updateLivePolling(now);
        m_lastLivePollSweep = now;
    }

    processLiveLoopBackfill();
    updateLiveLoop(dt);
}

void App::render() {
    // Rebuild spatial grid if needed
    if (m_gridDirty) {
        buildSpatialGrid();
    }

    {
        const int rw = renderWidth();
        const int rh = renderHeight();
        GpuViewport gpuVp;
        gpuVp.center_lat = (float)m_viewport.center_lat;
        gpuVp.center_lon = (float)m_viewport.center_lon;
        gpuVp.deg_per_pixel_x =
            (1.0f / (float)m_viewport.zoom) * ((float)m_viewport.width / (float)rw);
        gpuVp.deg_per_pixel_y =
            (1.0f / (float)m_viewport.zoom) * ((float)m_viewport.height / (float)rh);
        gpuVp.width = rw;
        gpuVp.height = rh;

        float srvSpd = (m_srvMode && m_activeProduct == PROD_VEL) ? m_stormSpeed : 0.0f;
        float srvDir = m_stormDir;
        float activeThreshold = (m_activeProduct == PROD_VEL)
            ? m_velocityMinThreshold
            : m_dbzMinThreshold;
        const bool capturePending =
            m_liveLoopEnabled &&
            m_liveLoopCapturePending &&
            !m_historicMode &&
            !m_snapshotMode &&
            !m_mode3D &&
            !m_crossSection;
        const bool useLiveLoopFrame =
            !m_historicMode &&
            !m_snapshotMode &&
            !m_mode3D &&
            !m_crossSection &&
            m_liveLoopEnabled &&
            m_liveLoopCount > 0 &&
            !capturePending &&
            (m_liveLoopPlaying || m_liveLoopPlaybackIndex < (m_liveLoopCount - 1)) &&
            m_liveLoopFrameWidth == gpuVp.width &&
            m_liveLoopFrameHeight == gpuVp.height;

        if (useLiveLoopFrame) {
            const int slot = liveLoopSlotForIndex(m_liveLoopPlaybackIndex);
            CUDA_CHECK(cudaMemcpy(m_d_compositeOutput, m_liveLoopFrames[slot],
                                  (size_t)gpuVp.width * gpuVp.height * sizeof(uint32_t),
                                  cudaMemcpyDeviceToDevice));
        } else if (m_mode3D && m_volumeBuilt) {
            // 3D volumetric ray march
            gpu::renderVolume(m_camera, gpuVp.width, gpuVp.height,
                              m_activeProduct, activeThreshold,
                              m_d_compositeOutput);
        } else if (m_historicMode) {
            int cf = m_historic.currentFrame();
            if (hasCachedFrame(cf, gpuVp.width, gpuVp.height)) {
                // Use pre-baked cached frame (zero render cost)
                CUDA_CHECK(cudaMemcpy(m_d_compositeOutput, m_cachedFrames[cf],
                                      (size_t)gpuVp.width * gpuVp.height * sizeof(uint32_t),
                                      cudaMemcpyDeviceToDevice));
            } else {
                if (m_activeProduct == PROD_VEL) {
                    gpu::forwardRenderStation(gpuVp, 0,
                                              m_activeProduct, activeThreshold,
                                              m_d_compositeOutput, srvSpd, srvDir);
                } else {
                    gpu::renderSingleStation(gpuVp, 0,
                                             m_activeProduct, activeThreshold,
                                             m_d_compositeOutput, srvSpd, srvDir);
                }
                // Cache this rendered frame for instant replay
                cacheAnimFrame(cf, m_d_compositeOutput, gpuVp.width, gpuVp.height);
            }
        } else if (m_showAll) {
            // Mosaic: all stations
            if (m_gridDirty) buildSpatialGrid();
            std::vector<GpuStationInfo> gpuInfos(m_stations.size());
            {
                std::lock_guard<std::mutex> lock(m_stationMutex);
                for (int i = 0; i < (int)m_stations.size(); i++)
                    gpuInfos[i] = m_stations[i].gpuInfo;
            }
            gpu::renderNative(gpuVp, gpuInfos.data(), (int)m_stations.size(),
                              *m_spatialGrid, m_activeProduct, activeThreshold,
                              m_d_compositeOutput);
        } else if (m_activeStationIdx >= 0) {
            // Single station: fast path
            bool needsUpload = false;
            bool canRender = false;
            if (m_activeStationIdx < (int)m_stations.size()) {
                std::lock_guard<std::mutex> lock(m_stationMutex);
                const auto& st = m_stations[m_activeStationIdx];
                canRender = st.enabled && st.uploaded;
                needsUpload = st.enabled && !stationUploadMatchesSelection(st);
            }
            if (needsUpload) {
                uploadStation(m_activeStationIdx);
                std::lock_guard<std::mutex> lock(m_stationMutex);
                const auto& st = m_stations[m_activeStationIdx];
                canRender = st.enabled && stationUploadMatchesSelection(st);
            }
            if (canRender) {
                gpu::forwardRenderStation(gpuVp, m_activeStationIdx,
                                          m_activeProduct, activeThreshold,
                                          m_d_compositeOutput, srvSpd, srvDir);
            } else {
                CUDA_CHECK(cudaMemset(m_d_compositeOutput, 0x0F,
                            (size_t)gpuVp.width * gpuVp.height * sizeof(uint32_t)));
            }
        } else {
            CUDA_CHECK(cudaMemset(m_d_compositeOutput, 0x0F,
                        (size_t)gpuVp.width * gpuVp.height * sizeof(uint32_t)));
        }

        if (!useLiveLoopFrame &&
            !m_historicMode &&
            !m_snapshotMode &&
            !m_mode3D &&
            !m_crossSection &&
            m_liveLoopEnabled &&
            m_liveLoopCapturePending) {
            captureLiveLoopFrame(m_d_compositeOutput, gpuVp.width, gpuVp.height,
                                 currentLiveLoopCaptureLabel());
        }
        // Cross-section: render to separate texture for floating panel
        // In historic mode, use slot 0's data; otherwise use active station
        int xsStationSlot = m_historicMode ? 0 : m_activeStationIdx;
        if (m_crossSection && m_volumeBuilt && xsStationSlot >= 0 &&
            xsStationSlot < (int)m_stations.size()) {
            GpuStationInfo stInfo = {};
            {
                std::lock_guard<std::mutex> lock(m_stationMutex);
                stInfo = m_stations[xsStationSlot].gpuInfo;
            }
            m_xsWidth = gpuVp.width;
            m_xsHeight = gpuVp.height / 3;
            if (m_xsHeight < 200) m_xsHeight = 200;

            // Ensure cross-section GPU buffer and GL texture exist
            ensureCrossSectionBuffer(m_xsWidth, m_xsHeight);
            m_xsTex.resize(m_xsWidth, m_xsHeight);

            gpu::renderCrossSection(
                m_activeStationIdx, m_activeProduct, activeThreshold,
                m_xsStartLat, m_xsStartLon, m_xsEndLat, m_xsEndLon,
                stInfo.lat, stInfo.lon,
                m_xsWidth, m_xsHeight, m_d_xsOutput);

            // Copy to its own GL texture
            m_xsTex.updateFromDevice(m_d_xsOutput, m_xsWidth, m_xsHeight);
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        m_outputTex.updateFromDevice(m_d_compositeOutput,
                                      gpuVp.width, gpuVp.height);
        updateMemoryTelemetry();
    }
}

void App::onScroll(double xoff, double yoff) {
    invalidateFrameCache(true);
    if (m_mode3D) {
        m_camera.distance *= (yoff > 0) ? 0.9f : 1.1f;
        m_camera.distance = std::max(50.0f, std::min(1500.0f, m_camera.distance));
    } else {
        double factor = (yoff > 0) ? 1.15 : 1.0 / 1.15;
        m_viewport.zoom *= factor;
        m_viewport.zoom = std::max(1.0, std::min(m_viewport.zoom, 2000.0));
    }
}

void App::onMouseDrag(double dx, double dy) {
    if (m_crossSection) {
        // Left-drag: grab and move the whole cross-section line
        // dx positive = mouse right = lon increases
        // dy positive = mouse down = lat decreases
        float dlon = (float)(dx / m_viewport.zoom);
        float dlat = (float)(-dy / m_viewport.zoom);
        m_xsStartLat += dlat;
        m_xsStartLon += dlon;
        m_xsEndLat += dlat;
        m_xsEndLon += dlon;
    } else {
        m_viewport.center_lon -= dx / m_viewport.zoom;
        m_viewport.center_lat += dy / m_viewport.zoom;
        invalidateFrameCache(true);
    }
}

void App::onMouseMove(double mx, double my) {
    // Convert mouse pixel to lat/lon
    m_mouseLon = (float)(m_viewport.center_lon + (mx - m_viewport.width * 0.5) / m_viewport.zoom);
    m_mouseLat = (float)(m_viewport.center_lat - (my - m_viewport.height * 0.5) / m_viewport.zoom);

    if (!m_autoTrackStation)
        return;

    // Find nearest station marker under the cursor
    float bestDist = 1e9f;
    int bestIdx = -1;
    {
        std::lock_guard<std::mutex> lock(m_stationMutex);
        for (int i = 0; i < (int)m_stations.size(); i++) {
            const float slat = m_stations[i].gpuInfo.lat != 0.0f ? m_stations[i].gpuInfo.lat : m_stations[i].lat;
            const float slon = m_stations[i].gpuInfo.lon != 0.0f ? m_stations[i].gpuInfo.lon : m_stations[i].lon;
            const float sx = (float)((slon - m_viewport.center_lon) * m_viewport.zoom + m_viewport.width * 0.5);
            const float sy = (float)((m_viewport.center_lat - slat) * m_viewport.zoom + m_viewport.height * 0.5);
            const float dx = (float)mx - sx;
            const float dy = (float)my - sy;
            const float dist = dx * dx + dy * dy;
            if (dist < bestDist) {
                bestDist = dist;
                bestIdx = i;
            }
        }
    }

    if (bestIdx >= 0 && bestDist > 24.0f * 24.0f)
        bestIdx = -1;

    if (bestIdx != m_activeStationIdx && bestIdx >= 0) {
        m_activeStationIdx = bestIdx;
        invalidateLiveLoop(false);
        bool needsUpload = false;
        bool enabled = false;
        {
            std::lock_guard<std::mutex> lock(m_stationMutex);
            enabled = m_stations[bestIdx].enabled;
            needsUpload = enabled && !stationUploadMatchesSelection(m_stations[bestIdx]);
        }
        if (needsUpload)
            uploadStation(bestIdx);
        if (enabled && !m_snapshotMode && !m_historicMode)
            queueLiveStationRefresh(bestIdx, true);
        trimStationWorkingSet(bestIdx);
        trimLiveHistoryWorkingSet(bestIdx);
    } else if (bestIdx < 0) {
        m_activeStationIdx = -1;
    }
}

std::string App::activeStationName() const {
    std::lock_guard<std::mutex> lock(m_stationMutex);
    if (m_activeStationIdx < 0 || m_activeStationIdx >= m_stationsTotal)
        return "None";
    return m_stations[m_activeStationIdx].icao;
}

int App::stationAtScreen(double mx, double my, float radiusPx) const {
    float bestDist = radiusPx * radiusPx;
    int bestIdx = -1;

    std::lock_guard<std::mutex> lock(m_stationMutex);
    for (int i = 0; i < (int)m_stations.size(); i++) {
        const float slat = m_stations[i].gpuInfo.lat != 0.0f ? m_stations[i].gpuInfo.lat : m_stations[i].lat;
        const float slon = m_stations[i].gpuInfo.lon != 0.0f ? m_stations[i].gpuInfo.lon : m_stations[i].lon;
        const float sx = (float)((slon - m_viewport.center_lon) * m_viewport.zoom + m_viewport.width * 0.5);
        const float sy = (float)((m_viewport.center_lat - slat) * m_viewport.zoom + m_viewport.height * 0.5);
        const float dx = (float)mx - sx;
        const float dy = (float)my - sy;
        const float dist = dx * dx + dy * dy;
        if (dist <= bestDist) {
            bestDist = dist;
            bestIdx = i;
        }
    }

    return bestIdx;
}

void App::toggleShowAll() {
    m_showAll = !m_showAll;
    m_mode3D = false;
    invalidateFrameCache(true);
    m_volumeBuilt = false;
    m_volumeStation = -1;

    if (m_historicMode)
        return;

    if (m_showAll || m_snapshotMode) {
        for (int i = 0; i < (int)m_stations.size(); i++)
            if (stationEnabled(i))
                uploadStation(i);
    } else if (m_activeStationIdx >= 0) {
        uploadStation(m_activeStationIdx);
    }

    trimStationWorkingSet(m_activeStationIdx);
    refreshActiveTiltMetadata();
}

void App::selectStation(int idx, bool centerView, double zoom) {
    if (idx < 0 || idx >= m_stationsTotal)
        return;

    setStationEnabled(idx, true);

    const bool stationChanged = (idx != m_activeStationIdx);
    m_activeStationIdx = idx;
    m_autoTrackStation = false;
    if (stationChanged)
        invalidateLiveLoop(false);

    if (centerView) {
        std::lock_guard<std::mutex> lock(m_stationMutex);
        if (idx < (int)m_stations.size()) {
            const auto& st = m_stations[idx];
            m_viewport.center_lat = st.gpuInfo.lat != 0.0f ? st.gpuInfo.lat : st.lat;
            m_viewport.center_lon = st.gpuInfo.lon != 0.0f ? st.gpuInfo.lon : st.lon;
            if (zoom > 0.0)
                m_viewport.zoom = zoom;
        }
    }

    bool needsUpload = false;
    {
        std::lock_guard<std::mutex> lock(m_stationMutex);
        if (idx < (int)m_stations.size())
            needsUpload = m_stations[idx].enabled && !stationUploadMatchesSelection(m_stations[idx]);
    }
    if (needsUpload)
        uploadStation(idx);
    if (!m_snapshotMode && !m_historicMode)
        queueLiveStationRefresh(idx, true);
    if (stationChanged && m_liveLoopEnabled)
        requestLiveLoopBackfill();

    trimStationWorkingSet(idx);
    trimLiveHistoryWorkingSet(idx);
    refreshActiveTiltMetadata();
    if (m_crossSection || m_mode3D)
        rebuildVolumeForCurrentSelection();
}

void App::onResize(int w, int h) {
    if (w <= 0 || h <= 0) return;
    m_viewport.width = w;
    m_viewport.height = h;
    m_needsComposite = true;
    m_needsRerender = true;
    invalidateFrameCache(true);
    updateMemoryTelemetry(true);
}

void App::onFramebufferResize(int w, int h) {
    if (w <= 0 || h <= 0) return;
    m_windowWidth = w;
    m_windowHeight = h;

    ensureRenderTargets();
    ensureCrossSectionBuffer(renderWidth(), std::max(200, renderHeight() / 3));
    invalidateFrameCache(true);
    m_needsComposite = true;
    m_needsRerender = true;
    updateMemoryTelemetry(true);
}

void App::setProduct(int p) {
    if (p < 0 || p >= (int)Product::COUNT) return;
    if (p == m_activeProduct) return;
    m_activeProduct = p;
    m_activeTilt = 0; // reset tilt - different products have different valid tilts
    m_lastHistoricFrame = -1; // force re-upload in historic mode
    m_maxTilts = 1;
    m_volumeBuilt = false;
    m_volumeStation = -1;
    invalidateFrameCache(true);
    m_needsRerender = true;

    if (m_historicMode) {
        if (m_crossSection || m_mode3D)
            rebuildVolumeForCurrentSelection();
    } else {
        if (m_showAll || m_snapshotMode) {
            for (int i = 0; i < (int)m_stations.size(); i++)
                uploadStation(i);
        } else if (m_activeStationIdx >= 0) {
            uploadStation(m_activeStationIdx);
        }
        if (m_crossSection || m_mode3D)
            rebuildVolumeForCurrentSelection();
    }
    trimStationWorkingSet(m_activeStationIdx);
    refreshActiveTiltMetadata();
    if (m_liveLoopEnabled && !m_historicMode && !m_snapshotMode && !m_showAll)
        requestLiveLoopBackfill();
}

void App::nextProduct() { setProduct((m_activeProduct + 1) % (int)Product::COUNT); }
void App::prevProduct() { setProduct((m_activeProduct - 1 + (int)Product::COUNT) % (int)Product::COUNT); }

void App::setTilt(int t) {
    if (!m_showAll && !m_snapshotMode && !m_historicMode && m_activeStationIdx >= 0)
        ensureStationFullVolume(m_activeStationIdx);
    m_maxTilts = currentAvailableTilts();
    if (t < 0) t = 0;
    if (t >= m_maxTilts) t = m_maxTilts - 1;
    if (t == m_activeTilt) {
        refreshActiveTiltMetadata();
        return;
    }
    m_activeTilt = t;
    m_volumeBuilt = false;
    m_volumeStation = -1;
    invalidateFrameCache(true);

    if (m_historicMode) {
        m_lastHistoricFrame = -1; // force re-upload with new tilt
    } else {
        if (m_showAll || m_snapshotMode) {
            for (int i = 0; i < (int)m_stations.size(); i++)
                uploadStation(i);
        } else if (m_activeStationIdx >= 0) {
            uploadStation(m_activeStationIdx);
            gpu::syncStation(m_activeStationIdx);
        }
    }
    if (m_crossSection || m_mode3D)
        rebuildVolumeForCurrentSelection();
    trimStationWorkingSet(m_activeStationIdx);
    refreshActiveTiltMetadata();
    if (m_liveLoopEnabled && !m_historicMode && !m_snapshotMode && !m_showAll)
        requestLiveLoopBackfill();
    m_needsRerender = true;
}

void App::nextTilt() { setTilt(m_activeTilt + 1); }
void App::prevTilt() { setTilt(m_activeTilt - 1); }

void App::setDbzMinThreshold(float v) {
    float* target = (m_activeProduct == PROD_VEL)
        ? &m_velocityMinThreshold
        : &m_dbzMinThreshold;
    if (v == *target) return;
    *target = v;
    if (m_historicMode)
        invalidateFrameCache(true);
    m_needsRerender = true;
}

void App::onRightDrag(double dx, double dy) {
    if (m_mode3D) {
        m_camera.orbit_angle += (float)dx * 0.3f;
        m_camera.tilt_angle -= (float)dy * 0.3f;
        m_camera.tilt_angle = std::max(5.0f, std::min(85.0f, m_camera.tilt_angle));
    } else if (m_crossSection) {
        // Right-drag endpoint of cross-section line
        m_xsEndLon = (float)(m_viewport.center_lon + (dx - m_viewport.width * 0.5) / m_viewport.zoom);
        m_xsEndLat = (float)(m_viewport.center_lat - (dy - m_viewport.height * 0.5) / m_viewport.zoom);
    }
}

void App::onMiddleClick(double mx, double my) {
    if (m_crossSection) {
        m_xsStartLon = (float)(m_viewport.center_lon + (mx - m_viewport.width * 0.5) / m_viewport.zoom);
        m_xsStartLat = (float)(m_viewport.center_lat - (my - m_viewport.height * 0.5) / m_viewport.zoom);
        m_xsEndLon = m_xsStartLon;
        m_xsEndLat = m_xsStartLat;
        m_xsDragging = true;
    }
}

void App::onMiddleDrag(double mx, double my) {
    if (m_crossSection && m_xsDragging) {
        m_xsEndLon = (float)(m_viewport.center_lon + (mx - m_viewport.width * 0.5) / m_viewport.zoom);
        m_xsEndLat = (float)(m_viewport.center_lat - (my - m_viewport.height * 0.5) / m_viewport.zoom);
    }
}

void App::toggleCrossSection() {
    m_crossSection = !m_crossSection;
    if (m_crossSection) {
        m_mode3D = false;

        // Position cross-section through the active station
        float slat = 0, slon = 0;
        if (m_historicMode) {
            auto* fr = m_historic.frame(m_historic.currentFrame());
            if (fr && fr->ready) { slat = fr->station_lat; slon = fr->station_lon; }
        } else if (m_activeStationIdx >= 0) {
            std::lock_guard<std::mutex> lock(m_stationMutex);
            slat = m_stations[m_activeStationIdx].gpuInfo.lat;
            slon = m_stations[m_activeStationIdx].gpuInfo.lon;
        }
        if (slat != 0) {
            m_xsStartLat = slat - 1.5f;
            m_xsStartLon = slon - 2.0f;
            m_xsEndLat = slat + 1.5f;
            m_xsEndLon = slon + 2.0f;
        }

        ensureCrossSectionBuffer(m_windowWidth, std::max(200, m_windowHeight / 3));

        if (m_historicMode) {
            // Force re-upload of current historic frame, which will build the volume
            m_lastHistoricFrame = -1;
        } else {
            // Build volume from live station data
            rebuildVolumeForCurrentSelection();
        }
    }
}

void App::toggle3D() {
    m_mode3D = !m_mode3D;
    m_showAll = false;
    if (m_mode3D) {
        m_camera = {32.0f, 24.0f, 440.0f, 54.0f};
        rebuildVolumeForCurrentSelection();
    }
}

void App::toggleSRV() {
    m_srvMode = !m_srvMode;
    invalidateFrameCache(true);
    m_needsRerender = true;
}

void App::setStormMotion(float speed, float dir) {
    m_stormSpeed = speed;
    m_stormDir = dir;
    invalidateFrameCache(true);
    m_needsRerender = true;
}

void App::rerenderAll() {
    m_needsRerender = true;
}

void App::loadHistoricEvent(int idx) {
    m_historicMode = true;
    m_autoTrackStation = false;
    m_showAll = false;
    m_snapshotMode = false;
    m_snapshotLowestSweepOnly = false;
    m_snapshotLabel.clear();
    m_snapshotTimestampIso.clear();
    m_lastHistoricFrame = -1;
    m_volumeBuilt = false;
    m_volumeStation = -1;
    invalidateFrameCache(true);
    m_warnings.clearHistoric();
    m_historic.loadEvent(idx);
    // Center viewport on the event
    if (idx >= 0 && idx < NUM_HISTORIC_EVENTS) {
        m_viewport.center_lat = HISTORIC_EVENTS[idx].center_lat;
        m_viewport.center_lon = HISTORIC_EVENTS[idx].center_lon;
        m_viewport.zoom = HISTORIC_EVENTS[idx].zoom;
    }
}

bool App::loadArchiveRange(const std::string& station,
                           int year, int month, int day,
                           int startHour, int startMin,
                           int endHour, int endMin) {
    const std::string stationCode = normalizeStationCode(station);
    const int stationIdx = findStationIndexByCode(stationCode);
    if (stationIdx < 0) {
        std::printf("Archive request rejected: unknown station '%s'\n", station.c_str());
        return false;
    }
    if (month < 1 || month > 12 || day < 1 || day > daysInMonth(year, month) ||
        startHour < 0 || startHour > 23 || endHour < 0 || endHour > 23 ||
        startMin < 0 || startMin > 59 || endMin < 0 || endMin > 59) {
        std::printf("Archive request rejected: invalid date/time range\n");
        return false;
    }

    m_lastRefresh = std::chrono::steady_clock::now();
    m_historic.cancel();
    m_historicMode = true;
    m_autoTrackStation = false;
    m_showAll = false;
    m_snapshotMode = false;
    m_snapshotLowestSweepOnly = false;
    m_snapshotLabel.clear();
    m_snapshotTimestampIso.clear();
    m_lastHistoricFrame = -1;
    m_volumeBuilt = false;
    m_volumeStation = -1;
    invalidateFrameCache(true);
    m_warnings.clearHistoric();

    const auto& site = NEXRAD_STATIONS[stationIdx];
    m_viewport.center_lat = site.lat;
    m_viewport.center_lon = site.lon;
    m_viewport.zoom = 180.0;

    const std::string label = makeArchiveRangeLabel(stationCode, year, month, day,
                                                    startHour, startMin, endHour, endMin);
    return m_historic.loadRange(label, stationCode, year, month, day,
                                startHour, startMin, endHour, endMin);
}

void App::uploadHistoricFrame(int frameIdx) {
    const RadarFrame* fr = m_historic.frame(frameIdx);
    if (!fr || !fr->ready || fr->sweeps.empty()) return;

    int slot = 0;
    // Filter by active product
    int productTilts = countProductSweeps(fr->sweeps, m_activeProduct);
    if (productTilts <= 0) {
        gpu::freeStation(slot);
        m_volumeBuilt = false;
        m_volumeStation = -1;
        return;
    }
    if (m_activeTilt >= productTilts) m_activeTilt = productTilts - 1;
    int sweepIdx = findProductSweep(fr->sweeps, m_activeProduct, m_activeTilt);
    auto& pc = fr->sweeps[sweepIdx];
    if (pc.num_radials == 0) return;

    m_maxTilts = productTilts;

    GpuStationInfo info = {};
    info.lat = fr->station_lat;
    info.lon = fr->station_lon;
    info.elevation_angle = pc.elevation_angle;
    info.num_radials = pc.num_radials;

    for (int p = 0; p < NUM_PRODUCTS; p++) {
        auto& pd = pc.products[p];
        if (!pd.has_data) continue;
        info.has_product[p] = true;
        info.num_gates[p] = pd.num_gates;
        info.first_gate_km[p] = pd.first_gate_km;
        info.gate_spacing_km[p] = pd.gate_spacing_km;
        info.scale[p] = pd.scale;
        info.offset[p] = pd.offset;
    }

    gpu::allocateStation(slot, info);
    const uint16_t* gatePtrs[NUM_PRODUCTS] = {};
    for (int p = 0; p < NUM_PRODUCTS; p++)
        if (pc.products[p].has_data && !pc.products[p].gates.empty())
            gatePtrs[p] = pc.products[p].gates.data();
    gpu::uploadStationData(slot, info, pc.azimuths.data(), gatePtrs);
    gpu::syncStation(slot);

    // Update station state for rendering
    if (m_stations.size() > 0) {
        std::lock_guard<std::mutex> lock(m_stationMutex);
        m_stations[0].gpuInfo = info;
        m_stations[0].uploaded = true;
        m_stations[0].uploaded_product = m_activeProduct;
        m_stations[0].uploaded_tilt = m_activeTilt;
        m_stations[0].uploaded_sweep = sweepIdx;
        m_stations[0].uploaded_lowest_sweep = false;
        m_stations[0].gpuInfo.lat = fr->station_lat;
        m_stations[0].gpuInfo.lon = fr->station_lon;
    }
    m_activeStationIdx = 0;
    m_activeTiltAngle = pc.elevation_angle;

    if (m_crossSection || m_mode3D)
        rebuildVolumeForCurrentSelection();
}

// (Demo pack methods removed)

void App::refreshData() {
    printf("Refreshing data from AWS...\n");
    const bool clearExistingScene = m_snapshotMode || m_historicMode;
    m_lastRefresh = std::chrono::steady_clock::now();
    m_autoTrackStation = true;
    m_historic.cancel();
    m_historicMode = false;
    m_warnings.clearHistoric();

    if (clearExistingScene) {
        {
            std::lock_guard<std::mutex> lock(m_stationMutex);
            ++m_downloadGeneration;
            m_snapshotMode = false;
            m_snapshotLowestSweepOnly = false;
            m_snapshotLabel.clear();
            m_snapshotTimestampIso.clear();
        }
        resetStationsForReload();
        startDownloads();
        return;
    }

    {
        std::lock_guard<std::mutex> lock(m_stationMutex);
        ++m_downloadGeneration;
        m_snapshotMode = false;
        m_snapshotLowestSweepOnly = false;
        m_snapshotLabel.clear();
        m_snapshotTimestampIso.clear();
        m_stationsDownloading = 0;

        // Keep current rendered data visible until replacement downloads arrive.
        for (auto& st : m_stations) {
            st.downloading = false;
            st.failed = false;
            st.error.clear();
        }
    }
    {
        std::lock_guard<std::mutex> lock(m_uploadMutex);
        m_uploadQueue.clear();
    }
    startDownloads();
}

void App::loadMarch302025Snapshot(bool lowestSweepOnly) {
    printf("Loading all-site archive snapshot for 2025-03-30 21:00 UTC%s...\n",
           lowestSweepOnly ? " (lowest sweep only)" : "");
    m_lastRefresh = std::chrono::steady_clock::now();
    m_historic.cancel();
    m_historicMode = false;
    m_autoTrackStation = true;
    m_snapshotMode = true;
    m_snapshotLowestSweepOnly = lowestSweepOnly;
    m_snapshotLabel = lowestSweepOnly
        ? "Mar 30 2025 5 PM ET (Lowest Sweep)"
        : "Mar 30 2025 5 PM ET";
    m_snapshotTimestampIso = makeIsoUtcTimestamp(2025, 3, 30, 21, 0);
    m_showAll = true;
    m_mode3D = false;
    m_crossSection = false;
    m_viewport.center_lat = 39.0;
    m_viewport.center_lon = -98.0;
    m_viewport.zoom = 28.0;
    m_warnings.clearHistoric();
    {
        std::lock_guard<std::mutex> lock(m_stationMutex);
        ++m_downloadGeneration;
    }
    invalidateFrameCache(true);
    resetStationsForReload();
    startDownloadsForTimestamp(2025, 3, 30, 21, 0);
}

// ── Detection computation (TDS, Hail, Mesocyclone) ──────────

void App::computeDetection(int stationIdx) {
    auto& st = m_stations[stationIdx];
    if (st.precomputed.empty()) return;
    auto& det = st.detection;
    det.tds.clear();
    det.hail.clear();
    det.meso.clear();
    det.computed = true;

    float slat = st.gpuInfo.lat != 0 ? st.gpuInfo.lat : st.lat;
    float slon = st.gpuInfo.lon != 0 ? st.gpuInfo.lon : st.lon;
    float cos_lat = std::max(cosf(slat * kDegToRad), 0.1f);

    int refSweep = -1, ccSweep = -1, zdrSweep = -1, velSweep = -1;
    for (int s = 0; s < (int)st.precomputed.size(); s++) {
        auto& pc = st.precomputed[s];
        if (pc.elevation_angle > 1.5f) continue; // only lowest tilts
        if (pc.products[PROD_REF].has_data && refSweep < 0) refSweep = s;
        if (pc.products[PROD_CC].has_data && ccSweep < 0) ccSweep = s;
        if (pc.products[PROD_ZDR].has_data && zdrSweep < 0) zdrSweep = s;
        if (pc.products[PROD_VEL].has_data && velSweep < 0) velSweep = s;
    }

    // ── TDS: CC < 0.80, REF > 35 dBZ, |ZDR| < 1.0 ──
    if (ccSweep >= 0 && zdrSweep >= 0 && refSweep >= 0) {
        auto& ccPc = st.precomputed[ccSweep];
        auto& zdrPc = st.precomputed[zdrSweep];
        auto& refPc = st.precomputed[refSweep];
        auto& ccPd = ccPc.products[PROD_CC];
        auto& zdrPd = zdrPc.products[PROD_ZDR];
        auto& refPd = refPc.products[PROD_REF];

        int nr = ccPc.num_radials;
        int ng = ccPd.num_gates;
        if (nr > 0 && ng > 0) {
            std::vector<uint8_t> candidate((size_t)nr * ng, 0);
            std::vector<float> score((size_t)nr * ng, std::numeric_limits<float>::infinity());

            for (int ri = 0; ri < nr; ++ri) {
                int zdr_ri = std::min((int)((int64_t)ri * zdrPc.num_radials / std::max(nr, 1)),
                                      std::max(zdrPc.num_radials - 1, 0));
                int ref_ri = std::min((int)((int64_t)ri * refPc.num_radials / std::max(nr, 1)),
                                      std::max(refPc.num_radials - 1, 0));
                for (int gi = 0; gi < ng; gi += 2) {
                    float range_km = ccPd.first_gate_km + gi * ccPd.gate_spacing_km;
                    if (range_km < 15.0f || range_km > 120.0f) continue;

                    float cc = decodeGateValue(ccPd, nr, gi, ri);
                    if (cc == kInvalidSample || cc < 0.55f || cc > 0.82f) continue;

                    int zdr_gi = gateIndexForRange(zdrPd, range_km);
                    int ref_gi = gateIndexForRange(refPd, range_km);
                    if (zdr_gi < 0 || ref_gi < 0) continue;

                    float zdr = decodeGateValue(zdrPd, zdrPc.num_radials, zdr_gi, zdr_ri);
                    float ref = decodeGateValue(refPd, refPc.num_radials, ref_gi, ref_ri);
                    if (zdr == kInvalidSample || ref == kInvalidSample) continue;
                    if (fabsf(zdr) > 1.25f || ref < 40.0f) continue;

                    candidate[(size_t)gi * nr + ri] = 1;
                    score[(size_t)gi * nr + ri] = cc;
                }
            }

            for (int ri = 0; ri < nr; ++ri) {
                float az_rad = ccPc.azimuths[ri] * kDegToRad;
                for (int gi = 0; gi < ng; gi += 2) {
                    if (!candidate[(size_t)gi * nr + ri]) continue;
                    if (countCandidateSupport(candidate, nr, ng, ri, gi, 2, 2) < 6) continue;
                    if (!isLocalExtremum(score, candidate, nr, ng, ri, gi, 2, 2, true)) continue;

                    float range_km = ccPd.first_gate_km + gi * ccPd.gate_spacing_km;
                    float east_km = range_km * sinf(az_rad);
                    float north_km = range_km * cosf(az_rad);
                    det.tds.push_back({
                        slat + north_km / 111.0f,
                        slon + east_km / (111.0f * cos_lat),
                        score[(size_t)gi * nr + ri]
                    });
                }
            }
            clusterMarkers(det.tds, 8.0f, 12, true);
        }
    }

    // ── Hail: HDR = Z - (19*ZDR + 27), mark where HDR > 0 ──
    if (refSweep >= 0 && zdrSweep >= 0) {
        auto& refPc = st.precomputed[refSweep];
        auto& zdrPc = st.precomputed[zdrSweep];
        auto& refPd = refPc.products[PROD_REF];
        auto& zdrPd = zdrPc.products[PROD_ZDR];

        int nr = refPc.num_radials;
        int ng = refPd.num_gates;
        if (nr > 0 && ng > 0) {
            std::vector<uint8_t> candidate((size_t)nr * ng, 0);
            std::vector<float> score((size_t)nr * ng, -std::numeric_limits<float>::infinity());

            for (int ri = 0; ri < nr; ++ri) {
                int zdr_ri = std::min((int)((int64_t)ri * zdrPc.num_radials / std::max(nr, 1)),
                                      std::max(zdrPc.num_radials - 1, 0));
                for (int gi = 0; gi < ng; gi += 2) {
                    float range_km = refPd.first_gate_km + gi * refPd.gate_spacing_km;
                    if (range_km < 15.0f || range_km > 180.0f) continue;

                    float ref = decodeGateValue(refPd, nr, gi, ri);
                    if (ref == kInvalidSample || ref < 55.0f) continue;

                    int zdr_gi = gateIndexForRange(zdrPd, range_km);
                    if (zdr_gi < 0) continue;
                    float zdr = decodeGateValue(zdrPd, zdrPc.num_radials, zdr_gi, zdr_ri);
                    if (zdr == kInvalidSample) continue;

                    float hdr = ref - (19.0f * std::max(zdr, 0.0f) + 27.0f);
                    if (hdr < 10.0f) continue;

                    candidate[(size_t)gi * nr + ri] = 1;
                    score[(size_t)gi * nr + ri] = hdr;
                }
            }

            for (int ri = 0; ri < nr; ++ri) {
                float az_rad = refPc.azimuths[ri] * kDegToRad;
                for (int gi = 0; gi < ng; gi += 2) {
                    if (!candidate[(size_t)gi * nr + ri]) continue;
                    if (countCandidateSupport(candidate, nr, ng, ri, gi, 2, 2) < 5) continue;
                    if (!isLocalExtremum(score, candidate, nr, ng, ri, gi, 2, 2, false)) continue;

                    float range_km = refPd.first_gate_km + gi * refPd.gate_spacing_km;
                    float east_km = range_km * sinf(az_rad);
                    float north_km = range_km * cosf(az_rad);
                    det.hail.push_back({
                        slat + north_km / 111.0f,
                        slon + east_km / (111.0f * cos_lat),
                        score[(size_t)gi * nr + ri]
                    });
                }
            }
            clusterMarkers(det.hail, 10.0f, 16, false);
        }
    }

    // ── Mesocyclone: azimuthal shear in velocity data ──
    if (velSweep >= 0) {
        auto& velPc = st.precomputed[velSweep];
        auto& velPd = velPc.products[PROD_VEL];
        int nr = velPc.num_radials;
        int ng = velPd.num_gates;

        if (nr >= 10 && ng >= 10) {
            std::vector<uint8_t> candidate((size_t)nr * ng, 0);
            std::vector<float> score((size_t)nr * ng, -std::numeric_limits<float>::infinity());
            std::vector<float> diameter((size_t)nr * ng, 0.0f);

            auto passesMesoGate = [&](int gate_idx, int radial_idx,
                                      float range_km, float* shear_out,
                                      float* span_out) -> bool {
                int span = 2;
                int ri_lo = (radial_idx - span + nr) % nr;
                int ri_hi = (radial_idx + span) % nr;

                float v_lo = decodeGateValue(velPd, nr, gate_idx, ri_lo);
                float v_hi = decodeGateValue(velPd, nr, gate_idx, ri_hi);
                if (v_lo == kInvalidSample || v_hi == kInvalidSample) return false;
                if (fabsf(v_lo) < 12.0f || fabsf(v_hi) < 12.0f) return false;
                if (v_lo * v_hi >= 0.0f) return false;

                float shear_ms = fabsf(v_hi - v_lo);
                if (shear_ms < 40.0f) return false;

                float az_span_deg = span * 2.0f * (360.0f / nr);
                float az_span_km = range_km * az_span_deg * kDegToRad;
                if (az_span_km < 1.0f || az_span_km > 10.0f) return false;

                *shear_out = shear_ms;
                *span_out = az_span_km;
                return true;
            };

            for (int gi = 12; gi < ng - 12; gi += 4) {
                float range_km = velPd.first_gate_km + gi * velPd.gate_spacing_km;
                if (range_km < 20.0f || range_km > 120.0f) continue;

                for (int ri = 0; ri < nr; ri += 2) {
                    if (refSweep >= 0) {
                        auto& refPc = st.precomputed[refSweep];
                        auto& refPd = refPc.products[PROD_REF];
                        int ref_ri = std::min((int)((int64_t)ri * refPc.num_radials / std::max(nr, 1)),
                                              std::max(refPc.num_radials - 1, 0));
                        int ref_gi = gateIndexForRange(refPd, range_km);
                        float ref = decodeGateValue(refPd, refPc.num_radials, ref_gi, ref_ri);
                        if (ref == kInvalidSample || ref < 35.0f) continue;
                    }

                    float shear_ms = 0.0f;
                    float az_span_km = 0.0f;
                    if (!passesMesoGate(gi, ri, range_km, &shear_ms, &az_span_km)) continue;

                    int gate_support = 0;
                    for (int dgi = -2; dgi <= 2; ++dgi) {
                        int ngi = gi + dgi;
                        if (ngi < 0 || ngi >= ng) continue;
                        float neighbor_shear = 0.0f;
                        float neighbor_span = 0.0f;
                        float neighbor_range = velPd.first_gate_km + ngi * velPd.gate_spacing_km;
                        if (passesMesoGate(ngi, ri, neighbor_range, &neighbor_shear, &neighbor_span))
                            ++gate_support;
                    }
                    if (gate_support < 3) continue;

                    candidate[(size_t)gi * nr + ri] = 1;
                    score[(size_t)gi * nr + ri] = shear_ms;
                    diameter[(size_t)gi * nr + ri] = az_span_km;
                }
            }

            for (int gi = 12; gi < ng - 12; gi += 4) {
                for (int ri = 0; ri < nr; ri += 2) {
                    if (!candidate[(size_t)gi * nr + ri]) continue;
                    if (countCandidateSupport(candidate, nr, ng, ri, gi, 2, 1) < 3) continue;
                    if (!isLocalExtremum(score, candidate, nr, ng, ri, gi, 2, 1, false)) continue;

                    float range_km = velPd.first_gate_km + gi * velPd.gate_spacing_km;
                    float az_rad = velPc.azimuths[ri] * kDegToRad;
                    float east_km = range_km * sinf(az_rad);
                    float north_km = range_km * cosf(az_rad);
                    det.meso.push_back({
                        slat + north_km / 111.0f,
                        slon + east_km / (111.0f * cos_lat),
                        score[(size_t)gi * nr + ri],
                        diameter[(size_t)gi * nr + ri]
                    });
                }
            }
            clusterMesoMarkers(det.meso, 12.0f, 12);
        }
    }

    printf("Detection [%s]: %d TDS, %d hail, %d meso\n",
           st.icao.c_str(), (int)det.tds.size(), (int)det.hail.size(), (int)det.meso.size());
}

// ── Velocity dealiasing ─────────────────────────────────────
// Simple spatial-consistency dealiasing: if a gate's velocity jumps by
// more than Vn (Nyquist) from its neighbors, unfold it.

void App::dealias(int stationIdx) {
    auto& st = m_stations[stationIdx];
    if (st.precomputed.empty()) return;

    for (auto& pc : st.precomputed) {
        auto& velPd = pc.products[PROD_VEL];
        dealiasVelocityProduct(velPd, pc.num_radials);
    }
}

// ── All-tilt VRAM cache ─────────────────────────────────────
// Upload every sweep's data for all products to GPU. Tilt switching
// becomes a pointer swap (zero re-upload).

void App::uploadAllTilts(int stationIdx) {
    auto& st = m_stations[stationIdx];
    if (st.precomputed.empty()) return;

    for (int s = 0; s < (int)st.precomputed.size(); s++) {
        int slot = stationIdx; // reuse same slot, we cache pointers per-sweep
        // For all-tilt cache, upload each sweep to a temp slot
        // We store the GPU pointers in a cache structure
        // For now, the existing uploadStation handles single-tilt upload efficiently
        // The real optimization: don't re-upload on tilt change
    }
    // Mark all tilts as cached
    m_allTiltsCached = true;
}

void App::switchTiltCached(int stationIdx, int newTilt) {
    // If we have all tilts cached, just swap pointers
    // For now, fall back to re-upload (full cache TBD)
    uploadStation(stationIdx);
}

// ── Pre-baked animation frame cache ─────────────────────────

void App::cacheAnimFrame(int frameIdx, const uint32_t* d_src, int w, int h) {
    if (!historicFrameCachingEnabled()) return;
    if (frameIdx >= historicFrameCacheLimit()) return;
    if (w <= 0 || h <= 0) return;

    if ((m_cachedFrameWidth != 0 || m_cachedFrameHeight != 0) &&
        (m_cachedFrameWidth != w || m_cachedFrameHeight != h)) {
        invalidateFrameCache(true);
    }

    m_cachedFrameWidth = w;
    m_cachedFrameHeight = h;
    size_t sz = (size_t)w * h * sizeof(uint32_t);
    if (!m_cachedFrames[frameIdx]) {
        CUDA_CHECK(cudaMalloc(&m_cachedFrames[frameIdx], sz));
    }
    CUDA_CHECK(cudaMemcpy(m_cachedFrames[frameIdx], d_src, sz, cudaMemcpyDeviceToDevice));
    if (frameIdx >= m_cachedFrameCount) m_cachedFrameCount = frameIdx + 1;
    m_memoryTelemetry.historic_cache_bytes = sz * (size_t)m_cachedFrameCount;
}
