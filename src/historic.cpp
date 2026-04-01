#include "historic.h"
#include "net/downloader.h"
#include "net/aws_nexrad.h"
#include "nexrad/products.h"
#include "nexrad/stations.h"
#include <algorithm>
#include <cctype>
#include <thread>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <ctime>
#include <memory>

// ── Download and parse all frames for a historic event ──────

namespace {

std::string filenameFromKey(const std::string& key);

bool extractFilenameTime(const std::string& fname, int& hh, int& mm, int& ss) {
    int year = 0, month = 0, day = 0;
    return extractRadarFileDateTime(fname, year, month, day, hh, mm, ss);
}

bool extractKeyDateTime(const std::string& key, int& year, int& month, int& day,
                        int& hh, int& mm, int& ss) {
    return extractRadarFileDateTime(key, year, month, day, hh, mm, ss);
}

int64_t makeUtcEpoch(int year, int month, int day, int hh, int mm, int ss) {
    std::tm tm = {};
    tm.tm_year = year - 1900;
    tm.tm_mon = month - 1;
    tm.tm_mday = day;
    tm.tm_hour = hh;
    tm.tm_min = mm;
    tm.tm_sec = ss;
#ifdef _WIN32
    return static_cast<int64_t>(_mkgmtime(&tm));
#else
    return static_cast<int64_t>(timegm(&tm));
#endif
}

bool inHistoricWindow(const HistoricEvent& ev, int hh, int mm) {
    const int timeMinutes = hh * 60 + mm;
    const int startMin = ev.start_hour * 60 + ev.start_min;
    const int endMin = ev.end_hour * 60 + ev.end_min;

    if (endMin < startMin)
        return timeMinutes >= startMin || timeMinutes <= endMin;
    return timeMinutes >= startMin && timeMinutes <= endMin;
}

void populateSweepData(const ParsedSweep& sweep, PrecomputedSweep& pc) {
    pc.elevation_angle = sweep.elevation_angle;
    pc.num_radials = (int)sweep.radials.size();
    if (pc.num_radials <= 0) return;

    pc.azimuths.resize(pc.num_radials);
    for (int r = 0; r < pc.num_radials; r++)
        pc.azimuths[r] = sweep.radials[r].azimuth;

    for (const auto& radial : sweep.radials) {
        for (const auto& moment : radial.moments) {
            const int p = moment.product_index;
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

        const int ng = pd.num_gates;
        const int nr = pc.num_radials;
        pd.gates.assign((size_t)ng * nr, 0);

        for (int r = 0; r < nr; r++) {
            for (const auto& mom : sweep.radials[r].moments) {
                if (mom.product_index != p) continue;
                const int gc = std::min((int)mom.gates.size(), ng);
                for (int g = 0; g < gc; g++)
                    pd.gates[(size_t)g * nr + r] = mom.gates[g];
                break;
            }
        }
    }
}

std::shared_ptr<RadarFrame> buildFrame(const ParsedRadarData& parsed, const std::string& key) {
    auto frame = std::make_shared<RadarFrame>();
    frame->filename = filenameFromKey(key);
    frame->station_lat = parsed.station_lat;
    frame->station_lon = parsed.station_lon;

    int year = 0, month = 0, day = 0;
    int hh = 0, mm = 0, ss = 0;
    if (extractKeyDateTime(key, year, month, day, hh, mm, ss)) {
        char ts[16];
        std::snprintf(ts, sizeof(ts), "%02d:%02d:%02d", hh, mm, ss);
        frame->timestamp = ts;

        char iso[32];
        std::snprintf(iso, sizeof(iso), "%04d-%02d-%02dT%02d:%02d:%02dZ",
                      year, month, day, hh, mm, ss);
        frame->valid_time_iso = iso;
        frame->valid_time_epoch = makeUtcEpoch(year, month, day, hh, mm, ss);
    }

    frame->sweeps.resize(parsed.sweeps.size());
    for (int si = 0; si < (int)parsed.sweeps.size(); si++)
        populateSweepData(parsed.sweeps[si], frame->sweeps[si]);

    frame->ready = true;
    return frame;
}

std::string filenameFromKey(const std::string& key) {
    return radarFilenameFromKey(key);
}

} // namespace

HistoricLoader::~HistoricLoader() {
    cancel();
}

std::string HistoricLoader::currentLabel() const {
    std::lock_guard<std::mutex> lock(m_metaMutex);
    return m_currentLabel;
}

std::string HistoricLoader::currentStation() const {
    std::lock_guard<std::mutex> lock(m_metaMutex);
    return m_currentStation;
}

std::string HistoricLoader::lastError() const {
    std::lock_guard<std::mutex> lock(m_metaMutex);
    return m_lastError;
}

void HistoricLoader::joinWorker() {
    if (m_worker.joinable() && m_worker.get_id() != std::this_thread::get_id())
        m_worker.join();
}

void HistoricLoader::cancel() {
    m_cancel = true;

    std::shared_ptr<Downloader> downloader;
    {
        std::lock_guard<std::mutex> lock(m_workerMutex);
        downloader = m_downloader;
    }
    if (downloader) downloader->shutdown();

    joinWorker();
    m_loading = false;
    m_playing = false;
}

int HistoricLoader::numFrames() const {
    std::lock_guard<std::mutex> lock(m_framesMutex);
    return (int)m_frames.size();
}

const RadarFrame* HistoricLoader::frame(int idx) const {
    std::lock_guard<std::mutex> lock(m_framesMutex);
    if (idx < 0 || idx >= (int)m_frames.size()) return nullptr;
    return m_frames[idx] ? m_frames[idx].get() : nullptr;
}

bool HistoricLoader::startLoad(const std::string& label,
                               const std::string& station,
                               int year, int month, int day,
                               int start_hour, int start_min,
                               int end_hour, int end_min,
                               const HistoricEvent* eventRef,
                               ProgressCallback cb) {
    cancel();

    m_event = eventRef;
    m_loading = true;
    m_loaded = false;
    m_cancel = false;
    {
        std::lock_guard<std::mutex> lock(m_metaMutex);
        m_currentLabel = label;
        m_currentStation = station;
        m_lastError.clear();
    }
    {
        std::lock_guard<std::mutex> lock(m_framesMutex);
        m_frames.clear();
    }
    m_currentFrame = 0;
    m_downloadedFrames = 0;
    m_totalFrames = 0;
    m_playing = false;
    m_accumulator = 0.0f;

    m_worker = std::thread([this, cb, label, station, year, month, day,
                            start_hour, start_min, end_hour, end_min]() {
        printf("Loading archive range: %s (%s %04d-%02d-%02d %02d:%02d-%02d:%02d UTC)\n",
               label.c_str(), station.c_str(), year, month, day,
               start_hour, start_min, end_hour, end_min);

        std::shared_ptr<Downloader> downloader;
        auto finish = [this, &downloader](bool loaded) {
            {
                std::lock_guard<std::mutex> lock(m_workerMutex);
                if (m_downloader == downloader) m_downloader.reset();
            }
            m_loaded = loaded && !m_cancel.load();
            m_loading = false;
        };
        auto setError = [this](std::string error) {
            std::lock_guard<std::mutex> lock(m_metaMutex);
            m_lastError = std::move(error);
        };

        int stationIdx = -1;
        for (int i = 0; i < NUM_NEXRAD_STATIONS; i++) {
            if (station == NEXRAD_STATIONS[i].icao) {
                stationIdx = i;
                break;
            }
        }
        StationInfo stationInfo = {};
        if (stationIdx >= 0) {
            stationInfo = NEXRAD_STATIONS[stationIdx];
        } else {
            stationInfo.icao = station.c_str();
            stationInfo.name = station.c_str();
            stationInfo.state = "";
            stationInfo.feed = RadarFeedKind::AwsS3DatePartitioned;
            stationInfo.feed_code = station.c_str();
        }

        auto fetchList = [&](int y, int m, int d, std::vector<NexradFile>& outFiles) -> bool {
            auto listResult = Downloader::httpGet(radarDataHost(stationInfo),
                                                  buildRadarListRequest(stationInfo, y, m, d, {}));
            if (!listResult.success || listResult.data.empty())
                return false;
            outFiles = parseRadarListResponse(stationInfo, listResult.data);
            return !outFiles.empty();
        };

        std::vector<NexradFile> files;
        if (!fetchList(year, month, day, files)) {
            printf("Failed to list files for %s\n", station.c_str());
            setError("Archive listing failed");
            finish(false);
            return;
        }

        auto makeUtcEpoch = [](int y, int mo, int d, int hh, int mi, int ss) -> int64_t {
            std::tm tm = {};
            tm.tm_year = y - 1900;
            tm.tm_mon = mo - 1;
            tm.tm_mday = d;
            tm.tm_hour = hh;
            tm.tm_min = mi;
            tm.tm_sec = ss;
#ifdef _WIN32
            return static_cast<int64_t>(_mkgmtime(&tm));
#else
            return static_cast<int64_t>(timegm(&tm));
#endif
        };

        const int64_t startEpoch = makeUtcEpoch(year, month, day, start_hour, start_min, 0);
        int endYear = year;
        int endMonth = month;
        int endDay = day;
        if (end_hour < start_hour || (end_hour == start_hour && end_min < start_min))
            shiftDate(endYear, endMonth, endDay, 1);
        const int64_t endEpoch = makeUtcEpoch(endYear, endMonth, endDay, end_hour, end_min, 59);

        // Also check next day for overnight date-partitioned feeds.
        if (radarFeedUsesDatePartitionedListing(stationInfo) &&
            (endYear != year || endMonth != month || endDay != day)) {
            std::vector<NexradFile> files2;
            if (fetchList(endYear, endMonth, endDay, files2))
                files.insert(files.end(), files2.begin(), files2.end());
        }

        std::vector<NexradFile> filtered;
        filtered.reserve(files.size());
        for (const auto& f : files) {
            int fy = 0, fm = 0, fd = 0;
            int hh = 0, mm = 0, ss = 0;
            if (!extractKeyDateTime(f.key, fy, fm, fd, hh, mm, ss))
                continue;
            const int64_t epoch = makeUtcEpoch(fy, fm, fd, hh, mm, ss);
            if (epoch >= startEpoch && epoch <= endEpoch)
                filtered.push_back(f);
        }

        if (filtered.empty()) {
            printf("No files found in time range\n");
            setError("No Level 2 files found in the requested time range");
            finish(false);
            return;
        }

        // Sort by key (chronological)
        std::sort(filtered.begin(), filtered.end(),
                  [](const NexradFile& a, const NexradFile& b) { return a.key < b.key; });

        m_totalFrames = (int)filtered.size();
        {
            std::lock_guard<std::mutex> lock(m_framesMutex);
            m_frames.assign((size_t)m_totalFrames.load(), nullptr);
        }
        printf("Found %d frames to download\n", m_totalFrames.load());

        // Download and parse each frame (parallel with 8 threads)
        downloader = std::make_shared<Downloader>(8);
        {
            std::lock_guard<std::mutex> lock(m_workerMutex);
            m_downloader = downloader;
        }
        for (int i = 0; i < m_totalFrames.load(); i++) {
            if (m_cancel.load()) break;

            auto& nf = filtered[i];
            int idx = i;

            downloader->queueDownload(nf.key, radarDataHost(stationInfo),
                                      buildRadarDownloadRequest(stationInfo, nf.key),
                [this, idx, cb](const std::string& id, DownloadResult result) {
                    if (m_cancel.load()) return;

                    if (!result.success || result.data.empty()) {
                        int done = ++m_downloadedFrames;
                        if (cb) cb(done, m_totalFrames.load());
                        return;
                    }

                    // Parse
                    auto parsed = Level2Parser::parse(result.data);
                    if (parsed.sweeps.empty()) {
                        int done = ++m_downloadedFrames;
                        if (cb) cb(done, m_totalFrames.load());
                        return;
                    }

                    // Extract timestamp from key
                    auto frame = buildFrame(parsed, id);
                    {
                        std::lock_guard<std::mutex> lock(m_framesMutex);
                        if (idx >= 0 && idx < (int)m_frames.size()) {
                            m_frames[idx] = std::move(frame);
                        }
                    }

                    int done = ++m_downloadedFrames;
                    if (cb) cb(done, m_totalFrames.load());
                    printf("\rFrames: %d/%d", done, m_totalFrames.load());
                    fflush(stdout);
                }
            );
        }

        if (m_cancel.load()) {
            downloader->shutdown();
            finish(false);
            return;
        }

        downloader->waitAll();
        if (m_cancel.load()) {
            finish(false);
            return;
        }

        int readyFrames = 0;
        {
            std::lock_guard<std::mutex> lock(m_framesMutex);
            for (const auto& frame : m_frames)
                readyFrames += frame ? 1 : 0;
        }

        if (readyFrames <= 0) {
            setError("Archive download completed, but no valid frames were parsed");
            finish(false);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(m_metaMutex);
            m_lastError.clear();
        }
        printf("\nHistoric event loaded: %d frames ready\n", m_downloadedFrames.load());
        finish(true);
    });
    return true;
}

void HistoricLoader::loadEvent(int eventIdx, ProgressCallback cb) {
    if (eventIdx < 0 || eventIdx >= NUM_HISTORIC_EVENTS) return;
    const auto& ev = HISTORIC_EVENTS[eventIdx];
    startLoad(ev.name, ev.station, ev.year, ev.month, ev.day,
              ev.start_hour, ev.start_min, ev.end_hour, ev.end_min,
              &ev, std::move(cb));
}

bool HistoricLoader::loadRange(const std::string& label,
                               const std::string& station,
                               int year, int month, int day,
                               int start_hour, int start_min,
                               int end_hour, int end_min,
                               ProgressCallback cb) {
    return startLoad(label, station, year, month, day,
                     start_hour, start_min, end_hour, end_min,
                     nullptr, std::move(cb));
}

void HistoricLoader::update(float dt) {
    if (!m_playing || numFrames() <= 0) return;

    m_accumulator += dt;
    float frameDur = 1.0f / m_fps;
    while (m_accumulator >= frameDur) {
        m_accumulator -= frameDur;
        m_currentFrame++;
        // Skip frames that have not been published yet.
        while (m_currentFrame < numFrames() && frame(m_currentFrame) == nullptr)
            m_currentFrame++;
        if (m_currentFrame >= numFrames())
            m_currentFrame = 0; // loop
    }
}

// ── Helper: precompute one parsed file into a RadarFrame ────
static void precomputeFrame(RadarFrame& frame, ParsedRadarData& parsed, const std::string& fname) {
    auto built = buildFrame(parsed, fname);
    if (built) frame = *built;
}

// ── DemoPack Loader ─────────────────────────────────────────

// Extract time in minutes from midnight from a NEXRAD filename
static float extractTimeMinutes(const std::string& fname) {
    size_t us = fname.find('_');
    if (us == std::string::npos || us + 7 > fname.size()) return -1;
    std::string ts = fname.substr(us + 1, 6);
    if (ts.size() < 6) return -1;
    int hh = std::stoi(ts.substr(0, 2));
    int mm = std::stoi(ts.substr(2, 2));
    int ss = std::stoi(ts.substr(4, 2));
    return (float)(hh * 60 + mm) + ss / 60.0f;
}

// (Demo pack code removed)
#if 0
void DemoPackLoader_REMOVED_loadPack(int packIdx) {
    if (packIdx < 0 || packIdx >= NUM_DEMO_PACKS || m_loading) return;

    m_pack = &DEMO_PACKS[packIdx];
    m_loading = true;
    m_loaded = false;
    m_cancel = false;
    m_stationFrames.clear();
    m_downloadedFiles = 0;
    m_currentTime = (float)m_pack->start_hour * 60;
    m_playing = false;

    std::thread([this]() {
        const auto& pk = *m_pack;
        printf("Loading demo pack: %s (%d stations)\n", pk.name, pk.num_stations);

        // Set up per-station frame lists
        m_stationFrames.resize(pk.num_stations);
        for (int s = 0; s < pk.num_stations; s++) {
            m_stationFrames[s].station = pk.stations[s];
            // Find lat/lon from station list
            for (int j = 0; j < NUM_NEXRAD_STATIONS; j++) {
                if (m_stationFrames[s].station == NEXRAD_STATIONS[j].icao) {
                    m_stationFrames[s].lat = NEXRAD_STATIONS[j].lat;
                    m_stationFrames[s].lon = NEXRAD_STATIONS[j].lon;
                    m_stationFrames[s].station_global_idx = j;
                    break;
                }
            }
        }

        // List + download all files for each station
        Downloader dl(12);
        int startMin = pk.start_hour * 60;
        int endMin = pk.end_hour * 60;

        for (int s = 0; s < pk.num_stations; s++) {
            if (m_cancel) break;

            std::string station = pk.stations[s];
            std::string listPath = "/?list-type=2&prefix=" +
                std::to_string(pk.year) + "/" +
                (pk.month < 10 ? "0" : "") + std::to_string(pk.month) + "/" +
                (pk.day < 10 ? "0" : "") + std::to_string(pk.day) + "/" +
                station + "/&max-keys=1000";

            auto listResult = Downloader::httpGet(NEXRAD_HOST, listPath);
            if (!listResult.success) {
                printf("  %s: listing failed\n", station.c_str());
                continue;
            }

            std::string xml(listResult.data.begin(), listResult.data.end());
            auto files = parseS3ListResponse(xml);

            // Filter by time range
            std::vector<NexradFile> filtered;
            for (auto& f : files) {
                size_t us = f.key.rfind('/');
                std::string fname = (us != std::string::npos) ? f.key.substr(us + 1) : f.key;
                float tmin = extractTimeMinutes(fname);
                if (tmin >= 0 && tmin >= startMin && tmin <= endMin)
                    filtered.push_back(f);
            }

            printf("  %s: %d files in time range\n", station.c_str(), (int)filtered.size());
            m_totalFiles += (int)filtered.size();
            m_stationFrames[s].frames.resize(filtered.size());

            for (int fi = 0; fi < (int)filtered.size(); fi++) {
                if (m_cancel) break;
                int stIdx = s;
                int frameIdx = fi;
                auto& nf = filtered[fi];

                dl.queueDownload(nf.key, NEXRAD_HOST, "/" + nf.key,
                    [this, stIdx, frameIdx](const std::string& id, DownloadResult result) {
                        if (!result.success || result.data.empty()) {
                            m_downloadedFiles++;
                            return;
                        }
                        auto parsed = Level2Parser::parse(result.data);
                        if (parsed.sweeps.empty()) {
                            m_downloadedFiles++;
                            return;
                        }

                        size_t us = id.rfind('/');
                        std::string fname = (us != std::string::npos) ? id.substr(us + 1) : id;

                        auto& frame = m_stationFrames[stIdx].frames[frameIdx];
                        precomputeFrame(frame, parsed, fname);

                        int done = ++m_downloadedFiles;
                        if (done % 10 == 0)
                            printf("\r  Demo pack: %d/%d files", done, m_totalFiles);
                    }
                );
            }
        }

        dl.waitAll();
        printf("\nDemo pack loaded: %d files across %d stations\n",
               m_downloadedFiles.load(), pk.num_stations);
        m_loaded = true;
        m_loading = false;
    }).detach();
}

void DemoPackLoader::update(float dt) {
    if (!m_playing) return;
    m_currentTime += dt * m_speed;
    if (m_currentTime > timelineMax())
        m_currentTime = timelineMin(); // loop
}

const RadarFrame* DemoPackLoader::getFrameAtTime(int stationIdx, float timeMinutes) const {
    if (stationIdx < 0 || stationIdx >= (int)m_stationFrames.size()) return nullptr;
    auto& sf = m_stationFrames[stationIdx];

    const RadarFrame* best = nullptr;
    float bestDist = 1e9f;
    for (auto& f : sf.frames) {
        if (!f.ready) continue;
        float tmin = extractTimeMinutes(f.filename);
        if (tmin < 0) continue;
        float dist = fabsf(tmin - timeMinutes);
        if (dist < bestDist) {
            bestDist = dist;
            best = &f;
        }
    }
    return best;
}
#endif
