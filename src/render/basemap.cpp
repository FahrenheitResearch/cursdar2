#include "basemap.h"

#include "data/us_boundaries.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <objbase.h>
#include <wincodec.h>
#endif

namespace {

constexpr double kMercatorMaxLat = 85.05112878;
constexpr int kTileSize = 256;
constexpr int kTileSubdivisions = 4;
constexpr size_t kMaxTileCacheEntries = 320;
constexpr auto kRetryCooldown = std::chrono::seconds(12);

struct RasterSourceInfo {
    const char* service = "";
    const char* attribution = "";
};

double clampLat(double lat) {
    return std::max(-kMercatorMaxLat, std::min(kMercatorMaxLat, lat));
}

double lonToTileX(double lon, int z) {
    const double n = (double)(1 << z);
    return (lon + 180.0) / 360.0 * n;
}

double latToTileY(double lat, int z) {
    lat = clampLat(lat);
    const double latRad = lat * DEG_TO_RAD;
    const double n = (double)(1 << z);
    return (1.0 - std::asinh(std::tan(latRad)) / M_PI) * 0.5 * n;
}

double tileXToLon(double x, int z) {
    const double n = (double)(1 << z);
    return x / n * 360.0 - 180.0;
}

double tileYToLat(double y, int z) {
    const double n = M_PI - (2.0 * M_PI * y) / (double)(1 << z);
    return RAD_TO_DEG * std::atan(std::sinh(n));
}

int viewportTileZoom(const Viewport& vp) {
    const double worldPixels = std::max(vp.zoom * 360.0, 256.0);
    const double zoom = std::log2(worldPixels / 256.0);
    return std::clamp((int)std::lround(zoom), 0, 16);
}

ImVec2 latLonToScreen(const Viewport& vp, ImVec2 origin, double lat, double lon) {
    return ImVec2(
        origin.x + (float)((lon - vp.center_lon) * vp.zoom + vp.width * 0.5),
        origin.y + (float)((vp.center_lat - lat) * vp.zoom + vp.height * 0.5));
}

ImU32 scaleAlpha(ImU32 color, float alphaScale) {
    const int a = (color >> IM_COL32_A_SHIFT) & 0xFF;
    const int scaled = (int)std::lround(std::clamp(alphaScale, 0.0f, 1.0f) * (float)a);
    return (color & ~IM_COL32_A_MASK) | ((ImU32)scaled << IM_COL32_A_SHIFT);
}

const RasterSourceInfo* rasterSource(BasemapRenderer::RasterLayer layer) {
    static const RasterSourceInfo kRelief{
        "USGSShadedReliefOnly",
        "Basemap: USGS The National Map shaded relief"
    };
    static const RasterSourceInfo kImagery{
        "USGSImageryOnly",
        "Basemap: USGS The National Map imagery"
    };
    static const RasterSourceInfo kHybrid{
        "USGSImageryTopo",
        "Basemap: USGS The National Map imagery topo"
    };
    switch (layer) {
        case BasemapRenderer::RasterLayer::Relief: return &kRelief;
        case BasemapRenderer::RasterLayer::Imagery: return &kImagery;
        case BasemapRenderer::RasterLayer::ImageryTopo: return &kHybrid;
        case BasemapRenderer::RasterLayer::None:
        default: return nullptr;
    }
}

#ifdef _WIN32
struct WicThreadContext {
    HRESULT initHr = E_FAIL;
    IWICImagingFactory* factory = nullptr;

    WicThreadContext() {
        initHr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        const bool usable = SUCCEEDED(initHr) || initHr == RPC_E_CHANGED_MODE;
        if (!usable)
            return;
        CoCreateInstance(CLSID_WICImagingFactory, nullptr,
                         CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory));
    }

    ~WicThreadContext() {
        if (factory)
            factory->Release();
        if (SUCCEEDED(initHr))
            CoUninitialize();
    }
};

bool decodeImageRgba(const std::vector<uint8_t>& bytes,
                     std::vector<unsigned char>& rgba,
                     int& width, int& height,
                     std::string& error) {
    width = 0;
    height = 0;
    rgba.clear();
    if (bytes.empty()) {
        error = "empty image payload";
        return false;
    }

    thread_local WicThreadContext wic;
    if (!wic.factory) {
        error = "WIC factory unavailable";
        return false;
    }

    IWICStream* stream = nullptr;
    IWICBitmapDecoder* decoder = nullptr;
    IWICBitmapFrameDecode* frame = nullptr;
    IWICFormatConverter* converter = nullptr;

    auto fail = [&](const char* message, HRESULT hr) {
        error = std::string(message) + " (0x" + std::to_string((unsigned long)hr) + ")";
    };

    HRESULT hr = wic.factory->CreateStream(&stream);
    if (FAILED(hr)) {
        fail("CreateStream failed", hr);
        return false;
    }

    hr = stream->InitializeFromMemory(const_cast<BYTE*>(bytes.data()), (DWORD)bytes.size());
    if (FAILED(hr)) {
        fail("InitializeFromMemory failed", hr);
        stream->Release();
        return false;
    }

    hr = wic.factory->CreateDecoderFromStream(stream, nullptr, WICDecodeMetadataCacheOnLoad, &decoder);
    if (FAILED(hr)) {
        fail("CreateDecoderFromStream failed", hr);
        stream->Release();
        return false;
    }

    hr = decoder->GetFrame(0, &frame);
    if (FAILED(hr)) {
        fail("GetFrame failed", hr);
        decoder->Release();
        stream->Release();
        return false;
    }

    UINT w = 0;
    UINT h = 0;
    hr = frame->GetSize(&w, &h);
    if (FAILED(hr) || w == 0 || h == 0) {
        fail("GetSize failed", hr);
        frame->Release();
        decoder->Release();
        stream->Release();
        return false;
    }

    hr = wic.factory->CreateFormatConverter(&converter);
    if (FAILED(hr)) {
        fail("CreateFormatConverter failed", hr);
        frame->Release();
        decoder->Release();
        stream->Release();
        return false;
    }

    hr = converter->Initialize(frame, GUID_WICPixelFormat32bppRGBA,
                               WICBitmapDitherTypeNone, nullptr, 0.0,
                               WICBitmapPaletteTypeCustom);
    if (FAILED(hr)) {
        fail("FormatConverter::Initialize failed", hr);
        converter->Release();
        frame->Release();
        decoder->Release();
        stream->Release();
        return false;
    }

    const size_t byteCount = (size_t)w * (size_t)h * 4u;
    rgba.resize(byteCount);
    hr = converter->CopyPixels(nullptr, (UINT)(w * 4), (UINT)byteCount, rgba.data());
    if (FAILED(hr)) {
        fail("CopyPixels failed", hr);
        rgba.clear();
    } else {
        width = (int)w;
        height = (int)h;
    }

    converter->Release();
    frame->Release();
    decoder->Release();
    stream->Release();
    return SUCCEEDED(hr);
}
#else
bool decodeImageRgba(const std::vector<uint8_t>&,
                     std::vector<unsigned char>&,
                     int&,
                     int&,
                     std::string& error) {
    error = "raster basemap decode is only implemented on Windows in this build";
    return false;
}
#endif

void drawTextWithHalo(ImDrawList* drawList, ImVec2 pos, ImU32 halo, ImU32 color, const char* text) {
    drawList->AddText(ImVec2(pos.x + 1.0f, pos.y + 1.0f), halo, text);
    drawList->AddText(pos, color, text);
}

} // namespace

const char* basemapStyleLabel(BasemapStyle style) {
    switch (style) {
        case BasemapStyle::Relief: return "Relief";
        case BasemapStyle::OpsDark: return "Ops Dark";
        case BasemapStyle::Satellite: return "Satellite";
        case BasemapStyle::SatelliteHybrid: return "Satellite Hybrid";
        default: return "Unknown";
    }
}

BasemapRenderer::BasemapRenderer()
    : m_downloader(8) {
#ifdef _WIN32
    char localAppData[MAX_PATH] = "";
    DWORD len = GetEnvironmentVariableA("LOCALAPPDATA", localAppData, (DWORD)sizeof(localAppData));
    if (len > 0 && len < sizeof(localAppData)) {
        m_diskCacheDir = std::filesystem::path(localAppData) / "cursdar2" / "basemap_cache";
    } else {
        m_diskCacheDir = std::filesystem::temp_directory_path() / "cursdar2_basemap_cache";
    }
#else
    m_diskCacheDir = std::filesystem::temp_directory_path() / "cursdar2_basemap_cache";
#endif
    std::error_code ec;
    std::filesystem::create_directories(m_diskCacheDir, ec);
    updateAttribution();
}

BasemapRenderer::~BasemapRenderer() {
    m_downloader.shutdown();
}

size_t BasemapRenderer::TileKeyHash::operator()(const TileKey& key) const {
    size_t value = (size_t)key.layer;
    value = value * 1315423911u + (size_t)key.z;
    value = value * 1315423911u + (size_t)key.x;
    value = value * 1315423911u + (size_t)key.y;
    return value;
}

BasemapRenderer::RasterLayer BasemapRenderer::activeRasterLayer() const {
    switch (m_style) {
        case BasemapStyle::Relief: return RasterLayer::Relief;
        case BasemapStyle::Satellite: return RasterLayer::Imagery;
        case BasemapStyle::SatelliteHybrid: return RasterLayer::ImageryTopo;
        case BasemapStyle::OpsDark:
        default:
            return RasterLayer::None;
    }
}

BasemapRenderer::StylePalette BasemapRenderer::activePalette() const {
    switch (m_style) {
        case BasemapStyle::OpsDark:
            return {
                IM_COL32(5, 10, 20, 255),
                IM_COL32(10, 18, 30, 255),
                IM_COL32(20, 48, 70, 120),
                IM_COL32(0, 0, 0, 0),
                IM_COL32(86, 106, 130, 210),
                IM_COL32(130, 190, 255, 180),
                IM_COL32(210, 225, 255, 220),
                IM_COL32(3, 6, 14, 220),
                IM_COL32(36, 58, 84, 130)
            };
        case BasemapStyle::Satellite:
            return {
                IM_COL32(6, 9, 14, 255),
                IM_COL32(8, 12, 18, 255),
                IM_COL32(22, 48, 68, 105),
                IM_COL32(9, 18, 28, 92),
                IM_COL32(255, 255, 255, 140),
                IM_COL32(255, 255, 255, 160),
                IM_COL32(245, 248, 255, 235),
                IM_COL32(5, 5, 8, 210),
                IM_COL32(255, 255, 255, 48)
            };
        case BasemapStyle::SatelliteHybrid:
            return {
                IM_COL32(7, 10, 16, 255),
                IM_COL32(10, 14, 20, 255),
                IM_COL32(32, 66, 88, 115),
                IM_COL32(6, 15, 22, 72),
                IM_COL32(235, 242, 255, 160),
                IM_COL32(255, 255, 255, 170),
                IM_COL32(245, 250, 255, 235),
                IM_COL32(6, 6, 8, 220),
                IM_COL32(255, 255, 255, 42)
            };
        case BasemapStyle::Relief:
        default:
            return {
                IM_COL32(8, 13, 20, 255),
                IM_COL32(14, 24, 34, 255),
                IM_COL32(44, 88, 110, 110),
                IM_COL32(12, 22, 28, 94),
                IM_COL32(150, 176, 204, 190),
                IM_COL32(204, 226, 255, 170),
                IM_COL32(220, 232, 246, 220),
                IM_COL32(4, 8, 12, 220),
                IM_COL32(118, 148, 172, 74)
            };
    }
}

void BasemapRenderer::updateAttribution() {
    if (const RasterSourceInfo* source = rasterSource(activeRasterLayer())) {
        m_attribution = source->attribution;
    } else {
        m_attribution = "Basemap: custom vector ops style";
    }
}

void BasemapRenderer::setStyle(BasemapStyle style) {
    if (m_style == style)
        return;
    m_style = style;
    updateAttribution();
}

void BasemapRenderer::setRasterOpacity(float opacity) {
    m_rasterOpacity = std::clamp(opacity, 0.0f, 1.0f);
}

void BasemapRenderer::setOverlayOpacity(float opacity) {
    m_overlayOpacity = std::clamp(opacity, 0.0f, 1.0f);
}

void BasemapRenderer::touchVisibleTile(const TileKey& key) {
    std::lock_guard<std::mutex> cacheLock(m_cacheMutex);
    auto it = m_tiles.find(key);
    if (it == m_tiles.end())
        return;
    std::lock_guard<std::mutex> tileLock(it->second->mutex);
    it->second->lastTouch = std::chrono::steady_clock::now();
}

std::filesystem::path BasemapRenderer::tileCachePath(const TileKey& key) const {
    const RasterSourceInfo* source = rasterSource(key.layer);
    const std::string service = source ? source->service : "unknown";
    return m_diskCacheDir /
           service /
           std::to_string(key.z) /
           std::to_string(key.x) /
           (std::to_string(key.y) + ".tile");
}

bool BasemapRenderer::loadTileFromDiskCache(const TileKey& key, TileEntry& entry) {
    std::error_code ec;
    const std::filesystem::path path = tileCachePath(key);
    if (!std::filesystem::exists(path, ec))
        return false;

    std::ifstream in(path, std::ios::binary);
    if (!in)
        return false;

    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
    if (bytes.empty())
        return false;

    std::vector<unsigned char> rgba;
    int width = 0;
    int height = 0;
    std::string decodeError;
    if (!decodeImageRgba(bytes, rgba, width, height, decodeError))
        return false;

    entry.width = width;
    entry.height = height;
    entry.rgba = std::move(rgba);
    entry.state = TileState::Decoded;
    entry.error.clear();
    entry.failureCount = 0;
    entry.nextRetry = {};
    return true;
}

void BasemapRenderer::storeTileToDiskCache(const TileKey& key, const std::vector<uint8_t>& bytes) {
    if (bytes.empty())
        return;

    std::error_code ec;
    const std::filesystem::path path = tileCachePath(key);
    std::filesystem::create_directories(path.parent_path(), ec);

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
        return;
    out.write((const char*)bytes.data(), (std::streamsize)bytes.size());
}

void BasemapRenderer::queueTileRequest(const TileKey& key) {
    const RasterSourceInfo* source = rasterSource(key.layer);
    if (!source)
        return;

    std::shared_ptr<TileEntry> entry;
    {
        std::lock_guard<std::mutex> cacheLock(m_cacheMutex);
        auto& slot = m_tiles[key];
        if (!slot)
            slot = std::make_shared<TileEntry>();
        entry = slot;
    }

    {
        std::lock_guard<std::mutex> lock(entry->mutex);
        const auto now = std::chrono::steady_clock::now();
        if (entry->state == TileState::Queued || entry->state == TileState::Ready ||
            entry->state == TileState::Decoded) {
            entry->lastTouch = now;
            return;
        }
        if (loadTileFromDiskCache(key, *entry)) {
            entry->lastTouch = now;
            return;
        }
        if (entry->state == TileState::Failed) {
            if (now < entry->nextRetry)
                return;
        }
        entry->state = TileState::Queued;
        entry->error.clear();
        entry->lastTouch = now;
    }

    const std::string id = std::string(source->service) + "/" +
                           std::to_string(key.z) + "/" +
                           std::to_string(key.x) + "/" +
                           std::to_string(key.y);
    const std::string path = "/arcgis/rest/services/" + std::string(source->service) +
                             "/MapServer/tile/" +
                             std::to_string(key.z) + "/" +
                             std::to_string(key.y) + "/" +
                             std::to_string(key.x);

    m_downloader.queueDownload(id, "basemap.nationalmap.gov", path,
        [this, key, entry](const std::string&, DownloadResult result) {
            std::vector<unsigned char> rgba;
            int width = 0;
            int height = 0;
            std::string decodeError;

            if (!result.success) {
                std::lock_guard<std::mutex> lock(entry->mutex);
                entry->state = TileState::Failed;
                entry->failureCount++;
                entry->nextRetry = std::chrono::steady_clock::now() + kRetryCooldown;
                entry->error = result.error.empty()
                    ? ("HTTP " + std::to_string(result.status_code))
                    : result.error;
                return;
            }

            if (!decodeImageRgba(result.data, rgba, width, height, decodeError)) {
                std::lock_guard<std::mutex> lock(entry->mutex);
                entry->state = TileState::Failed;
                entry->failureCount++;
                entry->nextRetry = std::chrono::steady_clock::now() + kRetryCooldown;
                entry->error = decodeError;
                return;
            }

            storeTileToDiskCache(key, result.data);

            std::lock_guard<std::mutex> lock(entry->mutex);
            entry->width = width;
            entry->height = height;
            entry->rgba = std::move(rgba);
            entry->state = TileState::Decoded;
            entry->failureCount = 0;
            entry->nextRetry = {};
            entry->error.clear();
        });
}

void BasemapRenderer::requestVisibleTiles(const Viewport& vp) {
    const RasterLayer layer = activeRasterLayer();
    if (layer == RasterLayer::None)
        return;

    const int z = viewportTileZoom(vp);
    const double minLon = vp.center_lon - vp.halfExtentLon();
    const double maxLon = vp.center_lon + vp.halfExtentLon();
    const double minLat = vp.center_lat - vp.halfExtentLat();
    const double maxLat = vp.center_lat + vp.halfExtentLat();

    const int tilesPerAxis = 1 << z;
    int x0 = (int)std::floor(lonToTileX(minLon, z)) - 1;
    int x1 = (int)std::floor(lonToTileX(maxLon, z)) + 1;
    int y0 = (int)std::floor(latToTileY(maxLat, z)) - 1;
    int y1 = (int)std::floor(latToTileY(minLat, z)) + 1;
    y0 = std::max(0, y0);
    y1 = std::min(tilesPerAxis - 1, y1);

    for (int ty = y0; ty <= y1; ty++) {
        for (int tx = x0; tx <= x1; tx++) {
            int wrappedX = tx % tilesPerAxis;
            if (wrappedX < 0)
                wrappedX += tilesPerAxis;
            TileKey key{layer, z, wrappedX, ty};
            touchVisibleTile(key);
            queueTileRequest(key);
        }
    }
}

void BasemapRenderer::uploadDecodedTiles() {
    std::vector<std::shared_ptr<TileEntry>> pendingUploads;
    {
        std::lock_guard<std::mutex> cacheLock(m_cacheMutex);
        pendingUploads.reserve(m_tiles.size());
        for (auto& kv : m_tiles)
            pendingUploads.push_back(kv.second);
    }

    for (auto& entry : pendingUploads) {
        std::lock_guard<std::mutex> lock(entry->mutex);
        if (entry->state != TileState::Decoded || entry->rgba.empty())
            continue;
        if (entry->texture.update(entry->width, entry->height, entry->rgba.data())) {
            entry->rgba.clear();
            entry->state = TileState::Ready;
        } else {
            entry->state = TileState::Failed;
            entry->error = "GL texture upload failed";
        }
    }
}

void BasemapRenderer::trimCache() {
    std::vector<std::pair<TileKey, std::chrono::steady_clock::time_point>> keys;
    {
        std::lock_guard<std::mutex> cacheLock(m_cacheMutex);
        if (m_tiles.size() <= kMaxTileCacheEntries)
            return;
        keys.reserve(m_tiles.size());
        for (const auto& kv : m_tiles) {
            std::lock_guard<std::mutex> tileLock(kv.second->mutex);
            keys.emplace_back(kv.first, kv.second->lastTouch);
        }
    }

    std::sort(keys.begin(), keys.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    const size_t removeCount = keys.size() - kMaxTileCacheEntries;
    std::lock_guard<std::mutex> cacheLock(m_cacheMutex);
    for (size_t i = 0; i < removeCount; i++)
        m_tiles.erase(keys[i].first);
}

void BasemapRenderer::update(const Viewport& vp) {
    if (activeRasterLayer() == RasterLayer::None) {
        m_status.clear();
        return;
    }
    requestVisibleTiles(vp);
    uploadDecodedTiles();
    trimCache();

    int failedCount = 0;
    {
        std::lock_guard<std::mutex> cacheLock(m_cacheMutex);
        for (const auto& kv : m_tiles) {
            std::lock_guard<std::mutex> tileLock(kv.second->mutex);
            failedCount += (kv.second->state == TileState::Failed) ? 1 : 0;
        }
    }

    if (m_downloader.pending() > 0) {
        m_status = "Fetching basemap tiles...";
        if (failedCount > 0)
            m_status += " " + std::to_string(failedCount) + " retrying.";
    } else if (failedCount > 0) {
        m_status = std::to_string(failedCount) + " basemap tile retries pending";
    } else {
        m_status.clear();
    }
}

void BasemapRenderer::drawGradientBackdrop(ImDrawList* drawList, const Viewport& vp,
                                           ImVec2 origin, const StylePalette& palette) {
    const ImVec2 p0 = origin;
    const ImVec2 p1(origin.x + (float)vp.width, origin.y + (float)vp.height);
    drawList->AddRectFilledMultiColor(
        p0, p1,
        palette.backgroundTop, palette.backgroundTop,
        palette.backgroundBottom, palette.backgroundBottom);

    const float accentW = (float)vp.width * 0.48f;
    const float accentH = (float)vp.height * 0.42f;
    drawList->AddRectFilledMultiColor(
        ImVec2(origin.x, origin.y),
        ImVec2(origin.x + accentW, origin.y + accentH),
        palette.backgroundAccent, 0,
        0, 0);
}

void BasemapRenderer::drawRasterTiles(ImDrawList* drawList, const Viewport& vp, ImVec2 origin) {
    const RasterLayer layer = activeRasterLayer();
    if (layer == RasterLayer::None || m_rasterOpacity <= 0.01f)
        return;

    const int z = viewportTileZoom(vp);
    const int tilesPerAxis = 1 << z;
    const double minLon = vp.center_lon - vp.halfExtentLon();
    const double maxLon = vp.center_lon + vp.halfExtentLon();
    const double minLat = vp.center_lat - vp.halfExtentLat();
    const double maxLat = vp.center_lat + vp.halfExtentLat();
    int x0 = (int)std::floor(lonToTileX(minLon, z)) - 1;
    int x1 = (int)std::floor(lonToTileX(maxLon, z)) + 1;
    int y0 = (int)std::floor(latToTileY(maxLat, z)) - 1;
    int y1 = (int)std::floor(latToTileY(minLat, z)) + 1;
    y0 = std::max(0, y0);
    y1 = std::min(tilesPerAxis - 1, y1);

    const ImU32 tint = IM_COL32(255, 255, 255,
        (int)std::lround(std::clamp(m_rasterOpacity, 0.0f, 1.0f) * 255.0f));

    auto findReadyTile = [&](int targetZ, int targetX, int targetY,
                             std::shared_ptr<TileEntry>& outEntry,
                             float& outU0, float& outV0, float& outU1, float& outV1) -> bool {
        std::lock_guard<std::mutex> cacheLock(m_cacheMutex);
        for (int candidateZ = targetZ; candidateZ >= 0; --candidateZ) {
            const int zoomDelta = targetZ - candidateZ;
            const int candidateTilesPerAxis = 1 << candidateZ;
            int candidateX = targetX >> zoomDelta;
            int candidateY = targetY >> zoomDelta;
            candidateX %= candidateTilesPerAxis;
            if (candidateX < 0)
                candidateX += candidateTilesPerAxis;
            if (candidateY < 0 || candidateY >= candidateTilesPerAxis)
                continue;

            TileKey candidateKey{layer, candidateZ, candidateX, candidateY};
            auto it = m_tiles.find(candidateKey);
            if (it == m_tiles.end())
                continue;

            std::lock_guard<std::mutex> tileLock(it->second->mutex);
            if (it->second->state != TileState::Ready || it->second->texture.textureId() == 0)
                continue;

            outEntry = it->second;
            const float invScale = 1.0f / float(1 << zoomDelta);
            const int localX = targetX - (candidateX << zoomDelta);
            const int localY = targetY - (candidateY << zoomDelta);
            outU0 = localX * invScale;
            outV0 = localY * invScale;
            outU1 = outU0 + invScale;
            outV1 = outV0 + invScale;
            return true;
        }
        return false;
    };

    for (int ty = y0; ty <= y1; ty++) {
        for (int tx = x0; tx <= x1; tx++) {
            int wrappedX = tx % tilesPerAxis;
            if (wrappedX < 0)
                wrappedX += tilesPerAxis;
            std::shared_ptr<TileEntry> entry;
            float tileU0 = 0.0f;
            float tileV0 = 0.0f;
            float tileU1 = 1.0f;
            float tileV1 = 1.0f;
            if (!findReadyTile(z, wrappedX, ty, entry, tileU0, tileV0, tileU1, tileV1))
                continue;

            const auto textureId = entry->texture.textureId();
            {
                std::lock_guard<std::mutex> lock(entry->mutex);
                entry->lastTouch = std::chrono::steady_clock::now();
            }

            for (int sy = 0; sy < kTileSubdivisions; sy++) {
                for (int sx = 0; sx < kTileSubdivisions; sx++) {
                    const double u0 = (double)sx / (double)kTileSubdivisions;
                    const double u1 = (double)(sx + 1) / (double)kTileSubdivisions;
                    const double v0 = (double)sy / (double)kTileSubdivisions;
                    const double v1 = (double)(sy + 1) / (double)kTileSubdivisions;

                    const ImVec2 p00 = latLonToScreen(vp, origin, tileYToLat((double)ty + v0, z),
                                                      tileXToLon((double)wrappedX + u0, z));
                    const ImVec2 p10 = latLonToScreen(vp, origin, tileYToLat((double)ty + v0, z),
                                                      tileXToLon((double)wrappedX + u1, z));
                    const ImVec2 p11 = latLonToScreen(vp, origin, tileYToLat((double)ty + v1, z),
                                                      tileXToLon((double)wrappedX + u1, z));
                    const ImVec2 p01 = latLonToScreen(vp, origin, tileYToLat((double)ty + v1, z),
                                                      tileXToLon((double)wrappedX + u0, z));

                    const float minX = std::min(std::min(p00.x, p10.x), std::min(p11.x, p01.x));
                    const float maxX = std::max(std::max(p00.x, p10.x), std::max(p11.x, p01.x));
                    const float minY = std::min(std::min(p00.y, p10.y), std::min(p11.y, p01.y));
                    const float maxY = std::max(std::max(p00.y, p10.y), std::max(p11.y, p01.y));
                    if (maxX < origin.x - 32.0f || minX > origin.x + vp.width + 32.0f ||
                        maxY < origin.y - 32.0f || minY > origin.y + vp.height + 32.0f) {
                        continue;
                    }

                    const float drawU0 = tileU0 + (tileU1 - tileU0) * (float)u0;
                    const float drawU1 = tileU0 + (tileU1 - tileU0) * (float)u1;
                    const float drawV0 = tileV0 + (tileV1 - tileV0) * (float)v0;
                    const float drawV1 = tileV0 + (tileV1 - tileV0) * (float)v1;
                    drawList->AddImageQuad(
                        (ImTextureID)(uintptr_t)textureId,
                        p00, p10, p11, p01,
                        ImVec2(drawU0, drawV0),
                        ImVec2(drawU1, drawV0),
                        ImVec2(drawU1, drawV1),
                        ImVec2(drawU0, drawV1),
                        tint);
                }
            }
        }
    }
}

void BasemapRenderer::drawGrid(ImDrawList* drawList, const Viewport& vp,
                               ImVec2 origin, const StylePalette& palette) const {
    if (!m_showGrid)
        return;

    double spacing = 10.0;
    if (vp.zoom > 70.0) spacing = 5.0;
    if (vp.zoom > 140.0) spacing = 2.0;
    if (vp.zoom > 320.0) spacing = 1.0;

    const double minLon = vp.center_lon - vp.halfExtentLon();
    const double maxLon = vp.center_lon + vp.halfExtentLon();
    const double minLat = vp.center_lat - vp.halfExtentLat();
    const double maxLat = vp.center_lat + vp.halfExtentLat();

    ImU32 gridColor = scaleAlpha(palette.gridLine, m_overlayOpacity);
    for (double lon = std::floor(minLon / spacing) * spacing; lon <= maxLon; lon += spacing) {
        const ImVec2 p0 = latLonToScreen(vp, origin, minLat, lon);
        const ImVec2 p1 = latLonToScreen(vp, origin, maxLat, lon);
        drawList->AddLine(p0, p1, gridColor, 1.0f);
    }
    for (double lat = std::floor(minLat / spacing) * spacing; lat <= maxLat; lat += spacing) {
        const ImVec2 p0 = latLonToScreen(vp, origin, lat, minLon);
        const ImVec2 p1 = latLonToScreen(vp, origin, lat, maxLon);
        drawList->AddLine(p0, p1, gridColor, 1.0f);
    }
}

void BasemapRenderer::drawStates(ImDrawList* drawList, const Viewport& vp,
                                 ImVec2 origin, const StylePalette& palette) const {
    if (!m_showStateLines)
        return;

    const ImU32 lineColor = scaleAlpha(palette.stateLine, m_overlayOpacity);
    const float lineWidth = (vp.zoom > 180.0) ? 1.4f : 1.0f;

    for (int i = 0; i < US_STATE_LINE_COUNT; i++) {
        const float lat1 = US_STATE_LINES[i * 4 + 0];
        const float lon1 = US_STATE_LINES[i * 4 + 1];
        const float lat2 = US_STATE_LINES[i * 4 + 2];
        const float lon2 = US_STATE_LINES[i * 4 + 3];
        const ImVec2 p0 = latLonToScreen(vp, origin, lat1, lon1);
        const ImVec2 p1 = latLonToScreen(vp, origin, lat2, lon2);
        if (p0.x < origin.x - 60.0f && p1.x < origin.x - 60.0f) continue;
        if (p0.x > origin.x + vp.width + 60.0f && p1.x > origin.x + vp.width + 60.0f) continue;
        if (p0.y < origin.y - 60.0f && p1.y < origin.y - 60.0f) continue;
        if (p0.y > origin.y + vp.height + 60.0f && p1.y > origin.y + vp.height + 60.0f) continue;
        drawList->AddLine(p0, p1, lineColor, lineWidth);
    }
}

void BasemapRenderer::drawCities(ImDrawList* drawList, const Viewport& vp,
                                 ImVec2 origin, const StylePalette& palette) const {
    if (!m_showCityLabels)
        return;

    int popThreshold = 1000000;
    if (vp.zoom > 40.0) popThreshold = 500000;
    if (vp.zoom > 80.0) popThreshold = 200000;
    if (vp.zoom > 150.0) popThreshold = 100000;
    if (vp.zoom > 300.0) popThreshold = 50000;

    const ImU32 dotColor = scaleAlpha(palette.cityDot, m_overlayOpacity);
    const ImU32 textColor = scaleAlpha(palette.cityText, m_overlayOpacity);
    const ImU32 haloColor = scaleAlpha(palette.cityHalo, m_overlayOpacity);

    for (int i = 0; i < US_CITY_COUNT; i++) {
        if (US_CITIES[i].population < popThreshold)
            continue;
        const ImVec2 p = latLonToScreen(vp, origin, US_CITIES[i].lat, US_CITIES[i].lon);
        if (p.x < origin.x - 60.0f || p.x > origin.x + vp.width + 60.0f ||
            p.y < origin.y - 60.0f || p.y > origin.y + vp.height + 60.0f) {
            continue;
        }

        drawList->AddCircleFilled(p, 2.2f, dotColor);
        drawTextWithHalo(drawList, ImVec2(p.x + 6.0f, p.y - 8.0f),
                         haloColor, textColor, US_CITIES[i].name);
    }
}

void BasemapRenderer::drawBase(ImDrawList* drawList, const Viewport& vp, ImVec2 origin) {
    const StylePalette palette = activePalette();
    drawGradientBackdrop(drawList, vp, origin, palette);
    drawRasterTiles(drawList, vp, origin);
    if (palette.rasterWash != 0) {
        drawList->AddRectFilled(origin,
                                ImVec2(origin.x + vp.width, origin.y + vp.height),
                                palette.rasterWash);
    }
}

void BasemapRenderer::drawOverlay(ImDrawList* drawList, const Viewport& vp, ImVec2 origin) {
    const StylePalette palette = activePalette();
    drawGrid(drawList, vp, origin, palette);
    drawStates(drawList, vp, origin, palette);
    drawCities(drawList, vp, origin, palette);
}
