#pragma once

#include "gl_texture.h"
#include "projection.h"
#include "net/downloader.h"

#include <imgui.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

enum class BasemapStyle {
    Relief = 0,
    OpsDark,
    Satellite,
    SatelliteHybrid
};

const char* basemapStyleLabel(BasemapStyle style);

class BasemapRenderer {
public:
    enum class RasterLayer {
        None = 0,
        Relief,
        Imagery,
        ImageryTopo
    };

    BasemapRenderer();
    ~BasemapRenderer();

    void update(const Viewport& vp);
    void drawBase(ImDrawList* drawList, const Viewport& vp, ImVec2 origin);
    void drawOverlay(ImDrawList* drawList, const Viewport& vp, ImVec2 origin);

    BasemapStyle style() const { return m_style; }
    void setStyle(BasemapStyle style);

    float rasterOpacity() const { return m_rasterOpacity; }
    void setRasterOpacity(float opacity);

    float overlayOpacity() const { return m_overlayOpacity; }
    void setOverlayOpacity(float opacity);

    bool showStateLines() const { return m_showStateLines; }
    void setShowStateLines(bool enabled) { m_showStateLines = enabled; }

    bool showCityLabels() const { return m_showCityLabels; }
    void setShowCityLabels(bool enabled) { m_showCityLabels = enabled; }

    bool showGrid() const { return m_showGrid; }
    void setShowGrid(bool enabled) { m_showGrid = enabled; }

    const std::string& attribution() const { return m_attribution; }
    const std::string& status() const { return m_status; }

private:
    struct TileKey {
        RasterLayer layer = RasterLayer::None;
        int z = 0;
        int x = 0;
        int y = 0;

        bool operator==(const TileKey& other) const {
            return layer == other.layer &&
                   z == other.z &&
                   x == other.x &&
                   y == other.y;
        }
    };

    struct TileKeyHash {
        size_t operator()(const TileKey& key) const;
    };

    enum class TileState {
        Empty = 0,
        Queued,
        Decoded,
        Ready,
        Failed
    };

    struct TileEntry {
        std::mutex mutex;
        TileState state = TileState::Empty;
        int width = 0;
        int height = 0;
        std::vector<unsigned char> rgba;
        GlTexture2D texture;
        std::string error;
        std::chrono::steady_clock::time_point lastTouch = {};
        std::chrono::steady_clock::time_point nextRetry = {};
        int failureCount = 0;
    };

    struct StylePalette {
        ImU32 backgroundTop = 0;
        ImU32 backgroundBottom = 0;
        ImU32 backgroundAccent = 0;
        ImU32 rasterWash = 0;
        ImU32 stateLine = 0;
        ImU32 cityDot = 0;
        ImU32 cityText = 0;
        ImU32 cityHalo = 0;
        ImU32 gridLine = 0;
    };

    RasterLayer activeRasterLayer() const;
    StylePalette activePalette() const;
    void updateAttribution();
    void requestVisibleTiles(const Viewport& vp);
    void queueTileRequest(const TileKey& key);
    void uploadDecodedTiles();
    void trimCache();
    void touchVisibleTile(const TileKey& key);
    bool loadTileFromDiskCache(const TileKey& key, TileEntry& entry);
    void storeTileToDiskCache(const TileKey& key, const std::vector<uint8_t>& bytes);
    std::filesystem::path tileCachePath(const TileKey& key) const;
    void drawRasterTiles(ImDrawList* drawList, const Viewport& vp, ImVec2 origin);
    void drawGradientBackdrop(ImDrawList* drawList, const Viewport& vp, ImVec2 origin, const StylePalette& palette);
    void drawGrid(ImDrawList* drawList, const Viewport& vp, ImVec2 origin, const StylePalette& palette) const;
    void drawStates(ImDrawList* drawList, const Viewport& vp, ImVec2 origin, const StylePalette& palette) const;
    void drawCities(ImDrawList* drawList, const Viewport& vp, ImVec2 origin, const StylePalette& palette) const;

    BasemapStyle m_style = BasemapStyle::OpsDark;
    float m_rasterOpacity = 0.82f;
    float m_overlayOpacity = 1.0f;
    bool m_showStateLines = true;
    bool m_showCityLabels = true;
    bool m_showGrid = true;

    Downloader m_downloader;
    mutable std::mutex m_cacheMutex;
    std::unordered_map<TileKey, std::shared_ptr<TileEntry>, TileKeyHash> m_tiles;
    std::filesystem::path m_diskCacheDir;

    std::string m_attribution;
    std::string m_status;
};
