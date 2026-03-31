#include "ui.h"
#include "app.h"
#include "nexrad/products.h"
#include "nexrad/stations.h"
#include "net/aws_nexrad.h"
#include "historic.h"
#include "net/warnings.h"
#include "data/us_boundaries.h"
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>
#endif
#include <imgui.h>
#include <imgui_internal.h>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <ctime>

namespace ui {

namespace {

void resetConusView(App& app) {
    app.viewport().center_lat = 39.0;
    app.viewport().center_lon = -98.0;
    app.viewport().zoom = 28.0;
}

bool containsCaseInsensitive(const std::string& haystack, const char* needle) {
    if (!needle || !needle[0]) return true;
    std::string lhs = haystack;
    std::string rhs = needle;
    std::transform(lhs.begin(), lhs.end(), lhs.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    std::transform(rhs.begin(), rhs.end(), rhs.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return lhs.find(rhs) != std::string::npos;
}

ImVec4 rgbaToImVec4(uint32_t color) {
    return ImVec4(
        (float)(color & 0xFF) / 255.0f,
        (float)((color >> 8) & 0xFF) / 255.0f,
        (float)((color >> 16) & 0xFF) / 255.0f,
        (float)((color >> 24) & 0xFF) / 255.0f);
}

uint32_t imVec4ToRgba(const ImVec4& color) {
    auto chan = [](float v) { return (uint32_t)std::clamp((int)std::lround(v * 255.0f), 0, 255); };
    return chan(color.x) | (chan(color.y) << 8) | (chan(color.z) << 16) | (chan(color.w) << 24);
}

void copyTextBuffer(char* dst, size_t dstCap, const char* src) {
    if (!dst || dstCap == 0) return;
    std::snprintf(dst, dstCap, "%s", src ? src : "");
}

void editWarningColor(const char* label, uint32_t& color) {
    ImVec4 value = rgbaToImVec4(color);
    if (ImGui::ColorEdit4(label, &value.x, ImGuiColorEditFlags_NoInputs))
        color = imVec4ToRgba(value);
}

const char* performanceProfileLabel(PerformanceProfile profile) {
    switch (profile) {
        case PerformanceProfile::Auto: return "Auto";
        case PerformanceProfile::Quality: return "Quality";
        case PerformanceProfile::Balanced: return "Balanced";
        case PerformanceProfile::Performance: return "Performance";
        default: return "Unknown";
    }
}

std::string formatBytes(size_t bytes) {
    static const char* kUnits[] = {"B", "KB", "MB", "GB", "TB"};
    double value = (double)bytes;
    int unit = 0;
    while (value >= 1024.0 && unit < 4) {
        value /= 1024.0;
        unit++;
    }

    char buffer[32];
    if (unit == 0)
        std::snprintf(buffer, sizeof(buffer), "%llu %s", (unsigned long long)bytes, kUnits[unit]);
    else
        std::snprintf(buffer, sizeof(buffer), "%.2f %s", value, kUnits[unit]);
    return buffer;
}

void centerOnWarning(App& app, const WarningPolygon& warning) {
    if (warning.lats.empty() || warning.lons.empty()) return;

    float minLat = warning.lats[0], maxLat = warning.lats[0];
    float minLon = warning.lons[0], maxLon = warning.lons[0];
    float latSum = 0.0f, lonSum = 0.0f;
    for (size_t i = 0; i < warning.lats.size(); i++) {
        minLat = std::min(minLat, warning.lats[i]);
        maxLat = std::max(maxLat, warning.lats[i]);
        minLon = std::min(minLon, warning.lons[i]);
        maxLon = std::max(maxLon, warning.lons[i]);
        latSum += warning.lats[i];
        lonSum += warning.lons[i];
    }

    app.viewport().center_lat = latSum / (float)warning.lats.size();
    app.viewport().center_lon = lonSum / (float)warning.lons.size();

    float spanLat = std::max(0.2f, maxLat - minLat);
    float spanLon = std::max(0.2f, maxLon - minLon);
    float zoomLat = (float)app.viewport().height / (spanLat * 3.0f);
    float zoomLon = (float)app.viewport().width / (spanLon * 3.0f);
    app.viewport().zoom = std::max(20.0, std::min((double)std::min(zoomLat, zoomLon), 400.0));
}

bool browseForColorTable(char* path, size_t pathCapacity) {
#ifdef _WIN32
    if (!path || pathCapacity == 0) return false;

    char fileBuf[1024] = "";
    copyTextBuffer(fileBuf, sizeof(fileBuf), path);

    static const char kFilter[] =
        "Radar Color Tables (*.pal;*.ct;*.ct3;*.tbl;*.txt)\0*.pal;*.ct;*.ct3;*.tbl;*.txt\0"
        "All Files (*.*)\0*.*\0";

    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = GetActiveWindow();
    ofn.lpstrFilter = kFilter;
    ofn.lpstrFile = fileBuf;
    ofn.nMaxFile = (DWORD)sizeof(fileBuf);
    ofn.lpstrTitle = "Open Radar Color Table";
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    ofn.lpstrDefExt = "pal";

    if (!GetOpenFileNameA(&ofn))
        return false;

    copyTextBuffer(path, pathCapacity, fileBuf);
    return true;
#else
    (void)path;
    (void)pathCapacity;
    return false;
#endif
}

bool parseArchiveDate(const char* text, int& year, int& month, int& day) {
    if (!text) return false;
    return std::sscanf(text, "%d-%d-%d", &year, &month, &day) == 3;
}

bool parseArchiveTime(const char* text, int& hour, int& minute) {
    if (!text) return false;
    return std::sscanf(text, "%d:%d", &hour, &minute) == 2;
}

void seedArchiveInputs(char* dateText, size_t dateCap,
                       char* startText, size_t startCap,
                       char* endText, size_t endCap) {
    static bool seeded = false;
    if (seeded) return;
    seeded = true;

    std::time_t now = std::time(nullptr);
    std::tm utc = {};
#ifdef _WIN32
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif

    int startYear = utc.tm_year + 1900;
    int startMonth = utc.tm_mon + 1;
    int startDay = utc.tm_mday;
    int startHour = utc.tm_hour - 1;
    if (startHour < 0) {
        startHour += 24;
        shiftDate(startYear, startMonth, startDay, -1);
    }

    std::snprintf(dateText, dateCap, "%04d-%02d-%02d", startYear, startMonth, startDay);
    std::snprintf(startText, startCap, "%02d:%02d", startHour, utc.tm_min);
    std::snprintf(endText, endCap, "%02d:%02d", utc.tm_hour, utc.tm_min);
}

void ensureDockLayout() {
    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    ImGuiID dockspaceId = ImGui::GetID("cursdar2.main_dockspace");
    ImGuiViewport* viewport = ImGui::GetMainViewport();

    ImGui::DockBuilderRemoveNode(dockspaceId);
    ImGui::DockBuilderAddNode(dockspaceId,
                              ImGuiDockNodeFlags_DockSpace |
                              ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::DockBuilderSetNodeSize(dockspaceId, viewport->WorkSize);

    ImGuiID dockLeft = dockspaceId;
    ImGuiID dockCenter = 0;
    ImGuiID dockRight = 0;
    ImGuiID dockBottom = 0;
    ImGuiID dockRightTop = 0;
    ImGuiID dockRightBottom = 0;

    ImGui::DockBuilderSplitNode(dockspaceId, ImGuiDir_Left, 0.23f, &dockLeft, &dockCenter);
    ImGui::DockBuilderSplitNode(dockCenter, ImGuiDir_Right, 0.27f, &dockRight, &dockCenter);
    ImGui::DockBuilderSplitNode(dockCenter, ImGuiDir_Down, 0.28f, &dockBottom, &dockCenter);
    ImGui::DockBuilderSplitNode(dockRight, ImGuiDir_Down, 0.52f, &dockRightBottom, &dockRightTop);

    ImGui::DockBuilderDockWindow("Operator Console", dockLeft);
    ImGui::DockBuilderDockWindow("Inspector", dockRightTop);
    ImGui::DockBuilderDockWindow("Station Browser", dockRightBottom);
    ImGui::DockBuilderDockWindow("Warnings", dockBottom);
    ImGui::DockBuilderDockWindow("Cross-Section Console", dockBottom);
    ImGui::DockBuilderDockWindow("Historic Timeline", dockBottom);
    ImGui::DockBuilderFinish(dockspaceId);
}

} // namespace

void init() {
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 4.0f;
    style.FrameRounding = 3.0f;
    style.GrabRounding = 3.0f;
    style.WindowBorderSize = 1.0f;
    style.FramePadding = ImVec2(8, 4);
    style.ItemSpacing = ImVec2(8, 6);

    // Dark operator-console theme
    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg] = ImVec4(0.055f, 0.06f, 0.07f, 0.94f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.08f, 0.09f, 0.11f, 1.0f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.11f, 0.14f, 0.18f, 1.0f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.11f, 0.13f, 0.16f, 1.0f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.17f, 0.21f, 0.28f, 1.0f);
    colors[ImGuiCol_Button] = ImVec4(0.14f, 0.17f, 0.22f, 1.0f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.22f, 0.28f, 0.38f, 1.0f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.27f, 0.33f, 0.46f, 1.0f);
    colors[ImGuiCol_Header] = ImVec4(0.15f, 0.18f, 0.24f, 1.0f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.20f, 0.26f, 0.34f, 1.0f);
    colors[ImGuiCol_Tab] = ImVec4(0.10f, 0.11f, 0.13f, 1.0f);
    colors[ImGuiCol_TabSelected] = ImVec4(0.20f, 0.24f, 0.31f, 1.0f);
}

void render(App& app) {
    auto& vp = app.viewport();
    ImGuiViewport* mainViewport = ImGui::GetMainViewport();
    const auto stations = app.stations();
    const auto warnings = app.currentWarnings();
    static char palettePath[512] = "";
    static char pollingLinkUrl[1024] = "";
    static char archiveStation[8] = "";
    static char archiveDate[16] = "";
    static char archiveStart[8] = "";
    static char archiveEnd[8] = "";
    static std::string archiveStatus;

    seedArchiveInputs(archiveDate, sizeof(archiveDate),
                      archiveStart, sizeof(archiveStart),
                      archiveEnd, sizeof(archiveEnd));
    if (!archiveStation[0] && !app.m_historicMode) {
        const int activeIdx = app.activeStation();
        if (activeIdx >= 0 && activeIdx < NUM_NEXRAD_STATIONS)
            copyTextBuffer(archiveStation, sizeof(archiveStation),
                           NEXRAD_STATIONS[activeIdx].icao);
    }

    ImGuiID dockspaceId = ImGui::GetID("cursdar2.main_dockspace");
    ImGui::DockSpaceOverViewport(dockspaceId, ImGui::GetMainViewport(),
                                 ImGuiDockNodeFlags_PassthruCentralNode);
    ensureDockLayout();

    // Background radar image
    auto* drawList = ImGui::GetBackgroundDrawList();
    drawList->AddImage(
        (ImTextureID)(uintptr_t)app.outputTexture().textureId(),
        mainViewport->Pos,
        ImVec2(mainViewport->Pos.x + mainViewport->Size.x,
               mainViewport->Pos.y + mainViewport->Size.y));

    // ── State boundaries ─────────────────────────────────────
    {
        auto* bdl = ImGui::GetBackgroundDrawList();
        ImU32 lineCol = IM_COL32(50, 50, 70, 140);
        for (int i = 0; i < US_STATE_LINE_COUNT; i++) {
            float lat1 = US_STATE_LINES[i*4+0], lon1 = US_STATE_LINES[i*4+1];
            float lat2 = US_STATE_LINES[i*4+2], lon2 = US_STATE_LINES[i*4+3];
            float sx1 = (float)((lon1 - vp.center_lon) * vp.zoom + vp.width * 0.5);
            float sy1 = (float)((vp.center_lat - lat1) * vp.zoom + vp.height * 0.5);
            float sx2 = (float)((lon2 - vp.center_lon) * vp.zoom + vp.width * 0.5);
            float sy2 = (float)((vp.center_lat - lat2) * vp.zoom + vp.height * 0.5);
            // Coarse viewport cull
            if (sx1 < -50 && sx2 < -50) continue;
            if (sx1 > vp.width+50 && sx2 > vp.width+50) continue;
            if (sy1 < -50 && sy2 < -50) continue;
            if (sy1 > vp.height+50 && sy2 > vp.height+50) continue;
            bdl->AddLine(ImVec2(sx1,sy1), ImVec2(sx2,sy2), lineCol, 1.0f);
        }
    }

    // ── City labels (zoom-dependent) ────────────────────────
    {
        auto* cdl = ImGui::GetBackgroundDrawList();
        // Determine population threshold based on zoom
        int popThreshold = 1000000;  // very zoomed out: only mega cities
        if (vp.zoom > 40) popThreshold = 500000;
        if (vp.zoom > 80) popThreshold = 200000;
        if (vp.zoom > 150) popThreshold = 100000;
        if (vp.zoom > 300) popThreshold = 50000;

        for (int i = 0; i < US_CITY_COUNT; i++) {
            if (US_CITIES[i].population < popThreshold) continue;
            float sx = (float)((US_CITIES[i].lon - vp.center_lon) * vp.zoom + vp.width * 0.5);
            float sy = (float)((vp.center_lat - US_CITIES[i].lat) * vp.zoom + vp.height * 0.5);
            if (sx < -50 || sx > vp.width+50 || sy < -50 || sy > vp.height+50) continue;
            cdl->AddCircleFilled(ImVec2(sx, sy), 2.0f, IM_COL32(200, 200, 220, 180));
            cdl->AddText(ImVec2(sx + 5, sy - 7), IM_COL32(200, 200, 220, 160),
                         US_CITIES[i].name);
        }
    }

    // ── Range rings + azimuth lines ─────────────────────────
    {
        int asi = app.activeStation();
        float slat = 0, slon = 0;
        if (app.m_historicMode) {
            auto* ev = app.m_historic.currentEvent();
            if (ev) {
                // Find station lat/lon from NEXRAD_STATIONS
                for (int i = 0; i < NUM_NEXRAD_STATIONS; i++) {
                    if (strcmp(NEXRAD_STATIONS[i].icao, ev->station) == 0) {
                        slat = NEXRAD_STATIONS[i].lat;
                        slon = NEXRAD_STATIONS[i].lon;
                        break;
                    }
                }
            }
            if (slat == 0.0f && slon == 0.0f) {
                const std::string histStation = app.m_historic.currentStation();
                for (int i = 0; i < NUM_NEXRAD_STATIONS; i++) {
                    if (histStation == NEXRAD_STATIONS[i].icao) {
                        slat = NEXRAD_STATIONS[i].lat;
                        slon = NEXRAD_STATIONS[i].lon;
                        break;
                    }
                }
            }
        } else if (asi >= 0 && asi < (int)stations.size()) {
            const auto& st = stations[asi];
            slat = st.display_lat;
            slon = st.display_lon;
        }

        if (slat != 0 && slon != 0 && !app.showAll() && !app.mode3D()) {
            auto* rdl = ImGui::GetBackgroundDrawList();
            float scx = (float)((slon - vp.center_lon) * vp.zoom + vp.width * 0.5);
            float scy = (float)((vp.center_lat - slat) * vp.zoom + vp.height * 0.5);
            float km_per_deg = 111.0f;

            // Range rings at 50km intervals
            ImU32 ringCol = IM_COL32(60, 60, 80, 100);
            for (int r = 50; r <= 450; r += 50) {
                float deg = (float)r / km_per_deg;
                float px_radius = (float)(deg * vp.zoom);
                if (px_radius < 10 || px_radius > vp.width * 3) continue;
                rdl->AddCircle(ImVec2(scx, scy), px_radius, ringCol, 72);
                if (px_radius > 30) {
                    char buf[16];
                    snprintf(buf, sizeof(buf), "%d", r);
                    rdl->AddText(ImVec2(scx + px_radius + 2, scy - 7),
                                 IM_COL32(80, 80, 110, 140), buf);
                }
            }

            // Cardinal + intercardinal azimuth lines
            ImU32 azCol = IM_COL32(50, 50, 70, 70);
            float maxR = 460.0f / km_per_deg * (float)vp.zoom;
            const char* dirs[] = {"N","NE","E","SE","S","SW","W","NW"};
            for (int d = 0; d < 8; d++) {
                float angle = d * 45.0f * 3.14159265f / 180.0f;
                float ex = scx + sinf(angle) * maxR;
                float ey = scy - cosf(angle) * maxR;
                float lw = (d % 2 == 0) ? 1.0f : 0.5f; // cardinals thicker
                rdl->AddLine(ImVec2(scx, scy), ImVec2(ex, ey), azCol, lw);
                float lx = scx + sinf(angle) * fminf(maxR, 60.0f);
                float ly = scy - cosf(angle) * fminf(maxR, 60.0f);
                if (d % 2 == 0) // only label cardinals
                    rdl->AddText(ImVec2(lx - 4, ly - 7),
                                 IM_COL32(100, 100, 140, 180), dirs[d]);
            }
        }
    }

    // ── Status bar (top) ────────────────────────────────────
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2((float)vp.width, 38));
    ImGui::Begin("##statusbar", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDocking);

    int loaded = app.stationsLoaded();
    int total = app.stationsTotal();
    int downloading = app.stationsDownloading();
    int warningCount = (int)warnings.size();

    ImGui::TextColored(ImVec4(0.90f, 0.96f, 1.0f, 1.0f), "CURSDAR2");
    ImGui::SameLine(88);

    int asi = app.activeStation();
    if (app.m_historicMode) {
        const std::string histStation = app.m_historic.currentStation();
        const std::string histLabel = app.m_historic.currentLabel();
        auto* fr = app.m_historic.frame(app.m_historic.currentFrame());
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "%s",
                           histStation.empty() ? "---" : histStation.c_str());
        ImGui::SameLine(168);
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "%s",
                           histLabel.empty() ? "Archive Playback" : histLabel.c_str());
        ImGui::SameLine(430);
        ImGui::Text("%s UTC", fr ? fr->timestamp.c_str() : "--:--");
    } else {
        const char* stName = (asi >= 0 && asi < total) ? NEXRAD_STATIONS[asi].name : "---";
        std::string activeStation = app.activeStationName();
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1.0f), "%s", activeStation.c_str());
        ImGui::SameLine(168);
        ImGui::Text("%s", stName);
        if (app.snapshotMode()) {
            ImGui::SameLine(430);
            ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "SNAPSHOT: %s", app.snapshotLabel());
        } else {
            ImGui::SameLine(430);
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), "LIVE");
            if (app.liveLoopViewingHistory()) {
                ImGui::SameLine(500);
                ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.25f, 1.0f), "LOOP");
            }
        }
    }
    ImGui::SameLine(650);
    ImGui::Text("%s | Tilt %d/%d (%.1f deg)",
                PRODUCT_INFO[app.activeProduct()].name,
                app.activeTilt() + 1, app.maxTilts(), app.activeTiltAngle());
    ImGui::SameLine(960);
    ImGui::Text("Loaded: %d/%d", loaded, total);
    if (downloading > 0) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.3f, 0.8f, 1.0f, 1.0f), "(%d DL)", downloading);
    }
    ImGui::SameLine();
    ImGui::Text("| Alerts: %d", warningCount);
    if (!app.m_historicMode && !app.autoTrackStation()) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.35f, 1.0f), "| SITE LOCK");
    }

    ImGui::End();

    // ── Controls panel (left) ───────────────────────────────
    ImGui::SetNextWindowSize(ImVec2(320, 760), ImGuiCond_FirstUseEver);
    ImGui::Begin("Operator Console");

    // Product buttons
    ImGui::Text("Product (Left/Right):");
    for (int i = 0; i < (int)Product::COUNT; i++) {
        bool selected = (app.activeProduct() == i);
        if (selected)
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.35f, 0.55f, 1.0f));

        char label[64];
        snprintf(label, sizeof(label), "[%d] %s", i + 1, PRODUCT_INFO[i].name);
        if (ImGui::Button(label, ImVec2(210, 24)))
            app.setProduct(i);

        if (selected) ImGui::PopStyleColor();
    }

    ImGui::Separator();

    // Tilt selector
    ImGui::Text("Tilt / Elevation (Up/Down):");
    char tiltLabel[64];
    snprintf(tiltLabel, sizeof(tiltLabel), "Tilt %d/%d  (%.1f deg)",
             app.activeTilt() + 1, app.maxTilts(), app.activeTiltAngle());
    ImGui::Text("%s", tiltLabel);
    if (app.showAll() || app.snapshotMode())
        ImGui::TextDisabled("Mosaic uses lowest sweep per site");
    if (ImGui::Button("Tilt Up", ImVec2(100, 24))) app.nextTilt();
    ImGui::SameLine();
    if (ImGui::Button("Tilt Down", ImVec2(100, 24))) app.prevTilt();

    ImGui::Separator();

    // Product-aware threshold slider
    bool velocityFilter = (app.activeProduct() == PROD_VEL);
    ImGui::Text("%s", velocityFilter ? "Min |Velocity| Filter:" : "Min dBZ Filter:");
    float threshold = app.dbzMinThreshold();
    bool changed = velocityFilter
        ? ImGui::SliderFloat("##dbz", &threshold, 0.0f, 50.0f, "%.0f m/s")
        : ImGui::SliderFloat("##dbz", &threshold, -30.0f, 40.0f, "%.0f dBZ");
    if (changed) {
        app.setDbzMinThreshold(threshold);
    }

    ImGui::Separator();

    bool autoTrack = app.autoTrackStation();
    if (ImGui::Checkbox("Auto Track Nearest Site", &autoTrack))
        app.setAutoTrackStation(autoTrack);

    if (ImGui::CollapsingHeader("Radar Activation", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Enabled Sites: %d / %d", app.enabledStationCount(), app.stationsTotal());
        ImGui::TextWrapped("Gray map sites are idle. Click a site in the browser or on the map to enable it.");
        const int activeIdx = app.activeStation();
        if (activeIdx >= 0 && activeIdx < NUM_NEXRAD_STATIONS) {
            const bool enabled = app.stationEnabled(activeIdx);
            if (ImGui::Button(enabled ? "Disable Active Site" : "Enable Active Site", ImVec2(210, 24)))
                app.setStationEnabled(activeIdx, !enabled);
        } else {
            ImGui::BeginDisabled();
            ImGui::Button("Enable Active Site", ImVec2(210, 24));
            ImGui::EndDisabled();
        }
        if (ImGui::Button("Disable All Sites", ImVec2(210, 24)))
            app.disableAllStations();
        ImGui::Separator();
    }

    if (ImGui::Button(app.mode3D() ? "Exit 3D (V)" : "3D Volume (V)", ImVec2(210, 24)))
        app.toggle3D();
    if (ImGui::Button(app.crossSection() ? "Close Cross Section (X)" : "Cross Section (X)", ImVec2(210, 24)))
        app.toggleCrossSection();

    ImGui::Separator();

    // Show All toggle
    bool showAll = app.showAll();
    if (ImGui::Button(showAll ? "Single Station" : "Show All (A)", ImVec2(210, 24)))
        app.toggleShowAll();

    ImGui::Separator();
    if (ImGui::Button("Refresh Data", ImVec2(210, 24)))
        app.refreshData();
    if (ImGui::Button("Reset CONUS View", ImVec2(210, 24)))
        resetConusView(app);

    if (!app.snapshotMode()) {
        if (ImGui::Button("Load Mar 30 2025 5 PM ET", ImVec2(210, 24)))
            app.loadMarch302025Snapshot();
        if (ImGui::Button("Load Mar 30 2025 Lowest Sweep", ImVec2(210, 24)))
            app.loadMarch302025Snapshot(true);
    } else {
        if (ImGui::Button("Back to Live", ImVec2(210, 24)))
            app.refreshData();
    }

    ImGui::Separator();

    // ── SRV mode ────────────────────────────────────────────
    if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
        static const char* kProfileItems[] = {"Auto", "Quality", "Balanced", "Performance"};
        int profileIdx = (int)app.requestedPerformanceProfile();
        ImGui::SetNextItemWidth(210);
        if (ImGui::Combo("Profile", &profileIdx, kProfileItems, IM_ARRAYSIZE(kProfileItems)))
            app.setPerformanceProfile((PerformanceProfile)profileIdx);

        const auto& mem = app.memoryTelemetry();
        ImGui::TextWrapped("GPU: %s", app.gpuName().empty() ? "Unknown" : app.gpuName().c_str());
        ImGui::Text("Effective: %s", performanceProfileLabel(app.effectivePerformanceProfile()));
        ImGui::Text("Internal Render: %dx%d (%.2fx)",
                    mem.internal_render_width, mem.internal_render_height, mem.render_scale);
        ImGui::Text("VRAM: %s / %s",
                    formatBytes(mem.gpu_used_bytes).c_str(),
                    formatBytes(mem.gpu_total_bytes).c_str());
        ImGui::Text("VRAM Peak: %s", formatBytes(mem.gpu_peak_used_bytes).c_str());
        ImGui::Text("Process RAM: %s", formatBytes(mem.process_working_set_bytes).c_str());
        ImGui::Text("Process Peak: %s", formatBytes(mem.process_peak_working_set_bytes).c_str());
        ImGui::Text("Archive Cache: %s", formatBytes(mem.historic_cache_bytes).c_str());
        ImGui::Text("Live Loop Cache: %s", formatBytes(mem.live_loop_bytes).c_str());
        ImGui::Text("3D Volume Working Set: %s", formatBytes(mem.volume_working_set_bytes).c_str());
        if (ImGui::Button("Reset Memory Peaks", ImVec2(210, 24)))
            app.resetMemoryPeaks();
    }

    ImGui::Separator();

    if (!app.m_historicMode && !app.snapshotMode() &&
        ImGui::CollapsingHeader("Live Loop", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool liveLoopEnabled = app.liveLoopEnabled();
        if (ImGui::Checkbox("Enable Realtime Loop", &liveLoopEnabled))
            app.setLiveLoopEnabled(liveLoopEnabled);

        ImGui::BeginDisabled(!liveLoopEnabled);
        int loopFrames = app.liveLoopLength();
        if (ImGui::SliderInt("Loop Frames", &loopFrames, 1, app.liveLoopMaxFrames()))
            app.setLiveLoopLength(loopFrames);

        float loopSpeed = app.liveLoopSpeed();
        if (ImGui::SliderFloat("Loop FPS", &loopSpeed, 1.0f, 15.0f, "%.0f fps"))
            app.setLiveLoopSpeed(loopSpeed);

        if (ImGui::Button(app.liveLoopPlaying() ? "Pause Loop" : "Play Loop", ImVec2(102, 24)))
            app.toggleLiveLoopPlayback();
        ImGui::SameLine();
        if (ImGui::Button("Jump Live", ImVec2(102, 24)))
            app.setLiveLoopPlaybackFrame(app.liveLoopAvailableFrames() > 0
                ? app.liveLoopAvailableFrames() - 1
                : 0);

        ImGui::Text("Cached Frames: %d / %d",
                    app.liveLoopAvailableFrames(), app.liveLoopLength());
        std::string liveLoopLabel = app.liveLoopCurrentLabel();
        if (!liveLoopLabel.empty())
            ImGui::TextWrapped("%s", liveLoopLabel.c_str());
        if (app.mode3D() || app.crossSection())
            ImGui::TextDisabled("Realtime loop playback is 2D-only.");
        if (ImGui::Button("Clear Loop", ImVec2(210, 24)))
            app.clearLiveLoop();
        ImGui::EndDisabled();
        ImGui::Separator();
    }

    if (app.activeProduct() == PROD_VEL) {
        bool srv = app.srvMode();
        if (ImGui::Checkbox("Storm-Relative (S)", &srv))
            app.toggleSRV();
        if (srv) {
            float spd = app.stormSpeed();
            float dir = app.stormDir();
            ImGui::SetNextItemWidth(100);
            if (ImGui::SliderFloat("##srvSpd", &spd, 0.0f, 40.0f, "%.0f m/s"))
                app.setStormMotion(spd, dir);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(100);
            if (ImGui::SliderFloat("##srvDir", &dir, 0.0f, 360.0f, "%.0f deg"))
                app.setStormMotion(spd, dir);
        }
        ImGui::Separator();
    }

    // ── Detection overlays ──────────────────────────────────
    if (ImGui::CollapsingHeader("Detection Overlays", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("TDS (Debris)", &app.m_showTDS);
        ImGui::Checkbox("Hail (HDR)", &app.m_showHail);
        ImGui::Checkbox("Meso/TVS", &app.m_showMeso);
        ImGui::Checkbox("Dealiasing", &app.m_dealias);
    }

    ImGui::Separator();

    // ── Historic Events ─────────────────────────────────────
    if (ImGui::CollapsingHeader("Color Tables", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextWrapped("Load GR / RadarScope palette files for the matching radar product.");
        ImGui::SetNextItemWidth(210);
        ImGui::InputText("##palette_path", palettePath, sizeof(palettePath));
        ImGui::SameLine();
        if (ImGui::Button("Browse...", ImVec2(86, 24))) {
            if (browseForColorTable(palettePath, sizeof(palettePath)))
                app.loadColorTableFromFile(palettePath);
        }
        if (ImGui::Button("Load Palette", ImVec2(102, 24)))
            app.loadColorTableFromFile(palettePath);
        ImGui::SameLine();
        if (ImGui::Button("Reset Product", ImVec2(102, 24)))
            app.resetColorTable();
        ImGui::TextWrapped("%s", app.colorTableStatus().empty()
            ? "Built-in CUDA palettes are active."
            : app.colorTableStatus().c_str());
        const std::string activePalette = app.colorTableLabel(app.activeProduct());
        if (!activePalette.empty())
            ImGui::Text("Active %s palette: %s",
                        PRODUCT_INFO[app.activeProduct()].code, activePalette.c_str());
    }

    ImGui::Separator();

    if (ImGui::CollapsingHeader("Polling Links")) {
        ImGui::TextWrapped("Initial GR-style placefile intake: fetch, inspect, and track polling links.");
        ImGui::SetNextItemWidth(210);
        ImGui::InputText("##polling_link", pollingLinkUrl, sizeof(pollingLinkUrl));
        if (ImGui::Button("Add Link", ImVec2(102, 24))) {
            std::string error;
            if (!app.m_pollingLinks.addLink(pollingLinkUrl, error))
                app.m_colorTableStatus = "Polling link failed: " + error;
            else
                pollingLinkUrl[0] = '\0';
        }
        ImGui::SameLine();
        if (ImGui::Button("Refresh Links", ImVec2(102, 24)))
            app.m_pollingLinks.refreshAll();

        auto pollingEntries = app.m_pollingLinks.entries();
        if (pollingEntries.empty()) {
            ImGui::TextDisabled("No polling links loaded.");
        } else {
            for (size_t i = 0; i < pollingEntries.size(); i++) {
                const auto& entry = pollingEntries[i];
                ImGui::Separator();
                ImGui::TextWrapped("%s", entry.title.c_str());
                ImGui::TextDisabled("%s", entry.url.c_str());
                ImGui::Text("Polygons %d  Lines %d  Text %d  Icons %d",
                            entry.polygon_count, entry.line_count, entry.text_count, entry.icon_count);
                if (!entry.last_error.empty())
                    ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.4f, 1.0f), "%s", entry.last_error.c_str());
                else
                    ImGui::Text("Last fetch: %s", entry.last_fetch_utc.c_str());
            }
        }
    }

    ImGui::Separator();

    if (ImGui::CollapsingHeader("Archive Downloads", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextWrapped("Download arbitrary Level 2 volume ranges for one radar. Times are UTC; if the end time is earlier than the start time, playback rolls into the next UTC day.");
        ImGui::SetNextItemWidth(92);
        ImGui::InputText("Station", archiveStation, sizeof(archiveStation));
        ImGui::SameLine();
        if (ImGui::Button("Use Active Site", ImVec2(110, 24))) {
            const int activeIdx = app.activeStation();
            if (activeIdx >= 0 && activeIdx < NUM_NEXRAD_STATIONS)
                copyTextBuffer(archiveStation, sizeof(archiveStation),
                               NEXRAD_STATIONS[activeIdx].icao);
        }
        ImGui::SetNextItemWidth(110);
        ImGui::InputText("UTC Date", archiveDate, sizeof(archiveDate));
        ImGui::SetNextItemWidth(76);
        ImGui::InputText("Start", archiveStart, sizeof(archiveStart));
        ImGui::SameLine();
        ImGui::SetNextItemWidth(76);
        ImGui::InputText("End", archiveEnd, sizeof(archiveEnd));
        if (ImGui::Button("Download Range", ImVec2(210, 24))) {
            int year = 0, month = 0, day = 0;
            int startHour = 0, startMin = 0;
            int endHour = 0, endMin = 0;

            for (char& c : archiveStation)
                c = (char)std::toupper((unsigned char)c);

            if (!archiveStation[0]) {
                archiveStatus = "Enter a 4-letter radar site ICAO.";
            } else if (!parseArchiveDate(archiveDate, year, month, day)) {
                archiveStatus = "Date must use YYYY-MM-DD in UTC.";
            } else if (month < 1 || month > 12 || day < 1 || day > daysInMonth(year, month)) {
                archiveStatus = "Date is outside the valid calendar range.";
            } else if (!parseArchiveTime(archiveStart, startHour, startMin) ||
                       !parseArchiveTime(archiveEnd, endHour, endMin) ||
                       startHour < 0 || startHour > 23 || endHour < 0 || endHour > 23 ||
                       startMin < 0 || startMin > 59 || endMin < 0 || endMin > 59) {
                archiveStatus = "Times must use HH:MM in UTC.";
            } else if (app.loadArchiveRange(archiveStation, year, month, day,
                                            startHour, startMin, endHour, endMin)) {
                archiveStatus = "Downloading archive range...";
            } else {
                archiveStatus = "Archive request rejected. Check the radar site and UTC range.";
            }
        }
        if (!archiveStatus.empty())
            ImGui::TextWrapped("%s", archiveStatus.c_str());
        const std::string histError = app.m_historic.lastError();
        if (!histError.empty())
            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.4f, 1.0f), "%s", histError.c_str());
        if (app.m_historicMode) {
            ImGui::Separator();
            if (ImGui::Button("Back to Live", ImVec2(210, 24)))
                app.refreshData();
        }
    }

    ImGui::Separator();

    if (ImGui::CollapsingHeader("Historic Cases")) {
        for (int i = 0; i < NUM_HISTORIC_EVENTS; i++) {
            auto& ev = HISTORIC_EVENTS[i];
            if (ImGui::Button(ev.name, ImVec2(210, 22))) {
                app.loadHistoricEvent(i);
                archiveStatus.clear();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::Text("%s", ev.description);
                ImGui::Text("Station: %s  |  %04d-%02d-%02d",
                            ev.station, ev.year, ev.month, ev.day);
                ImGui::Text("%02d:%02d - %02d:%02d UTC",
                            ev.start_hour, ev.start_min, ev.end_hour, ev.end_min);
                ImGui::EndTooltip();
            }
        }

    }

    // (Demo packs removed)

    ImGui::End();

    // ── Single-station timeline (historic mode) ─────────────
    if (app.m_historicMode) {
        auto& hist = app.m_historic;
        ImGui::SetNextWindowSize(ImVec2(640, 180), ImGuiCond_FirstUseEver);
        ImGui::Begin("Historic Timeline");

        if (hist.loading()) {
            const std::string histStation = hist.currentStation();
            const std::string histLabel = hist.currentLabel();
            if (!histStation.empty() || !histLabel.empty())
                ImGui::Text("%s | %s", histStation.c_str(), histLabel.c_str());
            ImGui::TextColored(ImVec4(1, 0.8f, 0.2f, 1),
                               "Downloading: %d / %d frames",
                               hist.downloadedFrames(), hist.totalFrames());
            float prog = hist.totalFrames() > 0 ?
                         (float)hist.downloadedFrames() / hist.totalFrames() : 0;
            ImGui::ProgressBar(prog, ImVec2(-1, 14));
        } else if (hist.loaded() && hist.numFrames() > 0) {
            // Event name + current time
            const std::string histLabel = hist.currentLabel();
            const auto* fr = hist.frame(hist.currentFrame());
            ImGui::Text("%s  |  %s UTC",
                        histLabel.empty() ? "Archive Playback" : histLabel.c_str(),
                        fr ? fr->timestamp.c_str() : "--:--:--");

            // Play/pause + speed
            if (ImGui::Button(hist.playing() ? "Pause" : "Play", ImVec2(60, 20)))
                hist.togglePlay();
            ImGui::SameLine();
            float spd = hist.speed();
            ImGui::SetNextItemWidth(80);
            if (ImGui::SliderFloat("##spd", &spd, 1.0f, 15.0f, "%.0f fps"))
                hist.setSpeed(spd);

            // Timeline scrubber
            int frame = hist.currentFrame();
            ImGui::SetNextItemWidth(-1);
            if (ImGui::SliderInt("##frame", &frame, 0, hist.numFrames() - 1)) {
                hist.setFrame(frame);
                app.m_lastHistoricFrame = -1; // force re-upload
            }
        } else {
            const std::string histError = hist.lastError();
            if (!histError.empty())
                ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.4f, 1.0f), "%s", histError.c_str());
            else
                ImGui::TextDisabled("No archive frames loaded.");
        }

        ImGui::End();
    }

    if (!app.m_historicMode && !app.snapshotMode() && app.liveLoopEnabled()) {
        ImGui::SetNextWindowSize(ImVec2(640, 160), ImGuiCond_FirstUseEver);
        ImGui::Begin("Realtime Loop");

        if (app.liveLoopAvailableFrames() <= 0) {
            ImGui::TextDisabled("Waiting for live frames...");
        } else {
            std::string liveLoopLabel = app.liveLoopCurrentLabel();
            ImGui::Text("%s", liveLoopLabel.empty() ? "Realtime Loop" : liveLoopLabel.c_str());

            if (ImGui::Button(app.liveLoopPlaying() ? "Pause" : "Play", ImVec2(60, 20)))
                app.toggleLiveLoopPlayback();
            ImGui::SameLine();
            if (ImGui::Button("Live", ImVec2(60, 20)))
                app.setLiveLoopPlaybackFrame(app.liveLoopAvailableFrames() - 1);
            ImGui::SameLine();
            float loopSpeed = app.liveLoopSpeed();
            ImGui::SetNextItemWidth(90);
            if (ImGui::SliderFloat("##live_loop_spd", &loopSpeed, 1.0f, 15.0f, "%.0f fps"))
                app.setLiveLoopSpeed(loopSpeed);

            int liveFrame = app.liveLoopPlaybackFrame();
            ImGui::SetNextItemWidth(-1);
            if (ImGui::SliderInt("##live_loop_frame", &liveFrame, 0, app.liveLoopAvailableFrames() - 1))
                app.setLiveLoopPlaybackFrame(liveFrame);
        }

        ImGui::End();
    }

    // ── Station list (right panel, hide in historic mode) ──
    if (app.m_historicMode) goto skip_station_list;

    static char stationFilter[64] = "";
    static bool onlyReady = false;

    ImGui::SetNextWindowSize(ImVec2(420, 620), ImGuiCond_FirstUseEver);
    ImGui::Begin("Station Browser");

    ImGui::InputTextWithHint("##station_filter", "Filter ICAO, city, state", stationFilter, sizeof(stationFilter));
    ImGui::Checkbox("Only Ready", &onlyReady);
    ImGui::Separator();
    ImGui::BeginChild("station_list", ImVec2(0, 0), ImGuiChildFlags_None);

    for (int i = 0; i < (int)stations.size(); i++) {
        const auto& st = stations[i];
        if (onlyReady && !st.uploaded && !st.parsed) continue;

        std::string searchBlob = st.icao + " " + NEXRAD_STATIONS[i].name + " " + NEXRAD_STATIONS[i].state;
        if (!containsCaseInsensitive(searchBlob, stationFilter)) continue;

        ImVec4 color;
        if (!st.enabled)          color = ImVec4(0.42f, 0.42f, 0.46f, 1.0f);
        else if (st.rendered)     color = ImVec4(0.3f, 1.0f, 0.3f, 1.0f);
        else if (st.uploaded)     color = ImVec4(0.8f, 0.8f, 0.3f, 1.0f);
        else if (st.parsed)       color = ImVec4(0.3f, 0.7f, 1.0f, 1.0f);
        else if (st.downloading)  color = ImVec4(0.75f, 0.75f, 0.75f, 1.0f);
        else if (st.failed)       color = ImVec4(1.0f, 0.3f, 0.3f, 0.8f);
        else                      color = ImVec4(0.6f, 0.6f, 0.66f, 1.0f);

        ImGui::PushID(i);
        bool enabled = st.enabled;
        if (ImGui::Checkbox("##enabled", &enabled))
            app.setStationEnabled(i, enabled);
        ImGui::SameLine();

        std::string label = st.icao + "  " + NEXRAD_STATIONS[i].name;
        ImGui::PushStyleColor(ImGuiCol_Text, color);
        if (ImGui::Selectable(label.c_str(), i == app.activeStation()))
            app.selectStation(i, true, 200.0);
        ImGui::PopStyleColor();
        ImGui::PopID();

        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::Text("%s (%s)", NEXRAD_STATIONS[i].name, NEXRAD_STATIONS[i].state);
            ImGui::Text("%s", st.enabled ? "Enabled for live refresh" : "Disabled / idle");
            ImGui::Text("Lat: %.4f  Lon: %.4f", st.display_lat, st.display_lon);
            if (!st.latest_scan_utc.empty())
                ImGui::Text("Latest scan: %s", st.latest_scan_utc.c_str());
            if (st.failed) ImGui::TextColored(ImVec4(1, 0.3f, 0.3f, 1), "Error: %s", st.error.c_str());
            if (st.parsed) {
                ImGui::Text("Sweeps: %d", st.sweep_count);
                if (st.sweep_count > 0) {
                    ImGui::Text("Lowest elev: %.1f deg", st.lowest_elev);
                    ImGui::Text("Radials: %d", st.lowest_radials);
                }
                if (st.timings.decode_ms > 0.0f || st.timings.sweep_build_ms > 0.0f ||
                    st.timings.upload_ms > 0.0f) {
                    ImGui::Separator();
                    ImGui::Text("Decode %.1f ms  Build %.1f ms  Upload %.1f ms",
                                st.timings.decode_ms, st.timings.sweep_build_ms, st.timings.upload_ms);
                }
            }
            ImGui::EndTooltip();
        }
    }

    ImGui::EndChild();
    ImGui::End();
    skip_station_list:

    ImGui::SetNextWindowSize(ImVec2(360, 360), ImGuiCond_FirstUseEver);
    ImGui::Begin("Inspector");
    ImGui::Text("Cursor");
    ImGui::Separator();
    ImGui::Text("Lat: %.4f", app.cursorLat());
    ImGui::Text("Lon: %.4f", app.cursorLon());
    ImGui::Separator();
    ImGui::Text("Mode: %s",
                app.m_historicMode ? "Historic" :
                app.snapshotMode() ? "Snapshot" :
                app.mode3D() ? "3D Volume" :
                app.crossSection() ? "Cross Section" :
                app.showAll() ? "National Mosaic" : "Single Site");
    ImGui::Text("Product: %s (%s)",
                PRODUCT_INFO[app.activeProduct()].name,
                PRODUCT_INFO[app.activeProduct()].code);
    ImGui::Text("Threshold: %.1f %s",
                app.dbzMinThreshold(),
                app.activeProduct() == PROD_VEL ? "m/s" : "dBZ");
    ImGui::Text("Stations: %d enabled | %d loaded", app.enabledStationCount(), app.stationsLoaded());
    ImGui::Text("Downloads: %d", app.stationsDownloading());
    ImGui::Text("Alerts: %d", warningCount);
    const auto& mem = app.memoryTelemetry();
    ImGui::Text("Profile: %s", performanceProfileLabel(app.effectivePerformanceProfile()));
    ImGui::Text("VRAM: %s / %s (peak %s)",
                formatBytes(mem.gpu_used_bytes).c_str(),
                formatBytes(mem.gpu_total_bytes).c_str(),
                formatBytes(mem.gpu_peak_used_bytes).c_str());
    ImGui::Text("RAM: %s (peak %s)",
                formatBytes(mem.process_working_set_bytes).c_str(),
                formatBytes(mem.process_peak_working_set_bytes).c_str());
    ImGui::Text("Internal Render: %dx%d",
                mem.internal_render_width, mem.internal_render_height);

    int inspectorStation = app.activeStation();
    if (inspectorStation >= 0 && inspectorStation < (int)stations.size()) {
        const auto& st = stations[inspectorStation];
        ImGui::Separator();
        ImGui::Text("Active Station");
        ImGui::Separator();
        ImGui::Text("%s  %s, %s",
                    st.icao.c_str(),
                    NEXRAD_STATIONS[inspectorStation].name,
                    NEXRAD_STATIONS[inspectorStation].state);
        ImGui::SameLine();
        ImGui::TextColored(st.enabled ? ImVec4(0.35f, 1.0f, 0.45f, 1.0f)
                                      : ImVec4(0.7f, 0.7f, 0.74f, 1.0f),
                           st.enabled ? "[ENABLED]" : "[DISABLED]");
        ImGui::Text("Lat %.4f  Lon %.4f", st.display_lat, st.display_lon);
        if (!st.latest_scan_utc.empty())
            ImGui::Text("Latest scan: %s", st.latest_scan_utc.c_str());
        if (ImGui::Button(st.enabled ? "Disable Site" : "Enable Site", ImVec2(210, 24)))
            app.setStationEnabled(inspectorStation, !st.enabled);
        ImGui::Text("Sweeps: %d", st.sweep_count);
        ImGui::Text("TDS %d  Hail %d  Meso %d",
                    (int)st.detection.tds.size(),
                    (int)st.detection.hail.size(),
                    (int)st.detection.meso.size());
        if (st.timings.decode_ms > 0.0f || st.timings.sweep_build_ms > 0.0f ||
            st.timings.preprocess_ms > 0.0f || st.timings.detection_ms > 0.0f ||
            st.timings.upload_ms > 0.0f || st.timings.gpu_detect_ms > 0.0f) {
            ImGui::Text("Decode %.1f  Build %.1f  Pre %.1f  Detect %.1f  Upload %.1f ms",
                        st.timings.decode_ms,
                        st.timings.sweep_build_ms,
                        st.timings.preprocess_ms,
                        st.timings.detection_ms,
                        st.timings.upload_ms);
            if (st.timings.used_gpu_detect_stage) {
                ImGui::Text("Fast GPU detect: build %.1f  pre %.1f  detect %.1f ms",
                            st.timings.gpu_detect_build_ms,
                            st.timings.gpu_detect_preprocess_ms,
                            st.timings.gpu_detect_ms);
            }
            ImGui::Text("Build path: %s%s",
                        st.timings.used_gpu_sweep_build ? "GPU sweep build" : "CPU parse/build",
                        st.timings.used_gpu_detect_stage ? " + GPU preview detect" : "");
        }
        if (st.failed)
            ImGui::TextColored(ImVec4(1, 0.3f, 0.3f, 1), "Error: %s", st.error.c_str());
    }
    ImGui::End();

    ImGui::SetNextWindowSize(ImVec2(520, 220), ImGuiCond_FirstUseEver);
    ImGui::Begin("Warnings");
    if (ImGui::CollapsingHeader("Display Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Enable Alert Overlays", &app.m_warningOptions.enabled);
        ImGui::Checkbox("Warnings", &app.m_warningOptions.showWarnings);
        ImGui::SameLine();
        ImGui::Checkbox("Watches", &app.m_warningOptions.showWatches);
        ImGui::SameLine();
        ImGui::Checkbox("Statements", &app.m_warningOptions.showStatements);
        ImGui::Checkbox("Advisories", &app.m_warningOptions.showAdvisories);
        ImGui::SameLine();
        ImGui::Checkbox("Special Wx Statements", &app.m_warningOptions.showSpecialWeatherStatements);
        ImGui::Checkbox("Tornado", &app.m_warningOptions.showTornado);
        ImGui::SameLine();
        ImGui::Checkbox("Severe", &app.m_warningOptions.showSevere);
        ImGui::SameLine();
        ImGui::Checkbox("Fire", &app.m_warningOptions.showFire);
        ImGui::SameLine();
        ImGui::Checkbox("Flood", &app.m_warningOptions.showFlood);
        ImGui::SameLine();
        ImGui::Checkbox("Marine", &app.m_warningOptions.showMarine);
        ImGui::Checkbox("Show Fills", &app.m_warningOptions.fillPolygons);
        ImGui::SameLine();
        ImGui::Checkbox("Show Outlines", &app.m_warningOptions.outlinePolygons);
        ImGui::SliderFloat("Fill Opacity", &app.m_warningOptions.fillOpacity, 0.0f, 0.8f, "%.2f");
        ImGui::SliderFloat("Outline Scale", &app.m_warningOptions.outlineScale, 0.5f, 3.0f, "%.1f");
    }
    if (ImGui::CollapsingHeader("Alert Colors")) {
        editWarningColor("Tornado", app.m_warningOptions.tornadoColor);
        editWarningColor("Severe", app.m_warningOptions.severeColor);
        editWarningColor("Fire", app.m_warningOptions.fireColor);
        editWarningColor("Flood", app.m_warningOptions.floodColor);
        editWarningColor("Marine", app.m_warningOptions.marineColor);
        editWarningColor("Watch", app.m_warningOptions.watchColor);
        editWarningColor("Statement", app.m_warningOptions.statementColor);
        editWarningColor("Advisory", app.m_warningOptions.advisoryColor);
        editWarningColor("Other", app.m_warningOptions.otherColor);
    }

    if (warnings.empty()) {
        ImGui::TextDisabled(app.m_historicMode
            ? "No cached historic polygons yet for this frame."
            : "No active alert polygons are loaded.");
    } else {
        ImGui::BeginChild("warning_list", ImVec2(0, 0), ImGuiChildFlags_None);
        for (size_t i = 0; i < warnings.size(); i++) {
            const auto& warning = warnings[i];
            ImGui::PushStyleColor(ImGuiCol_Text, rgbaToImVec4(warning.color));
            std::string label = warning.event + "##warning_" + std::to_string(i);
            if (ImGui::Selectable(label.c_str(), false))
                centerOnWarning(app, warning);
            ImGui::PopStyleColor();
            ImGui::TextWrapped("%s", warning.headline.c_str());
            if (!warning.office.empty())
                ImGui::TextDisabled("%s | %s", warning.office.c_str(),
                                    warning.historic ? "Historic" : "Live");
            ImGui::Spacing();
        }
        ImGui::EndChild();
    }
    ImGui::End();

    // ── Station markers on map (hide in historic mode) ──────
    if (app.m_historicMode) goto skip_station_markers;
    {
        auto* dl = ImGui::GetBackgroundDrawList();
        int activeIdx = app.activeStation();

        for (int i = 0; i < (int)stations.size(); i++) {
            const auto& st = stations[i];

            // Convert lat/lon to screen pixel
            float px = (float)((st.display_lon - vp.center_lon) * vp.zoom + vp.width * 0.5);
            float py = (float)((vp.center_lat - st.display_lat) * vp.zoom + vp.height * 0.5);

            // Skip if off-screen
            if (px < -50 || px > vp.width + 50 || py < -50 || py > vp.height + 50)
                continue;

            bool isActive = (i == activeIdx);
            float boxW = 36, boxH = 14;

            // Background rectangle
            ImU32 bgCol;
            ImU32 borderCol;
            ImU32 textCol;
            if (!st.enabled) {
                bgCol = isActive ? IM_COL32(95, 95, 105, 220) : IM_COL32(52, 52, 58, 170);
                borderCol = isActive ? IM_COL32(185, 185, 195, 255) : IM_COL32(96, 96, 104, 190);
                textCol = IM_COL32(185, 185, 195, 230);
            } else {
                bgCol = isActive ? IM_COL32(0, 180, 80, 220) : IM_COL32(40, 40, 50, 180);
                borderCol = isActive ? IM_COL32(100, 255, 150, 255) : IM_COL32(80, 80, 100, 200);
                textCol = isActive ? IM_COL32(255, 255, 255, 255) : IM_COL32(180, 180, 200, 220);
            }

            ImVec2 tl(px - boxW * 0.5f, py - boxH * 0.5f);
            ImVec2 br(px + boxW * 0.5f, py + boxH * 0.5f);

            dl->AddRectFilled(tl, br, bgCol, 3.0f);
            dl->AddRect(tl, br, borderCol, 3.0f);

            // Station ICAO text
            const char* label = st.icao.c_str();
            ImVec2 textSize = ImGui::CalcTextSize(label);
            dl->AddText(ImVec2(px - textSize.x * 0.5f, py - textSize.y * 0.5f),
                        textCol, label);
        }
    }
    skip_station_markers:

    // ── NWS Warning Polygons ────────────────────────────────
    if (!warnings.empty() && app.m_warningOptions.enabled) {
        auto* wdl = ImGui::GetBackgroundDrawList();
        for (const auto& w : warnings) {
            if (w.lats.size() < 3) continue;
            std::vector<ImVec2> pts;
            pts.reserve(w.lats.size());
            bool anyOnScreen = false;
            for (int i = 0; i < (int)w.lats.size(); i++) {
                float sx = (float)((w.lons[i] - vp.center_lon) * vp.zoom + vp.width * 0.5);
                float sy = (float)((vp.center_lat - w.lats[i]) * vp.zoom + vp.height * 0.5);
                pts.push_back(ImVec2(sx, sy));
                if (sx > -100 && sx < vp.width + 100 && sy > -100 && sy < vp.height + 100)
                    anyOnScreen = true;
            }
            if (!anyOnScreen) continue;

            if (app.m_warningOptions.fillPolygons && pts.size() >= 3)
                wdl->AddConcavePolyFilled(pts.data(), (int)pts.size(),
                                          app.m_warningOptions.resolvedFillColor(w));
            if (app.m_warningOptions.outlinePolygons) {
                uint32_t outlineCol = (w.color & 0x00FFFFFFu) | 0xFF000000u;
                for (int i = 0; i < (int)pts.size(); i++) {
                    int j = (i + 1) % (int)pts.size();
                    wdl->AddLine(pts[i], pts[j], outlineCol, w.line_width);
                }
            }
        }
    }

    // ── Detection overlays (TDS, Hail, Meso) ─────────────────
    {
        auto* ddl = ImGui::GetBackgroundDrawList();
        int dsi = app.activeStation();
        if (dsi >= 0 && dsi < (int)stations.size()) {
            const auto& dst = stations[dsi];
            const auto& det = dst.detection;

            // TDS markers: white inverted triangles with red border
            if (app.m_showTDS && !det.tds.empty()) {
                for (auto& t : det.tds) {
                    float sx = (float)((t.lon - vp.center_lon) * vp.zoom + vp.width * 0.5);
                    float sy = (float)((vp.center_lat - t.lat) * vp.zoom + vp.height * 0.5);
                    if (sx < -20 || sx > vp.width+20 || sy < -20 || sy > vp.height+20) continue;
                    float sz = 6.0f;
                    ddl->AddTriangleFilled(
                        ImVec2(sx, sy + sz), ImVec2(sx - sz, sy - sz), ImVec2(sx + sz, sy - sz),
                        IM_COL32(255, 255, 255, 200));
                    ddl->AddTriangle(
                        ImVec2(sx, sy + sz), ImVec2(sx - sz, sy - sz), ImVec2(sx + sz, sy - sz),
                        IM_COL32(255, 0, 0, 255), 2.0f);
                }
            }

            // Hail markers: green/magenta circles with H
            if (app.m_showHail && !det.hail.empty()) {
                for (auto& h : det.hail) {
                    float sx = (float)((h.lon - vp.center_lon) * vp.zoom + vp.width * 0.5);
                    float sy = (float)((vp.center_lat - h.lat) * vp.zoom + vp.height * 0.5);
                    if (sx < -20 || sx > vp.width+20 || sy < -20 || sy > vp.height+20) continue;
                    float r = 5.0f;
                    ImU32 col = h.value > 10.0f ? IM_COL32(255, 50, 255, 220) :
                                                   IM_COL32(0, 255, 100, 200);
                    ddl->AddCircleFilled(ImVec2(sx, sy), r, col);
                    ddl->AddText(ImVec2(sx - 3, sy - 6), IM_COL32(0, 0, 0, 255), "H");
                }
            }

            // Mesocyclone markers: circles with rotation indicator
            if (app.m_showMeso && !det.meso.empty()) {
                for (auto& m : det.meso) {
                    float sx = (float)((m.lon - vp.center_lon) * vp.zoom + vp.width * 0.5);
                    float sy = (float)((vp.center_lat - m.lat) * vp.zoom + vp.height * 0.5);
                    if (sx < -20 || sx > vp.width+20 || sy < -20 || sy > vp.height+20) continue;
                    float r = m.shear > 30.0f ? 10.0f : 7.0f;
                    ImU32 col = m.shear > 30.0f ? IM_COL32(255, 0, 0, 255) :
                                                    IM_COL32(255, 255, 0, 255);
                    ddl->AddCircle(ImVec2(sx, sy), r, col, 12, 2.5f);
                    ddl->AddLine(ImVec2(sx + r, sy), ImVec2(sx + r - 3, sy - 3), col, 2.0f);
                    ddl->AddLine(ImVec2(sx + r, sy), ImVec2(sx + r + 1, sy - 4), col, 2.0f);
                }
            }
        }
    }

    // ── Cross-section line overlay ────────────────────���───────
    if (app.crossSection()) {
        auto* dl2 = ImGui::GetBackgroundDrawList();
        // Draw the cross-section line on the radar view
        float sx = (float)((app.xsStartLon() - vp.center_lon) * vp.zoom + vp.width * 0.5);
        float sy = (float)((vp.center_lat - app.xsStartLat()) * vp.zoom + vp.height * 0.5);
        float ex = (float)((app.xsEndLon() - vp.center_lon) * vp.zoom + vp.width * 0.5);
        float ey = (float)((vp.center_lat - app.xsEndLat()) * vp.zoom + vp.height * 0.5);

        dl2->AddLine(ImVec2(sx, sy), ImVec2(ex, ey), IM_COL32(255, 255, 0, 200), 3.0f);
        dl2->AddCircleFilled(ImVec2(sx, sy), 6, IM_COL32(255, 100, 100, 255));
        dl2->AddCircleFilled(ImVec2(ex, ey), 6, IM_COL32(100, 255, 100, 255));

        // Label
        float xsBottom = (float)(vp.height / 2);
        // Cross-section floating panel (book view)
        if (app.xsTexture().textureId() != 0 && app.xsWidth() > 0) {
            float panelW = (float)vp.width * 0.8f;
            float panelH = (float)app.xsHeight() + 40.0f;
            ImGui::SetNextWindowPos(ImVec2((float)vp.width * 0.1f,
                                           (float)vp.height - panelH - 10), ImGuiCond_Once);
            ImGui::SetNextWindowSize(ImVec2(panelW, panelH), ImGuiCond_Once);
            ImGui::Begin("Cross-Section Console", nullptr,
                         ImGuiWindowFlags_NoCollapse);

            ImVec2 avail = ImGui::GetContentRegionAvail();
            float imgW = avail.x, imgH = avail.y;

            ImGui::Image((ImTextureID)(uintptr_t)app.xsTexture().textureId(),
                         ImVec2(imgW, imgH));

            // Altitude labels (kft like GR2Analyst)
            ImVec2 imgPos = ImGui::GetItemRectMin();
            auto* wdl = ImGui::GetWindowDrawList();
            for (int kft = 0; kft <= 45; kft += 5) {
                float alt_km = (float)kft * 0.3048f; // kft to km
                float frac = alt_km / 15.0f; // 15km max
                if (frac > 1.0f) break;
                float yy = imgPos.y + imgH * (1.0f - frac);
                char altLabel[16];
                snprintf(altLabel, sizeof(altLabel), "%d kft", kft);
                wdl->AddText(ImVec2(imgPos.x + 4, yy - 7),
                             IM_COL32(200, 200, 255, 200), altLabel);
                wdl->AddLine(ImVec2(imgPos.x + 40, yy),
                             ImVec2(imgPos.x + imgW, yy),
                             IM_COL32(100, 100, 140, 60), 1.0f);
            }

            ImGui::End();
        }
    }

    // ── Keyboard shortcuts ──────────────────────────────────
    if (!ImGui::GetIO().WantCaptureKeyboard) {
        // Number keys: direct product select
        for (int i = 0; i < (int)Product::COUNT; i++) {
            if (ImGui::IsKeyPressed((ImGuiKey)(ImGuiKey_1 + i)))
                app.setProduct(i);
        }
        // Arrow keys: left/right = product, up/down = tilt
        if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow))  app.prevProduct();
        if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) app.nextProduct();
        if (ImGui::IsKeyPressed(ImGuiKey_UpArrow))    app.nextTilt();
        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow))  app.prevTilt();
        // V = 3D volume, X = cross-section, A = toggle show all
        if (ImGui::IsKeyPressed(ImGuiKey_V)) app.toggle3D();
        if (ImGui::IsKeyPressed(ImGuiKey_X)) app.toggleCrossSection();
        if (ImGui::IsKeyPressed(ImGuiKey_A)) app.toggleShowAll();
        if (ImGui::IsKeyPressed(ImGuiKey_R)) app.refreshData();
        if (ImGui::IsKeyPressed(ImGuiKey_S)) app.toggleSRV();
        if (ImGui::IsKeyPressed(ImGuiKey_Home)) resetConusView(app);
        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) app.setAutoTrackStation(true);
        if (app.m_historicMode && ImGui::IsKeyPressed(ImGuiKey_Space)) app.m_historic.togglePlay();
        else if (!app.m_historicMode && app.liveLoopEnabled() && ImGui::IsKeyPressed(ImGuiKey_Space))
            app.toggleLiveLoopPlayback();
    }
}

void shutdown() {
}

} // namespace ui
