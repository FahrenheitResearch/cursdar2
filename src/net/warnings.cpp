#include "warnings.h"
#include "downloader.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <optional>
#include <string>
#include <unordered_map>

using json = nlohmann::json;

namespace {

std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

bool containsInsensitive(const std::string& haystack, const char* needle) {
    if (!needle || !needle[0]) return false;
    return toLower(haystack).find(toLower(needle)) != std::string::npos;
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

int64_t parseIsoUtc(const std::string& iso) {
    if (iso.size() < 16) return 0;
    int year = 0, month = 0, day = 0, hh = 0, mm = 0, ss = 0;
    if (std::sscanf(iso.c_str(), "%4d-%2d-%2dT%2d:%2d:%2d",
                    &year, &month, &day, &hh, &mm, &ss) < 5) {
        return 0;
    }
    return makeUtcEpoch(year, month, day, hh, mm, ss);
}

int64_t parseCompactUtc(const std::string& value) {
    if (value.size() < 12) return 0;
    int year = std::stoi(value.substr(0, 4));
    int month = std::stoi(value.substr(4, 2));
    int day = std::stoi(value.substr(6, 2));
    int hh = std::stoi(value.substr(8, 2));
    int mm = std::stoi(value.substr(10, 2));
    return makeUtcEpoch(year, month, day, hh, mm, 0);
}

std::string formatIsoUtc(int64_t epoch) {
    if (epoch <= 0) return {};
    std::time_t t = static_cast<std::time_t>(epoch);
    std::tm tm = {};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02dZ",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec);
    return buf;
}

std::string normalizeTimestampKey(const std::string& isoTimestamp) {
    const int64_t epoch = parseIsoUtc(isoTimestamp);
    return formatIsoUtc(epoch);
}

std::string nextUtcDay(const std::string& yyyy_mm_dd) {
    if (yyyy_mm_dd.size() < 10) return {};
    int year = std::stoi(yyyy_mm_dd.substr(0, 4));
    int month = std::stoi(yyyy_mm_dd.substr(5, 2));
    int day = std::stoi(yyyy_mm_dd.substr(8, 2));
    return formatIsoUtc(makeUtcEpoch(year, month, day, 0, 0, 0) + 86400).substr(0, 10);
}

uint16_t readLe16(const uint8_t* data) {
    return uint16_t(data[0]) | (uint16_t(data[1]) << 8);
}

uint32_t readLe32(const uint8_t* data) {
    return uint32_t(data[0]) |
           (uint32_t(data[1]) << 8) |
           (uint32_t(data[2]) << 16) |
           (uint32_t(data[3]) << 24);
}

int32_t readLeI32(const uint8_t* data) {
    return static_cast<int32_t>(readLe32(data));
}

uint32_t readBe32(const uint8_t* data) {
    return (uint32_t(data[0]) << 24) |
           (uint32_t(data[1]) << 16) |
           (uint32_t(data[2]) << 8) |
           uint32_t(data[3]);
}

double readLeDouble(const uint8_t* data) {
    double value = 0.0;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

std::string trimField(std::string value) {
    auto notSpace = [](unsigned char c) { return !std::isspace(c); };
    auto begin = std::find_if(value.begin(), value.end(), notSpace);
    auto end = std::find_if(value.rbegin(), value.rend(), notSpace).base();
    if (begin >= end) return {};
    value.assign(begin, end);
    while (!value.empty() && value.back() == '\0')
        value.pop_back();
    return value;
}

struct DbfField {
    std::string name;
    uint8_t length = 0;
    size_t offset = 0;
};

using DbfRow = std::unordered_map<std::string, std::string>;

std::vector<DbfRow> parseDbfRows(const std::vector<uint8_t>& data) {
    std::vector<DbfRow> rows;
    if (data.size() < 32) return rows;

    const uint32_t numRecords = readLe32(data.data() + 4);
    const uint16_t headerLength = readLe16(data.data() + 8);
    const uint16_t recordLength = readLe16(data.data() + 10);
    if (headerLength >= data.size() || recordLength == 0) return rows;

    std::vector<DbfField> fields;
    fields.reserve(24);
    size_t offset = 1;
    for (size_t pos = 32; pos + 32 <= data.size(); pos += 32) {
        if (data[pos] == 0x0D) break;
        DbfField field;
        field.name.assign(reinterpret_cast<const char*>(data.data() + pos), 11);
        field.name = trimField(field.name);
        field.length = data[pos + 16];
        field.offset = offset;
        offset += field.length;
        if (!field.name.empty())
            fields.push_back(std::move(field));
    }

    rows.reserve(numRecords);
    size_t recordPos = headerLength;
    for (uint32_t i = 0; i < numRecords && recordPos + recordLength <= data.size(); i++) {
        const uint8_t* record = data.data() + recordPos;
        recordPos += recordLength;
        if (record[0] == 0x2A) continue;

        DbfRow row;
        row.reserve(fields.size());
        for (const auto& field : fields) {
            if (field.offset + field.length > recordLength) continue;
            std::string value(reinterpret_cast<const char*>(record + field.offset), field.length);
            row[field.name] = trimField(value);
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

struct ParsedPolygon {
    std::vector<float> lats;
    std::vector<float> lons;
};

std::vector<ParsedPolygon> parseShapefilePolygons(const std::vector<uint8_t>& data) {
    std::vector<ParsedPolygon> polygons;
    if (data.size() < 100) return polygons;

    size_t pos = 100;
    while (pos + 8 <= data.size()) {
        const uint32_t contentWords = readBe32(data.data() + pos + 4);
        pos += 8;
        const size_t contentBytes = size_t(contentWords) * 2;
        if (pos + contentBytes > data.size()) break;

        const uint8_t* record = data.data() + pos;
        pos += contentBytes;
        if (contentBytes < 4) continue;

        const int32_t shapeType = readLeI32(record);
        if (shapeType == 0) {
            polygons.push_back({});
            continue;
        }
        if (shapeType != 5) continue;
        if (contentBytes < 44) {
            polygons.push_back({});
            continue;
        }

        const int32_t numParts = readLeI32(record + 36);
        const int32_t numPoints = readLeI32(record + 40);
        if (numParts <= 0 || numPoints <= 0) {
            polygons.push_back({});
            continue;
        }

        const size_t partsOffset = 44;
        const size_t pointsOffset = partsOffset + size_t(numParts) * 4;
        const size_t neededBytes = pointsOffset + size_t(numPoints) * 16;
        if (neededBytes > contentBytes) {
            polygons.push_back({});
            continue;
        }

        std::vector<int32_t> parts;
        parts.reserve(numParts + 1);
        for (int32_t i = 0; i < numParts; i++)
            parts.push_back(readLeI32(record + partsOffset + size_t(i) * 4));
        parts.push_back(numPoints);

        int bestPart = -1;
        int bestCount = 0;
        for (int32_t i = 0; i < numParts; i++) {
            const int32_t start = parts[i];
            const int32_t end = parts[i + 1];
            const int32_t count = end - start;
            if (start < 0 || end > numPoints || count < 3) continue;
            if (count > bestCount) {
                bestCount = count;
                bestPart = i;
            }
        }

        ParsedPolygon polygon;
        if (bestPart >= 0) {
            const int32_t start = parts[bestPart];
            const int32_t end = parts[bestPart + 1];
            polygon.lats.reserve(end - start);
            polygon.lons.reserve(end - start);
            for (int32_t i = start; i < end; i++) {
                const uint8_t* point = record + pointsOffset + size_t(i) * 16;
                polygon.lons.push_back(static_cast<float>(readLeDouble(point)));
                polygon.lats.push_back(static_cast<float>(readLeDouble(point + 8)));
            }
        }
        polygons.push_back(std::move(polygon));
    }

    return polygons;
}

std::string significanceLabel(const std::string& significance) {
    if (significance == "W") return "Warning";
    if (significance == "A") return "Watch";
    if (significance == "Y") return "Advisory";
    if (significance == "S") return "Statement";
    if (significance == "O") return "Outlook";
    if (significance == "N") return "Synopsis";
    if (significance == "F") return "Forecast";
    return {};
}

std::string phenomenaLabel(const std::string& phenomena) {
    static const std::unordered_map<std::string, std::string> kPhenomena = {
        {"AF", "Ashfall"},
        {"AS", "Air Stagnation"},
        {"BH", "Beach Hazard"},
        {"BS", "Blowing Snow"},
        {"BW", "Brisk Wind"},
        {"BZ", "Blizzard"},
        {"CF", "Coastal Flood"},
        {"CW", "Cold Weather"},
        {"DF", "Debris Flow"},
        {"DS", "Dust Storm"},
        {"DU", "Blowing Dust"},
        {"EC", "Extreme Cold"},
        {"EH", "Excessive Heat"},
        {"EW", "Extreme Wind"},
        {"FA", "Flood"},
        {"FF", "Flash Flood"},
        {"FG", "Dense Fog"},
        {"FL", "Flood"},
        {"FR", "Frost"},
        {"FW", "Fire Weather"},
        {"FZ", "Freeze"},
        {"GL", "Gale"},
        {"HF", "Hurricane Force Wind"},
        {"HI", "Inland Hurricane"},
        {"HS", "Heavy Snow"},
        {"HT", "Heat"},
        {"HU", "Hurricane"},
        {"HW", "High Wind"},
        {"HY", "Hydrologic"},
        {"HZ", "Hard Freeze"},
        {"IP", "Sleet"},
        {"IS", "Ice Storm"},
        {"LB", "Lake Effect Snow and Blowing Snow"},
        {"LE", "Lake Effect Snow"},
        {"LO", "Low Water"},
        {"LS", "Lakeshore Flood"},
        {"LW", "Lake Wind"},
        {"MA", "Marine"},
        {"MF", "Marine Dense Fog"},
        {"MH", "Marine Ashfall"},
        {"MS", "Marine Dense Smoke"},
        {"RB", "Small Craft for Rough Bar"},
        {"RP", "Rip Current"},
        {"SB", "Snow and Blowing Snow"},
        {"SC", "Small Craft"},
        {"SE", "Hazardous Seas"},
        {"SI", "Small Craft for Winds"},
        {"SM", "Dense Smoke"},
        {"SN", "Snow"},
        {"SQ", "Snow Squall"},
        {"SR", "Storm"},
        {"SS", "Storm Surge"},
        {"SU", "High Surf"},
        {"SV", "Severe Thunderstorm"},
        {"SW", "Small Craft for Hazardous Seas"},
        {"TI", "Inland Tropical Storm"},
        {"TO", "Tornado"},
        {"TR", "Tropical Storm"},
        {"TS", "Tsunami"},
        {"TY", "Typhoon"},
        {"WC", "Wind Chill"},
        {"WI", "Wind"},
        {"WS", "Winter Storm"},
        {"WW", "Winter Weather"},
        {"XH", "Extreme Heat"},
        {"ZF", "Freezing Fog"},
        {"ZR", "Freezing Rain"},
    };

    auto it = kPhenomena.find(phenomena);
    if (it == kPhenomena.end()) return phenomena;
    return it->second;
}

std::string liveEventName(const DbfRow& row) {
    const auto phenomIt = row.find("PHENOM");
    const auto sigIt = row.find("SIG");
    const std::string phenom = (phenomIt != row.end()) ? phenomIt->second : "";
    const std::string sig = (sigIt != row.end()) ? sigIt->second : "";
    if (phenom == "FW" && sig == "W") return "Red Flag Warning";
    if (phenom == "FW" && sig == "A") return "Fire Weather Watch";
    if (phenom == "FA" && sig == "W") return "Flood Warning";

    const std::string base = phenomenaLabel(phenom);
    const std::string suffix = significanceLabel(sig);
    if (base.empty()) return {};
    if (suffix.empty()) return base;
    return base + " " + suffix;
}

WarningGroup classifyWarningGroup(const std::string& event, const std::string& phenomena = {}) {
    const std::string key = toLower(event + " " + phenomena);
    if (key.find("tornado") != std::string::npos || phenomena == "TO")
        return WarningGroup::Tornado;
    if (key.find("severe") != std::string::npos || phenomena == "SV")
        return WarningGroup::Severe;
    if (key.find("red flag") != std::string::npos ||
        key.find("fire weather") != std::string::npos ||
        phenomena == "FW")
        return WarningGroup::Fire;
    if (key.find("flood") != std::string::npos || phenomena == "FF" || phenomena == "FA" || phenomena == "FL")
        return WarningGroup::Flood;
    if (key.find("marine") != std::string::npos ||
        key.find("small craft") != std::string::npos ||
        key.find("rip current") != std::string::npos ||
        key.find("surf") != std::string::npos ||
        phenomena == "MA" || phenomena == "SC" || phenomena == "BW" ||
        phenomena == "GL" || phenomena == "HF" || phenomena == "LO" ||
        phenomena == "MH" || phenomena == "MS" || phenomena == "RB" ||
        phenomena == "RP" || phenomena == "SE" || phenomena == "SI" ||
        phenomena == "SU" || phenomena == "SW")
        return WarningGroup::Marine;
    return WarningGroup::Other;
}

uint32_t defaultColorForGroup(WarningGroup group) {
    switch (group) {
        case WarningGroup::Tornado: return 0xFF3030FFu;
        case WarningGroup::Severe:  return 0xFF7F00FFu;
        case WarningGroup::Fire:    return 0xFF2060FFu;
        case WarningGroup::Flood:   return 0xFF1C9B3Cu;
        case WarningGroup::Marine:  return 0xFF46B4FFu;
        default:                    return 0xFFD8C8A0u;
    }
}

bool isWatchEvent(const std::string& event) {
    return containsInsensitive(event, "watch");
}

bool isStatementEvent(const std::string& event) {
    return containsInsensitive(event, "statement");
}

bool isAdvisoryEvent(const std::string& event) {
    return containsInsensitive(event, "advisory");
}

bool isWarningEvent(const std::string& event) {
    return containsInsensitive(event, "warning");
}

float defaultLineWidth(const std::string& event, bool emergency) {
    if (emergency) return 4.0f;
    if (containsInsensitive(event, "tornado")) return 3.0f;
    if (isWatchEvent(event)) return 2.5f;
    if (isStatementEvent(event) || isAdvisoryEvent(event)) return 1.5f;
    return 2.0f;
}

bool extractLargestRing(const json& geometry, std::vector<float>& lats, std::vector<float>& lons) {
    if (!geometry.is_object()) return false;
    const std::string type = geometry.value("type", "");
    const json* bestRing = nullptr;
    size_t bestSize = 0;

    if (type == "Polygon") {
        if (geometry.contains("coordinates") && geometry["coordinates"].is_array()) {
            for (const auto& ring : geometry["coordinates"]) {
                if (ring.is_array() && ring.size() > bestSize) {
                    bestRing = &ring;
                    bestSize = ring.size();
                }
            }
        }
    } else if (type == "MultiPolygon") {
        if (geometry.contains("coordinates") && geometry["coordinates"].is_array()) {
            for (const auto& polygon : geometry["coordinates"]) {
                if (!polygon.is_array()) continue;
                for (const auto& ring : polygon) {
                    if (ring.is_array() && ring.size() > bestSize) {
                        bestRing = &ring;
                        bestSize = ring.size();
                    }
                }
            }
        }
    } else {
        return false;
    }

    if (!bestRing || bestSize < 3) return false;
    lats.clear();
    lons.clear();
    lats.reserve(bestSize);
    lons.reserve(bestSize);
    for (const auto& point : *bestRing) {
        if (!point.is_array() || point.size() < 2) continue;
        lons.push_back(point[0].get<float>());
        lats.push_back(point[1].get<float>());
    }
    return lats.size() >= 3 && lats.size() == lons.size();
}

std::optional<WarningPolygon> parseNwsWarning(const json& feature) {
    if (!feature.is_object() || !feature.contains("properties") || !feature.contains("geometry"))
        return std::nullopt;
    const json& props = feature["properties"];

    WarningPolygon warning;
    warning.id = feature.value("id", props.value("id", ""));
    warning.event = props.value("event", "");
    if (warning.event.empty()) return std::nullopt;
    warning.headline = props.value("headline", warning.event);
    warning.office = props.value("senderName", props.value("areaDesc", ""));
    warning.status = props.value("status", "");
    warning.source = "NWS Active Alerts";
    warning.issue_time = props.value("sent", props.value("effective", props.value("onset", "")));
    warning.expire_time = props.value("ends", props.value("expires", ""));
    warning.emergency = containsInsensitive(warning.event, "emergency") ||
                        toLower(props.value("severity", "")) == "extreme";
    warning.group = classifyWarningGroup(warning.event);
    warning.color = defaultColorForGroup(warning.group);
    warning.line_width = defaultLineWidth(warning.event, warning.emergency);

    if (!extractLargestRing(feature["geometry"], warning.lats, warning.lons))
        return std::nullopt;
    return warning;
}

std::optional<WarningPolygon> parseIemLiveWarning(const DbfRow& row, const ParsedPolygon& polygon) {
    if (polygon.lats.size() < 3 || polygon.lats.size() != polygon.lons.size())
        return std::nullopt;

    WarningPolygon warning;
    const std::string phenom = row.find("PHENOM") != row.end() ? row.at("PHENOM") : "";
    const std::string significance = row.find("SIG") != row.end() ? row.at("SIG") : "";
    const std::string office = row.find("WFO") != row.end() ? row.at("WFO") : "";
    const std::string etn = row.find("ETN") != row.end() ? row.at("ETN") : "";
    const std::string ugc = row.find("NWS_UGC") != row.end() ? row.at("NWS_UGC") : "";

    warning.id = office + "." + phenom + "." + significance + "." + etn + "." + ugc;
    warning.event = liveEventName(row);
    if (warning.event.empty()) return std::nullopt;
    warning.headline = warning.event;
    if (!ugc.empty())
        warning.headline += " [" + ugc + "]";
    warning.office = office;
    warning.status = row.find("STATUS") != row.end() ? row.at("STATUS") : "";
    warning.source = "IEM Current WWA";
    warning.issue_time = formatIsoUtc(parseCompactUtc(row.find("ISSUED") != row.end() ? row.at("ISSUED") : ""));
    warning.expire_time = formatIsoUtc(parseCompactUtc(row.find("EXPIRED") != row.end() ? row.at("EXPIRED") : ""));
    warning.emergency = false;
    warning.group = classifyWarningGroup(warning.event, phenom);
    warning.color = defaultColorForGroup(warning.group);
    warning.line_width = defaultLineWidth(warning.event, warning.emergency);
    warning.lats = polygon.lats;
    warning.lons = polygon.lons;
    return warning;
}

std::optional<WarningPolygon> parseIemWarning(const json& feature) {
    if (!feature.is_object() || !feature.contains("properties") || !feature.contains("geometry"))
        return std::nullopt;
    const json& props = feature["properties"];

    WarningPolygon warning;
    warning.id = feature.value("id", props.value("product_id", ""));
    warning.event = props.value("ps", props.value("event", ""));
    if (warning.event.empty()) return std::nullopt;
    warning.headline = warning.event;
    if (props.contains("eventid"))
        warning.headline += " " + std::to_string(props["eventid"].get<int>());
    warning.office = props.value("wfo", "");
    warning.status = props.value("status", "");
    warning.source = "IEM Historic Warnings";
    warning.issue_time = props.value("issue", props.value("polygon_begin", ""));
    warning.expire_time = props.value("expire_utc", props.value("expire", ""));
    warning.historic = true;
    warning.emergency = props.value("is_emergency", false) || props.value("is_pds", false);
    warning.group = classifyWarningGroup(warning.event, props.value("phenomena", ""));
    warning.color = defaultColorForGroup(warning.group);
    warning.line_width = defaultLineWidth(warning.event, warning.emergency);

    if (!extractLargestRing(feature["geometry"], warning.lats, warning.lons))
        return std::nullopt;
    return warning;
}

std::optional<WarningPolygon> parseSpcWatch(const json& feature) {
    if (!feature.is_object() || !feature.contains("properties") || !feature.contains("geometry"))
        return std::nullopt;
    const json& props = feature["properties"];
    const std::string type = props.value("TYPE", "");

    WarningPolygon warning;
    warning.id = "SPC-" + std::to_string(props.value("NUM", 0));
    warning.event = (type == "TOR") ? "Tornado Watch" : "Severe Thunderstorm Watch";
    warning.headline = warning.event + " " + std::to_string(props.value("NUM", 0));
    if (props.value("IS_PDS", false))
        warning.headline = "PDS " + warning.headline;
    warning.office = "SPC";
    warning.status = "ACTIVE";
    warning.source = "IEM SPC Watches";
    warning.issue_time = formatIsoUtc(parseCompactUtc(props.value("ISSUE", "")));
    warning.expire_time = formatIsoUtc(parseCompactUtc(props.value("EXPIRE", "")));
    warning.historic = true;
    warning.emergency = props.value("IS_PDS", false);
    warning.group = (type == "TOR") ? WarningGroup::Tornado : WarningGroup::Severe;
    warning.color = defaultColorForGroup(warning.group);
    warning.line_width = defaultLineWidth(warning.event, warning.emergency);

    if (!extractLargestRing(feature["geometry"], warning.lats, warning.lons))
        return std::nullopt;
    return warning;
}

std::vector<WarningPolygon> parseFeatureCollection(const std::vector<uint8_t>& data,
                                                   const char* collectionType) {
    std::vector<WarningPolygon> warnings;
    if (data.empty()) return warnings;

    const auto parsed = json::parse(data.begin(), data.end(), nullptr, false);
    if (parsed.is_discarded() || !parsed.contains("features") || !parsed["features"].is_array())
        return warnings;

    for (const auto& feature : parsed["features"]) {
        std::optional<WarningPolygon> warning;
        if (std::strcmp(collectionType, "nws") == 0)
            warning = parseNwsWarning(feature);
        else if (std::strcmp(collectionType, "iem") == 0)
            warning = parseIemWarning(feature);
        else
            warning = parseSpcWatch(feature);

        if (warning) warnings.push_back(std::move(*warning));
    }
    return warnings;
}

std::vector<WarningPolygon> parseLiveWwaShapefile(const std::vector<uint8_t>& shpData,
                                                  const std::vector<uint8_t>& dbfData) {
    std::vector<WarningPolygon> warnings;
    auto polygons = parseShapefilePolygons(shpData);
    auto rows = parseDbfRows(dbfData);
    const size_t count = std::min(polygons.size(), rows.size());
    warnings.reserve(count);
    for (size_t i = 0; i < count; i++) {
        auto warning = parseIemLiveWarning(rows[i], polygons[i]);
        if (warning)
            warnings.push_back(std::move(*warning));
    }
    return warnings;
}

} // namespace

bool WarningRenderOptions::allows(const WarningPolygon& warning) const {
    if (!enabled) return false;

    if (isWatchEvent(warning.event)) {
        return showWatches;
    } else if (containsInsensitive(warning.event, "special weather statement")) {
        return showSpecialWeatherStatements && showStatements;
    } else if (isStatementEvent(warning.event)) {
        return showStatements;
    } else if (isAdvisoryEvent(warning.event)) {
        return showAdvisories;
    } else if (isWarningEvent(warning.event)) {
        if (!showWarnings) return false;
        switch (warning.group) {
            case WarningGroup::Tornado: return showTornado;
            case WarningGroup::Severe:  return showSevere;
            case WarningGroup::Fire:    return showFire;
            case WarningGroup::Flood:   return showFlood;
            case WarningGroup::Marine:  return showMarine;
            default:                    return showOther;
        }
    } else if (!showOther) {
        return false;
    }
    return showOther;
}

uint32_t WarningRenderOptions::resolvedColor(const WarningPolygon& warning) const {
    if (isWatchEvent(warning.event)) return watchColor;
    if (containsInsensitive(warning.event, "special weather statement")) return statementColor;
    if (isStatementEvent(warning.event)) return statementColor;
    if (isAdvisoryEvent(warning.event)) return advisoryColor;
    switch (warning.group) {
        case WarningGroup::Tornado: return tornadoColor;
        case WarningGroup::Severe:  return severeColor;
        case WarningGroup::Fire:    return fireColor;
        case WarningGroup::Flood:   return floodColor;
        case WarningGroup::Marine:  return marineColor;
        default:                    return otherColor;
    }
}

uint32_t WarningRenderOptions::resolvedFillColor(const WarningPolygon& warning) const {
    const uint32_t rgb = resolvedColor(warning) & 0x00FFFFFFu;
    int alpha = (int)(std::clamp(fillOpacity, 0.0f, 1.0f) * 255.0f);
    return rgb | (uint32_t(alpha) << 24);
}

float WarningRenderOptions::resolvedLineWidth(const WarningPolygon& warning) const {
    return std::max(1.0f, warning.line_width * std::max(0.5f, outlineScale));
}

WarningFetcher::~WarningFetcher() {
    stop();
}

std::vector<WarningPolygon> WarningFetcher::getWarnings() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_warnings;
}

void WarningFetcher::clearHistoric() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_historicWarnings.clear();
    m_historicWatchDays.clear();
    m_historicInFlight.clear();
}

std::vector<WarningPolygon> WarningFetcher::getHistoricWarnings(const std::string& isoTimestamp) const {
    const std::string key = normalizeTimestampKey(isoTimestamp);
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_historicWarnings.find(key);
    if (it == m_historicWarnings.end()) return {};
    return it->second;
}

void WarningFetcher::fetchLiveOnce() {
    try {
        std::vector<WarningPolygon> warnings;

        auto shpResult = Downloader::httpGet(
            "www.mesonet.agron.iastate.edu",
            "/data/gis/shape/4326/us/current_ww.shp");
        auto dbfResult = Downloader::httpGet(
            "www.mesonet.agron.iastate.edu",
            "/data/gis/shape/4326/us/current_ww.dbf");
        if (shpResult.success && dbfResult.success) {
            warnings = parseLiveWwaShapefile(shpResult.data, dbfResult.data);
        }

        if (warnings.empty()) {
            auto result = Downloader::httpGet(
                "api.weather.gov",
                "/alerts/active?status=actual&message_type=alert,update");
            if (!result.success) return;
            warnings = parseFeatureCollection(result.data, "nws");
        }

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_warnings = std::move(warnings);
        }
    } catch (const std::exception& e) {
        std::printf("Warning fetch failed: %s\n", e.what());
    } catch (...) {
        std::printf("Warning fetch failed with unknown exception\n");
    }
}

void WarningFetcher::fetchHistoricSnapshotWorker(std::string isoTimestamp) {
    std::vector<WarningPolygon> combined;
    try {
        const int64_t snapshotEpoch = parseIsoUtc(isoTimestamp);

        auto sbwResult = Downloader::httpGet(
            "mesonet.agron.iastate.edu",
            "/geojson/sbw.geojson?ts=" + isoTimestamp);
        if (sbwResult.success) {
            auto sbwWarnings = parseFeatureCollection(sbwResult.data, "iem");
            combined.insert(combined.end(), sbwWarnings.begin(), sbwWarnings.end());
        }

        const std::string dayKey = isoTimestamp.substr(0, 10);
        std::vector<WarningPolygon> dayWatches;
        bool needFetchDay = false;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            auto it = m_historicWatchDays.find(dayKey);
            if (it != m_historicWatchDays.end()) {
                dayWatches = it->second;
            } else {
                needFetchDay = true;
            }
        }

        if (needFetchDay) {
            const std::string nextDay = nextUtcDay(dayKey);
            const std::string path =
                "/cgi-bin/request/gis/spc_watch.py?sts=" + dayKey +
                "T00:00:00Z&ets=" + nextDay + "T00:00:00Z&format=geojson";
            auto watchResult = Downloader::httpGet("mesonet.agron.iastate.edu", path);
            if (watchResult.success) {
                dayWatches = parseFeatureCollection(watchResult.data, "spc");
                std::lock_guard<std::mutex> lock(m_mutex);
                m_historicWatchDays[dayKey] = dayWatches;
            }
        }

        for (const auto& watch : dayWatches) {
            const int64_t issue = parseIsoUtc(watch.issue_time);
            const int64_t expire = parseIsoUtc(watch.expire_time);
            if (snapshotEpoch <= 0) continue;
            if (issue > 0 && snapshotEpoch < issue) continue;
            if (expire > 0 && snapshotEpoch > expire) continue;
            combined.push_back(watch);
        }
    } catch (const std::exception& e) {
        std::printf("Historic warning fetch failed: %s\n", e.what());
    } catch (...) {
        std::printf("Historic warning fetch failed with unknown exception\n");
    }

    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_historicWarnings[isoTimestamp] = std::move(combined);
        m_historicInFlight.erase(isoTimestamp);
    }
}

void WarningFetcher::requestHistoricSnapshot(const std::string& isoTimestamp) {
    const std::string key = normalizeTimestampKey(isoTimestamp);
    if (key.empty()) return;

    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_historicWarnings.find(key) != m_historicWarnings.end()) return;
        if (m_historicInFlight.find(key) != m_historicInFlight.end()) return;
        m_historicInFlight.insert(key);
    }

    std::lock_guard<std::mutex> lock(m_historicThreadMutex);
    m_historicThreads.emplace_back([this, key]() { fetchHistoricSnapshotWorker(key); });
}

void WarningFetcher::startPolling() {
    stop();
    m_running = true;

    m_thread = std::thread([this]() {
        constexpr auto kLiveWarningPollInterval = std::chrono::seconds(20);
        while (m_running.load()) {
            fetchLiveOnce();
            std::unique_lock<std::mutex> lock(m_pollMutex);
            m_pollCv.wait_for(lock, kLiveWarningPollInterval,
                              [this]() { return !m_running.load(); });
        }
    });
}

void WarningFetcher::stop() {
    m_running = false;
    m_pollCv.notify_all();
    if (m_thread.joinable() && m_thread.get_id() != std::this_thread::get_id())
        m_thread.join();

    std::lock_guard<std::mutex> lock(m_historicThreadMutex);
    for (auto& thread : m_historicThreads) {
        if (thread.joinable() && thread.get_id() != std::this_thread::get_id())
            thread.join();
    }
    m_historicThreads.clear();
}
