#pragma once
#include "nexrad/stations.h"
#include <string>
#include <vector>
#include <ctime>
#include <algorithm>
#include <cctype>
#include <sstream>

// AWS S3 NEXRAD Level 2 bucket
constexpr const char* NEXRAD_BUCKET = "unidata-nexrad-level2";
constexpr const char* NEXRAD_HOST = "unidata-nexrad-level2.s3.amazonaws.com";
constexpr const char* IEM_LEVEL2_HOST = "mesonet-nexrad.agron.iastate.edu";

struct NexradFile {
    std::string key;
    std::string url;
    size_t      size;
};

inline const char* stationFeedCode(const StationInfo& station) {
    return (station.feed_code && station.feed_code[0] != '\0')
        ? station.feed_code
        : station.icao;
}

inline const char* radarDataHost(const StationInfo& station) {
    switch (station.feed) {
        case RadarFeedKind::IemLevel2RawDirList: return IEM_LEVEL2_HOST;
        case RadarFeedKind::AwsS3DatePartitioned:
        default:
            return NEXRAD_HOST;
    }
}

inline bool radarFeedUsesDatePartitionedListing(const StationInfo& station) {
    return station.feed == RadarFeedKind::AwsS3DatePartitioned;
}

// Build the S3 list URL for a station on a given date
inline std::string buildListUrl(const std::string& station,
                                 int year, int month, int day) {
    char buf[256];
    snprintf(buf, sizeof(buf),
             "/%04d/%02d/%02d/%s/",
             year, month, day, station.c_str());
    return std::string(buf);
}

// Build the download URL for a specific key
inline std::string buildDownloadUrl(const std::string& key) {
    return "/" + key;
}

inline std::string buildRadarListRequest(const StationInfo& station,
                                         int year, int month, int day,
                                         const std::string& currentKey = {}) {
    switch (station.feed) {
        case RadarFeedKind::IemLevel2RawDirList:
            return "/level2/raw/" + std::string(stationFeedCode(station)) + "/dir.list";
        case RadarFeedKind::AwsS3DatePartitioned:
        default: {
            std::string listPath = buildListUrl(stationFeedCode(station), year, month, day);
            std::string query = "/?list-type=2&prefix=" + std::string(listPath.data() + 1);
            if (!currentKey.empty())
                query += "&start-after=" + currentKey;
            else
                query += "&max-keys=1000";
            return query;
        }
    }
}

inline std::string buildRadarDownloadRequest(const StationInfo& station,
                                             const std::string& key) {
    switch (station.feed) {
        case RadarFeedKind::IemLevel2RawDirList:
            return "/level2/raw/" + std::string(stationFeedCode(station)) + "/" + key;
        case RadarFeedKind::AwsS3DatePartitioned:
        default:
            return "/" + key;
    }
}

// Get current UTC date
inline void getUtcDate(int& year, int& month, int& day) {
    time_t t = time(nullptr);
    struct tm utc;
#ifdef _WIN32
    gmtime_s(&utc, &t);
#else
    gmtime_r(&t, &utc);
#endif
    year = utc.tm_year + 1900;
    month = utc.tm_mon + 1;
    day = utc.tm_mday;
}

inline bool isLeapYear(int year) {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

inline int daysInMonth(int year, int month) {
    static const int kDaysPerMonth[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (month == 2) return isLeapYear(year) ? 29 : 28;
    return kDaysPerMonth[month - 1];
}

inline void shiftDate(int& year, int& month, int& day, int deltaDays) {
    while (deltaDays < 0) {
        day--;
        if (day < 1) {
            month--;
            if (month < 1) {
                month = 12;
                year--;
            }
            day = daysInMonth(year, month);
        }
        deltaDays++;
    }

    while (deltaDays > 0) {
        day++;
        if (day > daysInMonth(year, month)) {
            day = 1;
            month++;
            if (month > 12) {
                month = 1;
                year++;
            }
        }
        deltaDays--;
    }
}

// Parse S3 XML list response to extract file keys
// Simple tag extraction - no XML library needed
inline std::vector<NexradFile> parseS3ListResponse(const std::string& xml) {
    std::vector<NexradFile> files;

    size_t pos = 0;
    while (true) {
        size_t keyStart = xml.find("<Key>", pos);
        if (keyStart == std::string::npos) break;
        keyStart += 5;

        size_t keyEnd = xml.find("</Key>", keyStart);
        if (keyEnd == std::string::npos) break;

        std::string key = xml.substr(keyStart, keyEnd - keyStart);
        pos = keyEnd + 6;

        // Skip MDM (metadata) files
        if (key.find("_MDM") != std::string::npos) continue;

        // Parse size if available
        size_t sizeVal = 0;
        size_t sizeStart = xml.find("<Size>", pos);
        if (sizeStart != std::string::npos && sizeStart < xml.find("<Key>", pos)) {
            sizeStart += 6;
            size_t sizeEnd = xml.find("</Size>", sizeStart);
            if (sizeEnd != std::string::npos) {
                sizeVal = std::stoull(xml.substr(sizeStart, sizeEnd - sizeStart));
            }
        }

        NexradFile f;
        f.key = key;
        f.url = "/" + key;
        f.size = sizeVal;
        files.push_back(std::move(f));
    }

    // Sort by key (which includes timestamp) so latest is last
    std::sort(files.begin(), files.end(),
              [](const NexradFile& a, const NexradFile& b) {
                  return a.key < b.key;
              });

    return files;
}

inline std::vector<NexradFile> parseIemDirListResponse(const StationInfo& station,
                                                       const std::string& text) {
    std::vector<NexradFile> files;
    std::istringstream stream(text);
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty())
            continue;

        std::istringstream lineStream(line);
        size_t sizeVal = 0;
        std::string filename;
        if (!(lineStream >> sizeVal >> filename))
            continue;
        if (filename.empty() || filename.find("_MDM") != std::string::npos)
            continue;

        NexradFile file;
        file.key = filename;
        file.url = buildRadarDownloadRequest(station, filename);
        file.size = sizeVal;
        files.push_back(std::move(file));
    }

    std::sort(files.begin(), files.end(),
              [](const NexradFile& a, const NexradFile& b) {
                  return a.key < b.key;
              });
    return files;
}

inline std::vector<NexradFile> parseRadarListResponse(const StationInfo& station,
                                                      const std::vector<uint8_t>& payload) {
    const std::string text(payload.begin(), payload.end());
    switch (station.feed) {
        case RadarFeedKind::IemLevel2RawDirList:
            return parseIemDirListResponse(station, text);
        case RadarFeedKind::AwsS3DatePartitioned:
        default:
            return parseS3ListResponse(text);
    }
}

inline bool isDigitSpan(const std::string& text, size_t pos, size_t count) {
    if (pos + count > text.size()) return false;
    for (size_t i = 0; i < count; i++) {
        if (!std::isdigit((unsigned char)text[pos + i]))
            return false;
    }
    return true;
}

inline std::string radarFilenameFromKey(const std::string& key) {
    const size_t slash = key.rfind('/');
    return (slash != std::string::npos) ? key.substr(slash + 1) : key;
}

inline bool extractRadarFileDateTime(const std::string& keyOrFilename,
                                     int& year, int& month, int& day,
                                     int& hh, int& mm, int& ss) {
    const std::string filename = radarFilenameFromKey(keyOrFilename);
    for (size_t i = 0; i < filename.size(); i++) {
        if (!isDigitSpan(filename, i, 8))
            continue;

        size_t timePos = std::string::npos;
        size_t timeDigits = 0;
        if (i + 15 <= filename.size() && filename[i + 8] == '_' && isDigitSpan(filename, i + 9, 6)) {
            timePos = i + 9;
            timeDigits = 6;
        } else if (i + 13 <= filename.size() && filename[i + 8] == '_' && isDigitSpan(filename, i + 9, 4)) {
            timePos = i + 9;
            timeDigits = 4;
        } else if (i + 14 <= filename.size() && isDigitSpan(filename, i + 8, 6)) {
            timePos = i + 8;
            timeDigits = 6;
        }

        if (timePos == std::string::npos)
            continue;

        year = std::stoi(filename.substr(i, 4));
        month = std::stoi(filename.substr(i + 4, 2));
        day = std::stoi(filename.substr(i + 6, 2));
        hh = std::stoi(filename.substr(timePos, 2));
        mm = std::stoi(filename.substr(timePos + 2, 2));
        ss = (timeDigits == 6) ? std::stoi(filename.substr(timePos + 4, 2)) : 0;
        return true;
    }

    return false;
}
