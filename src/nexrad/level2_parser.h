#pragma once
#include "level2.h"
#include <vector>
#include <string>
#include <functional>

// Parse a Level 2 Archive II file from raw bytes.
// Handles BZ2 decompression and message parsing.
class Level2Parser {
public:
    // Decode an Archive II payload to the combined decompressed message stream.
    // Returns empty on failure.
    static std::vector<uint8_t> decodeArchiveBytes(const std::vector<uint8_t>& fileData);

    // Parse an already-decoded Archive II message stream.
    // Returns empty data on failure.
    static ParsedRadarData parseDecodedMessages(const std::vector<uint8_t>& decodedBytes,
                                                const std::string& stationId = {});

    // Parse raw file bytes into structured radar data.
    // Returns empty data on failure.
    static ParsedRadarData parse(const std::vector<uint8_t>& fileData);

    // Progress callback: (decompressed_blocks, total_blocks)
    using ProgressCallback = std::function<void(int, int)>;
    static ParsedRadarData parse(const std::vector<uint8_t>& fileData, ProgressCallback cb);

private:
    // Find BZ2 block boundaries in the file
    static std::vector<size_t> findBZ2Blocks(const uint8_t* data, size_t size);

    // Decompress a single BZ2 block
    static std::vector<uint8_t> decompressBZ2Block(const uint8_t* data, size_t maxSize);

    // Parse decompressed messages into radials
    static void parseMessages(const uint8_t* data, size_t size,
                              ParsedRadarData& out);

    // Parse a single Message Type 31
    static void parseMsg31(const uint8_t* data, size_t size,
                           ParsedRadarData& out);

    // Organize parsed radials into sweeps
    static void organizeSweeps(ParsedRadarData& out);
};
