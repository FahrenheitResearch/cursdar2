# cursdar2

`cursdar2` is a CUDA-native NEXRAD workstation focused on fast live Level 2 ingest, responsive 2D/3D radar rendering, archive review, and warning-overlay analysis.

It is being built as a standalone successor to `cursdar`: same GPU-first mentality, much cleaner operator workflow, and a repo that can evolve independently.

## Highlights

- Live Level 2 ingest from AWS NEXRAD plus selected IEM-hosted experimental/testbed feeds
- Fast single-site 2D rendering with working tilt stepping
- National mosaic rendering across the radar network
- Real-time 3D volume mode
- Draggable vertical cross-sections
- Historic case playback with frame scrubbing
- Archive snapshot loading, including the March 30, 2025 multi-site case
- Live and historic warning polygons, watches, and related alert overlays
- Custom basemap system with relief, ops-dark, satellite, and hybrid imagery modes
- Support for Norman ROC/KCRI experimental feeds including `KCRI`, `DAN1`, `DOP1`, `FOP1`, `NOP3`, `NOP4`, `ROP3`, and `ROP4`
- Highly configurable warning styling:
  - per-category toggles
  - outline/fill controls
  - opacity and line scaling
  - custom colors
- Experimental storm interrogation overlays:
  - TDS markers
  - hail markers
  - mesocyclone / TVS markers
- Storm-relative velocity mode
- GR / RadarScope-style color table import
- Early GR-style polling link intake
- Hardware-aware performance profiles with live VRAM / RAM telemetry
- Docked workstation UI with station browser, inspector, warnings panel, and historic timeline

## Live Data Model

`cursdar2` now uses tiered live polling instead of a blunt full-network refresh loop.

- Active station: fast polling for newest available scans
- In-view stations: medium cadence polling
- Background stations: slower maintenance polling
- Warning overlays: separate live polling loop

Once a station has already loaded a scan, the app uses incremental S3 listing against the last known volume key instead of re-listing the entire day each time. That keeps the hot path much lighter while still picking up newly published volumes quickly.

Practical note: display latency is still bounded by upstream publication. The app can only render a scan once NOAA/AWS has published a complete object.

## Current Feature Surface

Implemented now:

- Standalone `cursdar2` source tree and build target
- Live single-site view
- Live national mosaic
- Experimental/testbed radar site support alongside the core WSR-88D network
- 3D volume rendering
- Cross-sections
- Tilt browsing in single-radar mode
- Archive playback
- Archive snapshot loading
- Live warning overlays
- Historic warning overlays matched to archive timestamps
- Warning customization controls
- Custom basemap renderer with vector overlays and cached raster imagery
- Color table import and per-product reset
- Polling-link fetch and inspection
- CUDA-backed rendering pipeline
- Performance profiles with lighter 3D quality and reduced archive caching

Not finished yet:

- Full GR2 feature parity
- Full placefile rendering
- Measurement / interrogation tools
- Broader polling-link product support
- Long-duration operational validation across many live weather events

## Build Requirements

### Windows

- NVIDIA GPU with CUDA support
- CUDA Toolkit
- Visual Studio 2022 Build Tools
- Ninja

### Linux

- NVIDIA GPU with CUDA support
- CUDA Toolkit
- CMake
- A recent C++17 compiler

## Build

### Windows

```bat
build.bat
```

Binary output:

```text
build/cursdar2.exe
```

### Linux

```bash
chmod +x build.sh
./build.sh
```

## Controls

- `1-7`: select radar product
- `Left` / `Right`: cycle products
- `Up` / `Down`: cycle tilts
- `A`: toggle national mosaic
- `V`: toggle 3D volume
- `X`: toggle cross-section
- `S`: toggle storm-relative velocity
- `R`: refresh live data
- `Home`: reset to CONUS
- `Escape`: return to auto-track
- `Space`: play / pause historic playback

## UI Notes

- The inspector shows the latest scan time for the active site.
- The operator console exposes `Auto`, `Quality`, `Balanced`, and `Performance` profiles, plus current/peak VRAM and process RAM.
- The warnings panel can show live or historic polygons depending on mode.
- The basemap panel exposes relief, ops-dark, satellite, and hybrid map modes plus overlay controls.
- `Ops Dark` is the default basemap for instant startup; imagery styles stream in on demand and fall back to cached parent tiles while sharper zoom levels load.
- Color tables can be loaded from the operator console with a file browser.
- Polling links are currently ingested and inspected, but not yet fully rendered as full GR-style placefile content.

## Status

This is already a real GPU radar workstation, but it is still an active build-out rather than a finished operational replacement for mature commercial software. The fast path is there now; the remaining work is on feature depth, workflow polish, and repeated validation with real events.
