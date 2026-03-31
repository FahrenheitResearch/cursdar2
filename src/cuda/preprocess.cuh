#pragma once
#include "../nexrad/sweep_data.h"
#include <vector>

namespace gpu_preprocess {

bool dealiasVelocity(PrecomputedSweep::ProductData& velPd, int numRadials);
bool suppressReflectivityRings(std::vector<PrecomputedSweep>& sweeps);

} // namespace gpu_preprocess
