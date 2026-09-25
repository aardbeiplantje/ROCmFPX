#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

struct diffusion_eb_row_result {
    int32_t argmax;
    int32_t sampled;
    double entropy;
};

// Keep the model's FP32 logit scaling and exponential, but accumulate across the
// vocabulary in double precision. Small probability tails must contribute even
// when a high-probability token occurs early in vocabulary order.
inline diffusion_eb_row_result diffusion_eb_sample_row(
        const float * logits, int32_t n_vocab, float temp_inv, float draw) {
    float maximum = -std::numeric_limits<float>::infinity();
    int32_t argmax = 0;
    for (int32_t v = 0; v < n_vocab; ++v) {
        const float scaled = logits[v] * temp_inv;
        if (scaled > maximum) {
            maximum = scaled;
            argmax = v;
        }
    }

    double normalizer = 0.0;
    for (int32_t v = 0; v < n_vocab; ++v) {
        normalizer += expf(logits[v] * temp_inv - maximum);
    }

    const double target = static_cast<double>(draw) * normalizer;
    double cumulative = 0.0;
    double entropy = 0.0;
    int32_t sampled = n_vocab - 1;
    bool picked = false;
    for (int32_t v = 0; v < n_vocab; ++v) {
        const float weight = expf(logits[v] * temp_inv - maximum);
        const double probability = static_cast<double>(weight) / normalizer;
        if (probability > 0.0) {
            entropy -= probability * std::log(probability);
        }
        cumulative += weight;
        if (!picked && weight > 0.0f && cumulative >= target) {
            sampled = v;
            picked = true;
        }
    }
    return { argmax, sampled, entropy };
}
