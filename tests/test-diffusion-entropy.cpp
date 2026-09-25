#include "../examples/diffusion/diffusion-entropy.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(EXIT_FAILURE);
    }
}

static void near(double actual, double expected, double tolerance, const char * message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::fprintf(stderr, "%s: %.12g != %.12g (tolerance %.3g)\n", message, actual, expected, tolerance);
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    // A normalized distribution with a large vocabulary and small tails. Its
    // entropy is just above the default stopping threshold. Sequential FP32
    // accumulation incorrectly puts it below that threshold for early peaks.
    constexpr int32_t n_vocab = 262144;
    const float tail_logit = -20.82862663269043f; // float32 log(float32(9e-10))
    const double tail_weight = std::exp(static_cast<double>(tail_logit));
    const double normalizer = 1.0 + (n_vocab - 1) * tail_weight;
    const double expected_entropy = std::log(normalizer) -
            (n_vocab - 1) * tail_weight / normalizer * tail_logit;
    std::vector<double> entropies;
    for (const int32_t peak : { 0, 14937, n_vocab - 1 }) {
        std::vector<float> logits(n_vocab, tail_logit);
        logits[peak] = 0.0f;
        const auto result = diffusion_eb_sample_row(logits.data(), n_vocab, 1.0f, 0.5f);
        check(result.argmax == peak && result.sampled == peak, "peaked distribution selected the wrong token");
        near(result.entropy, expected_entropy, 1e-7, "large-vocabulary analytic entropy");
        check(result.entropy > 0.005f, "small tails must prevent premature confidence stopping");
        entropies.push_back(result.entropy);

        // The MI bound accepts the first token plus enough subsequent tokens
        // that their strictly-earlier entropy sum stays within the bound.
        double cumulative = 0.0;
        int accepted = 0;
        for (int i = 0; i < 22; ++i) {
            cumulative += result.entropy;
            accepted += cumulative - result.entropy <= 0.1f;
        }
        check(accepted == 20, "small tails must affect entropy-bound acceptance");
    }
    near(*std::max_element(entropies.begin(), entropies.end()),
         *std::min_element(entropies.begin(), entropies.end()), 1e-9,
         "vocabulary permutation must not change entropy");

    // A draw in the small tail must not collapse onto the early peak. Compare
    // against a closed-form inverse CDF, using the same FP32 exponential weight.
    std::vector<float> tail_logits(n_vocab, tail_logit);
    tail_logits[0] = 0.0f;
    const float draw = 0.9999f;
    const double weight = expf(tail_logit);
    const double target = static_cast<double>(draw) * (1.0 + (n_vocab - 1) * weight);
    const int32_t expected_sample = static_cast<int32_t>(std::ceil((target - 1.0) / weight));
    check(expected_sample > 0 && expected_sample < n_vocab, "tail quantile fixture must select a tail token");
    check(diffusion_eb_sample_row(tail_logits.data(), n_vocab, 1.0f, draw).sampled == expected_sample,
          "large-vocabulary CDF must preserve small tail probabilities");

    const float uniform[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (int32_t i = 0; i < 4; ++i) {
        const auto result = diffusion_eb_sample_row(uniform, 4, 1.0f, (i + 0.5f) / 4.0f);
        check(result.argmax == 0 && result.sampled == i, "uniform quantile or argmax tie changed");
        near(result.entropy, std::log(4.0), 1e-12, "uniform entropy");
    }
    check(diffusion_eb_sample_row(uniform, 4, 1.0f, 0.0f).sampled == 0, "zero draw must select first positive bin");
    check(diffusion_eb_sample_row(uniform, 4, 1.0f, std::nextafter(1.0f, 0.0f)).sampled == 3,
          "last representable draw must select last uniform bin");

    // FP32 logit scaling is intentional; scaling the input explicitly must give
    // the same distribution as applying the inverse temperature in the helper.
    const float logits[] = { -2.0f, -1.0f, 0.0f };
    const float scaled[] = { -4.0f, -2.0f, 0.0f };
    const auto temperature_result = diffusion_eb_sample_row(logits, 3, 2.0f, 0.9f);
    const auto scaled_result = diffusion_eb_sample_row(scaled, 3, 1.0f, 0.9f);
    check(temperature_result.argmax == scaled_result.argmax && temperature_result.sampled == scaled_result.sampled,
          "temperature scaling changed token selection");
    near(temperature_result.entropy, scaled_result.entropy, 1e-12, "temperature scaling entropy");

    // Underflowed zero-probability bins must contribute zero entropy, not NaN.
    const float sharp[] = { 0.0f, -1000.0f, -1000.0f };
    const auto sharp_result = diffusion_eb_sample_row(sharp, 3, 1.0f, 0.5f);
    check(sharp_result.argmax == 0 && sharp_result.sampled == 0, "sharp distribution selected the wrong token");
    near(sharp_result.entropy, 0.0, 0.0, "sharp entropy");
    check(sharp_result.entropy < 0.005f, "certain distribution must remain confident");

    const float leading_zero[] = { -1000.0f, 0.0f, -1000.0f };
    check(diffusion_eb_sample_row(leading_zero, 3, 1.0f, 0.0f).sampled == 1,
          "zero draw must skip leading zero-probability bins");

    const float singleton[] = { 42.0f };
    const auto singleton_result = diffusion_eb_sample_row(singleton, 1, 2.0f, 0.0f);
    check(singleton_result.argmax == 0 && singleton_result.sampled == 0, "singleton must select its only token");
    near(singleton_result.entropy, 0.0, 0.0, "singleton entropy");

    std::puts("diffusion entropy tests passed");
}
