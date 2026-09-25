#include "../examples/diffusion/diffusion-output.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(const char * name, const std::vector<int32_t> & canvas, std::size_t n, std::size_t expected) {
    std::size_t inspected = 0;
    const auto length = diffusion_output_length(canvas.data(), n, [&](int32_t token) {
        ++inspected;
        return token == 106 || token == 1;
    });
    const auto expected_inspected = expected < n ? expected + 1 : n;
    if (length != expected || inspected != expected_inspected) {
        std::fprintf(stderr, "%s: length %zu (expected %zu), inspected %zu (expected %zu)\n",
                     name, length, expected, inspected, expected_inspected);
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    // Native DiffusionGemma tokenization of:
    // <|channel>thought\n<channel|>{"answers":{"q1":[0.0,0.0,0.0,0.95,0.05]}}
    // Zeros repeat at every other position from index 10 through 22, while
    // intervening dots and commas differ. The old repetition heuristic cut at 10.
    const std::vector<int32_t> json = {
        100, 45518, 107, 101, 14937, 53821, 68693, 236809, 236770, 89045,
        236771, 236761, 236771, 236764, 236771, 236761, 236771, 236764,
        236771, 236761, 236771, 236764, 236771, 236761, 236819, 236810,
        236764, 236771, 236761, 236771, 236810, 236842, 1807,
    };
    check("compact JSON without EOG", json, json.size(), 33);
    auto terminated_json = json;
    terminated_json.insert(terminated_json.end(), {106, 50, 50, 50, 50, 50, 50, 50});
    check("compact JSON with EOG and trailing garbage", terminated_json, terminated_json.size(), 33);

    check("repeated decimal digits", {236771, 236761, 236771, 236771, 236771, 236771,
                                     236771, 236771, 236771, 236771}, 10, 10);
    check("repeated two-token sequence", {50, 51, 50, 51, 50, 51, 50, 51,
                                          50, 51, 50, 51, 50, 51}, 14, 14);
    check("EOG first", {106, 50, 51}, 3, 0);
    check("alternative EOG first", {1, 106, 50}, 3, 0);
    check("EOG last", {50, 51, 106}, 3, 2);
    check("first of multiple EOG tokens", {50, 106, 51, 1}, 4, 1);
    check("repetitions extending beyond EOG", {50, 106, 50, 106, 50, 106, 50, 106,
                                             50, 106, 50, 106, 50, 106}, 14, 1);
    check("empty canvas", {}, 0, 0);

    // The caller owns output limits. This helper reads only its supplied canvas
    // span and does not infer a stopping point from repeated content.
    check("bounded span excludes later EOG", terminated_json, 10, 10);
    check("EOG immediately outside span", terminated_json, 33, 33);
    check("EOG at span boundary", terminated_json, 34, 33);
    check("zero span with backing tokens", terminated_json, 0, 0);
    std::puts("diffusion output tests passed (14 cases)");
}
