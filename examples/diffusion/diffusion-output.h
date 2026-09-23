#pragma once

#include <cstddef>

// Repeated tokens can be meaningful text, including JSON numbers. Only an
// end-of-generation token may shorten a decoded canvas here.
template <typename Token, typename IsEog>
std::size_t diffusion_output_length(const Token * canvas, std::size_t n, IsEog is_eog) {
    for (std::size_t i = 0; i < n; ++i) {
        if (is_eog(canvas[i])) {
            return i;
        }
    }
    return n;
}
