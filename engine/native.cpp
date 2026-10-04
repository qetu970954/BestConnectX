// Original, optional CPU kernels. No Python, PyTorch or third-party engine code.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

API int connection_features(const std::int8_t* boards, const std::int32_t* players,
                            const std::int32_t* left, const std::int32_t* lines,
                            int batch, int area, int windows, int length, int stones,
                            float* planes, std::uint8_t* counts, std::uint8_t* threats) {
    if (batch < 1 || area < 4 || area > 625 || windows < 1 || length < 2
        || length > 25 || stones < 1 || stones > 2) return -1;
    for (int i = 0; i < windows * length; ++i)
        if (lines[i] < 0 || lines[i] >= area) return -1;
    std::fill_n(planes, static_cast<std::size_t>(batch) * 8 * area, 0.f);
    std::fill_n(threats, batch, std::uint8_t{0});
    for (int row = 0; row < batch; ++row) {
        if (players[row] != -1 && players[row] != 1) return -1;
        const auto* board = boards + static_cast<std::size_t>(row) * area;
        auto* output = planes + static_cast<std::size_t>(row) * 8 * area;
        auto* line_counts = counts + static_cast<std::size_t>(row) * 3 * windows;
        for (int cell = 0; cell < area; ++cell) {
            output[cell] = board[cell] == players[row];
            output[area + cell] = board[cell] == -players[row];
            output[2 * area + cell] = left[row] == 2;
            output[3 * area + cell] = players[row] == 1;
        }
        for (int window = 0; window < windows; ++window) {
            int white = 0, empty = 0, black = 0;
            int holes[2] = {0, 0};
            const auto* cells = lines + window * length;
            for (int i = 0; i < length; ++i) {
                const int cell = cells[i];
                const int value = board[cell];
                white += value == -1;
                black += value == 1;
                if (value == 0) {
                    if (empty < 2) holes[empty] = cell;
                    ++empty;
                }
            }
            line_counts[window] = static_cast<std::uint8_t>(white);
            line_counts[windows + window] = static_cast<std::uint8_t>(empty);
            line_counts[2 * windows + window] = static_cast<std::uint8_t>(black);
            if (empty < 1 || empty > stones) continue;
            for (int color : {-1, 1}) {
                const int own = color == 1 ? black : white;
                const int enemy = color == 1 ? white : black;
                if (enemy != 0 || own < length - stones) continue;
                const int offset = color == players[row] ? 4 : 6;
                for (int i = 0; i < empty; ++i)
                    output[(offset + empty - 1) * area + holes[i]] = 1.f;
                const int slot = color == 1 ? 1 : 0;
                threats[row] |= static_cast<std::uint8_t>(1u << (2 * slot + empty - 1));
            }
        }
    }
    return 0;
}

API int puct_select(const double* prior, const std::int32_t* visits,
                    const double* total, int size) {
    if (size < 1) return -1;
    std::int64_t sum = 0;
    for (int i = 0; i < size; ++i) sum += visits[i];
    const double scale = std::sqrt(1.0 + static_cast<double>(sum));
    double best = -std::numeric_limits<double>::infinity();
    int chosen = 0;
    for (int i = 0; i < size; ++i) {
        const double q = visits[i] > 0 ? total[i] / visits[i] : 0.0;
        const double u = 1.5 * prior[i] * scale / (1.0 + visits[i]);
        const double score = q + u;
        // NumPy argmax selects the first tie and the first NaN.
        if (std::isnan(score)) return i;
        if (score > best) { best = score; chosen = i; }
    }
    return chosen;
}
