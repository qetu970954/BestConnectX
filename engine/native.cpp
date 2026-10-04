// Original, optional CPU kernels. No Python, PyTorch or third-party engine code.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

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

// A forest owns search statistics, not game rules or neural evaluation. One call
// walks every root; Python only creates a new leaf and supplies its checked state.
struct SearchNode {
    int player = 0, winner = 0, parent = -1, parent_edge = -1;
    bool done = false;
    std::vector<std::int32_t> actions, visits, children;
    std::vector<double> prior, total;
};

struct SearchTree {
    int roots;
    std::vector<SearchNode> nodes;

    SearchTree(const std::int32_t* players, int count) : roots(count), nodes(count) {
        for (int i = 0; i < count; ++i) nodes[i].player = players[i];
    }

    void backup(int id, double value) {
        const int player = nodes[id].player;
        while (nodes[id].parent >= 0) {
            const int edge = nodes[id].parent_edge;
            id = nodes[id].parent;
            ++nodes[id].visits[edge];
            nodes[id].total[edge] += nodes[id].player == player ? value : -value;
        }
    }
};

API void* tree_create(const std::int32_t* players, int count) {
    if (count < 1) return nullptr;
    for (int i = 0; i < count; ++i)
        if (players[i] != -1 && players[i] != 1) return nullptr;
    try { return new SearchTree(players, count); }
    catch (...) { return nullptr; }  // Exceptions must never cross the C ABI.
}

API void tree_destroy(void* handle) {
    delete static_cast<SearchTree*>(handle);
}

API int tree_advance(void* handle, std::int32_t* leaves,
                     std::int32_t* parents, std::int32_t* actions) {
    if (!handle) return -1;
    auto& tree = *static_cast<SearchTree*>(handle);
    try {
        int pending = 0;
        for (int root = 0; root < tree.roots; ++root) {
            int id = root;
            parents[root] = actions[root] = -1;
            while (!tree.nodes[id].done && !tree.nodes[id].actions.empty()) {
                const auto& node = tree.nodes[id];
                const int edge = puct_select(node.prior.data(), node.visits.data(),
                                             node.total.data(), static_cast<int>(node.actions.size()));
                if (node.children[edge] < 0) {
                    SearchNode child;
                    child.parent = id;
                    child.parent_edge = edge;
                    parents[root] = id;
                    actions[root] = node.actions[edge];
                    const int child_id = static_cast<int>(tree.nodes.size());
                    tree.nodes.push_back(std::move(child));
                    // push_back can relocate nodes: use IDs, never a stale reference.
                    tree.nodes[id].children[edge] = child_id;
                    id = child_id;
                    break;
                }
                id = node.children[edge];
            }
            if (tree.nodes[id].done) {
                tree.backup(id, static_cast<double>(tree.nodes[id].winner * tree.nodes[id].player));
                leaves[root] = -1;
            } else {
                leaves[root] = id;
                ++pending;
            }
        }
        return pending;
    } catch (...) { return -2; }
}

API int tree_commit(void* handle, const std::int32_t* ids, const std::int32_t* players,
                    const std::int32_t* winners, const std::uint8_t* done,
                    const std::int32_t* offsets, const std::int32_t* actions,
                    const double* prior, const double* values, int count, int size) {
    if (!handle || count < 1 || size < 0 || offsets[0] != 0 || offsets[count] != size) return -1;
    auto& tree = *static_cast<SearchTree*>(handle);
    for (int i = 0; i < count; ++i) {
        if (ids[i] < 0 || ids[i] >= static_cast<int>(tree.nodes.size())
            || (players[i] != -1 && players[i] != 1) || winners[i] < -1 || winners[i] > 1
            || done[i] > 1 || offsets[i] < 0 || offsets[i + 1] < offsets[i] || offsets[i + 1] > size
            || (done[i] ? offsets[i + 1] != offsets[i] : offsets[i + 1] == offsets[i])
            || !tree.nodes[ids[i]].actions.empty() || tree.nodes[ids[i]].done) return -1;
    }
    for (int i = 0; i < size; ++i) if (actions[i] < 0 || actions[i] >= 625) return -1;
    try {
        for (int i = 0; i < count; ++i) {
            auto& node = tree.nodes[ids[i]];
            node.player = players[i];
            node.winner = winners[i];
            node.done = done[i] != 0;
            const int begin = offsets[i], end = offsets[i + 1], length = end - begin;
            node.actions.assign(actions + begin, actions + end);
            node.prior.assign(prior + begin, prior + end);
            node.visits.assign(length, 0);
            node.total.assign(length, 0.0);
            node.children.assign(length, -1);
            tree.backup(ids[i], node.done ? static_cast<double>(node.winner * node.player) : values[i]);
        }
        return 0;
    } catch (...) { return -2; }
}

API int tree_policy(void* handle, const std::int32_t* offsets, float* output) {
    if (!handle || offsets[0] != 0) return -1;
    const auto& tree = *static_cast<SearchTree*>(handle);
    for (int i = 0; i < tree.roots; ++i) {
        const int area = offsets[i + 1] - offsets[i];
        if (offsets[i] < 0 || area < 4 || area > 625) return -1;
        const auto& node = tree.nodes[i];
        for (int action : node.actions) if (action >= area) return -1;
    }
    std::fill_n(output, offsets[tree.roots], 0.f);
    for (int i = 0; i < tree.roots; ++i) {
        const auto& node = tree.nodes[i];
        std::int64_t visits = 0;
        for (int count : node.visits) visits += count;
        const float sum = static_cast<float>(visits);
        for (std::size_t j = 0; j < node.actions.size(); ++j)
            output[offsets[i] + node.actions[j]] = visits ? static_cast<float>(node.visits[j]) / sum
                                                        : static_cast<float>(node.prior[j]);
    }
    return 0;
}
