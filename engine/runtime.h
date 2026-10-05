#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif
using Clock = std::chrono::steady_clock;
using Cells = std::vector<int>;
using Edges = std::set<Cells>;
using Stop = int (*)();
inline double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
struct Limit {};
struct Budget {
    Clock::time_point end;
    int remaining;
    Stop stopped;
    std::mt19937_64* rng=nullptr;
    void tick() {
        if (remaining-- <= 0 || Clock::now() >= end || (stopped && stopped())) throw Limit{};
    }
};
struct Position {
    int n, k, stones, starter, player, left, winner, done;
    std::vector<std::int8_t> board;
    Cells moves;
    std::shared_ptr<const std::vector<Cells>> windows;
    Position(const std::int8_t* data, const int* meta, const int* history, int count)
        : n(meta[0]), k(meta[1]), stones(meta[2]), starter(meta[3]), player(meta[4]),
          left(meta[5]), winner(meta[6]), done(meta[7]) {
        if (n < 2 || n > 25 || k < 2 || k > n || stones < 1 || stones > 2
            || starter < 1 || starter > 2 || (player != 1 && player != -1)
            || left < 1 || left > 2 || winner < -1 || winner > 1 || done < 0 || done > 1
            || (winner && !done) || count < 0 || count > n*n) throw std::invalid_argument("Invalid game state.");
        board.assign(data, data + n*n);
        for (int cell : board) if (cell < -1 || cell > 1) throw std::invalid_argument("Invalid board cell.");
        if (count) moves.assign(history, history + count);
        std::set<int> seen;
        for (int cell : moves) if (cell < 0 || cell >= n*n || !seen.insert(cell).second || !board[cell])
            throw std::invalid_argument("Invalid move history.");
        static thread_local std::map<int,std::shared_ptr<const std::vector<Cells>>> cache;
        auto& cached=cache[n*32+k];
        if (!cached) cached=std::make_shared<const std::vector<Cells>>(build_lines());
        windows=cached;
    }
    const std::vector<Cells>& lines() const { return *windows; }
    std::vector<Cells> build_lines() const {
        std::vector<Cells> result;
        for (int r=0; r<n; ++r) for (int c=0; c<n; ++c)
            for (auto [dr, dc] : std::array<std::pair<int,int>,4>{{{0,1},{1,0},{1,1},{1,-1}}}) {
                int y=r+(k-1)*dr, x=c+(k-1)*dc;
                if (y<0 || y>=n || x<0 || x>=n) continue;
                Cells line;
                for (int i=0; i<k; ++i) line.push_back((r+i*dr)*n+c+i*dc);
                result.push_back(std::move(line));
            }
        return result;
    }
    Cells empty() const {
        Cells cells;
        for (int i=0; i<n*n; ++i) if (!board[i]) cells.push_back(i);
        return cells;
    }
    void play(int action) {
        if (done || action<0 || action>=n*n || board[action]) throw std::invalid_argument("Illegal placement.");
        board[action]=static_cast<std::int8_t>(player); moves.push_back(action);
        int r=action/n, c=action%n;
        for (auto [dr,dc] : std::array<std::pair<int,int>,4>{{{0,1},{1,0},{1,1},{1,-1}}}) {
            int length=1;
            for (int sign : {-1,1}) {
                int y=r+sign*dr, x=c+sign*dc;
                while (y>=0 && y<n && x>=0 && x<n && board[y*n+x]==player) {
                    ++length; y+=sign*dr; x+=sign*dc;
                }
            }
            if (length>=k) { winner=player; done=1; return; }
        }
        if (empty().empty()) { done=1; return; }
        if (--left==0) { player=-player; left=stones; }
    }
    Edges threats(int color, int budget) const {
        Edges result;
        for (const auto& line : lines()) {
            std::array<int,25> holes; int count=0; bool blocked=false;
            for (int cell : line) {
                if (board[cell]==-color) { blocked=true; break; }
                if (!board[cell]) { holes[count++]=cell; if (count>budget) break; }
            }
            if (!blocked && count>0 && count<=budget) {
                Cells edge(holes.begin(),holes.begin()+count);
                std::sort(edge.begin(),edge.end()); result.insert(std::move(edge));
            }
        }
        return result;
    }
    std::vector<float> features() const {
        std::vector<float> out(8*n*n, 0);
        for (int i=0; i<n*n; ++i) {
            out[i]=board[i]==player; out[n*n+i]=board[i]==-player;
            out[2*n*n+i]=left==2; out[3*n*n+i]=player==1;
        }
        for (int color : {player,-player}) for (const auto& edge : threats(color,stones))
            for (int cell : edge) out[((color==player ? 4 : 6)+edge.size()-1)*n*n+cell]=1;
        return out;
    }
    Cells actions() const;
    std::vector<double> scores() const {
        std::vector<double> out(n*n,0);
        for (const auto& line : lines()) {
            int own=0, enemy=0;
            for (int cell : line) { own+=board[cell]==player; enemy+=board[cell]==-player; }
            double w=(enemy==0 ? std::pow(5.,own) : 0)+(own==0 ? std::pow(4.,enemy) : 0);
            for (int cell : line) out[cell]+=w;
        }
        for (int i=0; i<n*n; ++i) out[i]-=.001*((i/n-n/2)*(i/n-n/2)+(i%n-n/2)*(i%n-n/2));
        return out;
    }
    int heuristic() const {
        auto legal=actions(); if (legal.empty()) throw std::invalid_argument("Game has ended.");
        auto weights=scores(); return *std::max_element(legal.begin(),legal.end(),[&](int a,int b){return weights[a]<weights[b];});
    }
};
inline Edges covers(const Edges& edges, int budget) {
    if (edges.empty()) return Edges{Cells{}};
    if (!budget) return {};
    auto first=std::min_element(edges.begin(),edges.end(),[](const Cells& a,const Cells& b){
        return a.size()!=b.size() ? a.size()<b.size() : a<b;
    });
    Edges found;
    for (int cell : *first) {
        Edges rest;
        for (const auto& edge : edges) if (!std::binary_search(edge.begin(),edge.end(),cell)) rest.insert(edge);
        for (auto cover : covers(rest,budget-1)) {
            cover.push_back(cell); std::sort(cover.begin(),cover.end()); found.insert(cover);
        }
    }
    Edges minimal;
    for (const auto& s : found) {
        bool redundant=false;
        for (const auto& t : found) if (t.size()<s.size() && std::includes(s.begin(),s.end(),t.begin(),t.end())) { redundant=true; break; }
        if (!redundant) minimal.insert(s);
    }
    return minimal;
}
inline Cells Position::actions() const {
    if (done) return {};
    auto own=threats(player,left);
    std::set<int> result;
    if (!own.empty()) {
        auto shortest=std::min_element(own.begin(),own.end(),[](const Cells& a,const Cells& b){return a.size()<b.size();})->size();
        for (const auto& edge : own) if (edge.size()==shortest) result.insert(edge.begin(),edge.end());
    } else {
        auto enemy=threats(-player,stones);
        if (!enemy.empty()) for (const auto& cover : covers(enemy,left)) result.insert(cover.begin(),cover.end());
    }
    return result.empty() ? empty() : Cells(result.begin(),result.end());
}
struct Proof {
    Cells moves;
    std::vector<std::pair<Cells,std::shared_ptr<Proof>>> responses;
};
struct Certificate { int attacker=0, turns=0; std::shared_ptr<Proof> tree; };
using ProofTable = std::map<std::string,std::shared_ptr<Proof>>;
inline Position apply_turn(Position game, const Cells& moves, bool strict=false) {
    if (moves.empty() || moves.size()>static_cast<std::size_t>(game.left)) throw std::invalid_argument("Incomplete or overlong turn.");
    for (std::size_t i=0; i<moves.size(); ++i) {
        game.play(moves[i]);
        if (game.done) { if (strict && i+1!=moves.size()) throw std::invalid_argument("Moves after game end."); break; }
    }
    return game;
}
std::shared_ptr<Proof> discover(const Position&, int attacker, int turns, int width, int candidates, Budget&, ProofTable&);
bool verify(const Position&, int attacker, int turns, const Proof&, Budget&);
Cells fork(const Position&, Budget&, int width=24, int candidates=256);
bool verifies(const Position&, const Cells&);
// LibTorch lives in inference.cpp so game/search code does not include its headers.
void predict(void* model, const float* input, int batch, int size, float* policy, float* value);
const char* last_error();
void set_error(const char*);
