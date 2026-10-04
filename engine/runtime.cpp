#include "runtime.h"
#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <string>
#include <thread>

extern "C" int puct_select(const double*, const std::int32_t*, const double*, int);
static thread_local std::string error_message;
const char* last_error() { return error_message.c_str(); }
void set_error(const char* message) { error_message=message; }
EXPORT const char* runtime_error() { return last_error(); }

static Cells shortest(const Edges& edges) {
    return *std::min_element(edges.begin(),edges.end(),[](const Cells& a,const Cells& b){return a.size()!=b.size() ? a.size()<b.size() : a<b;});
}
bool verifies(const Position& game, const Cells& moves) {
    try {
        auto child=apply_turn(game,moves,true);
        if (child.done) return child.winner==game.player;
        if (child.player==game.player || !child.threats(child.player,child.left).empty()) return false;
        return covers(child.threats(game.player,game.stones),child.left).empty();
    } catch (const std::invalid_argument&) { return false; }
}
static std::vector<Cells> defenses(const Position& game, const Edges& threats, Budget& budget) {
    auto minimal=covers(threats,game.left); budget.tick();
    Edges result; auto empty=game.empty(); int count=std::min(game.left,static_cast<int>(empty.size()));
    for (const auto& cover : minimal) {
        int rest=count-static_cast<int>(cover.size());
        if (rest==0) result.insert(cover);
        else if (rest==1) for (int cell : empty) {
            budget.tick(); if (std::binary_search(cover.begin(),cover.end(),cell)) continue;
            auto reply=cover; reply.push_back(cell); std::sort(reply.begin(),reply.end()); result.insert(reply);
        }
    }
    return {result.begin(),result.end()};
}
static std::vector<Cells> verified_defenses(const Position& game, const Edges& threats, Budget& budget) {
    // Independent reply reconstruction: never call the discovery cover generator here.
    Edges result; auto empty=game.empty(); int count=std::min(game.left,static_cast<int>(empty.size()));
    for (int first : empty) {
        budget.tick(); Edges uncovered;
        for (const auto& edge : threats) if (!std::binary_search(edge.begin(),edge.end(),first)) uncovered.insert(edge);
        if (count==1) { if (uncovered.empty()) result.insert({first}); continue; }
        if (uncovered.empty()) {
            for (int second : empty) { budget.tick(); if (first!=second) result.insert({std::min(first,second),std::max(first,second)}); }
        } else {
            Cells possible=*uncovered.begin();
            for (const auto& edge : uncovered) {
                Cells common; std::set_intersection(possible.begin(),possible.end(),edge.begin(),edge.end(),std::back_inserter(common)); possible=std::move(common);
            }
            for (int second : possible) if (second!=first) { budget.tick(); result.insert({std::min(first,second),std::max(first,second)}); }
        }
    }
    return {result.begin(),result.end()};
}
static std::vector<Cells> candidates(const Position& game, int width, int maximum, Budget& budget, bool short_fork=false) {
    auto own=game.threats(game.player,game.left); if (!own.empty()) return {shortest(own)};
    int needed=std::min(game.left,static_cast<int>(game.empty().size()));
    std::set<int> pool; int windows=0;
    for (const auto& line : game.lines()) {
        int own_count=0, enemy=0;
        for (int cell : line) { own_count+=game.board[cell]==game.player; enemy+=game.board[cell]==-game.player; }
        if (!enemy && own_count>=game.k-game.stones-needed) {
            ++windows; for (int cell : line) if (!game.board[cell]) pool.insert(cell);
        }
    }
    if (short_fork && windows<=game.stones) return {};
    if (short_fork) for (const auto& edge : game.threats(-game.player,game.stones)) pool.insert(edge.begin(),edge.end());
    auto weights=game.scores(); Cells ranked(pool.begin(),pool.end());
    std::stable_sort(ranked.begin(),ranked.end(),[&](int a,int b){return weights[a]!=weights[b] ? weights[a]>weights[b] : a<b;});
    if (ranked.size()>static_cast<std::size_t>(width)) ranked.resize(width);
    // ponytail: bounded attack shortlist can miss wins; widen after measured benefit. Defense checks remain complete.
    std::vector<Cells> result;
    for (std::size_t a=0; a<ranked.size(); ++a) {
        if (needed==1) { budget.tick(); result.push_back({ranked[a]}); }
        else for (std::size_t b=a+1; b<ranked.size(); ++b) { budget.tick(); result.push_back({ranked[a],ranked[b]}); }
    }
    if (needed==2 || !short_fork) std::stable_sort(result.begin(),result.end(),[&](const Cells& a,const Cells& b){
        double x=0,y=0; for (int cell : a) x+=weights[cell]; for (int cell : b) y+=weights[cell];
        return x!=y ? x>y : (!short_fork && a<b);
    });
    if (result.size()>static_cast<std::size_t>(maximum)) result.resize(maximum);
    return result;
}
Cells fork(const Position& game, Budget& budget, int width, int maximum) {
    if (game.done) return {};
    auto own=game.threats(game.player,game.left); if (!own.empty()) return shortest(own);
    auto enemy=game.threats(-game.player,game.stones);
    for (const auto& moves : candidates(game,width,maximum,budget,true)) {
        budget.tick(); bool blocked=true;
        for (const auto& edge : enemy) {
            bool hit=false; for (int cell : moves) hit |= std::binary_search(edge.begin(),edge.end(),cell); blocked &= hit;
        }
        if (blocked && verifies(game,moves)) return moves;
    }
    return {};
}
std::shared_ptr<Proof> discover(const Position& game, int attacker, int turns, int width, int maximum, Budget& budget) {
    budget.tick(); if (game.done || game.player!=attacker || turns<1) return {};
    auto own=game.threats(attacker,game.left);
    if (!own.empty()) {
        auto moves=shortest(own); auto child=apply_turn(game,moves);
        if (child.done && child.winner==attacker) return std::make_shared<Proof>(Proof{moves,{}});
    }
    if (turns<2) return {};
    for (const auto& moves : candidates(game,width,maximum,budget)) {
        auto child=apply_turn(game,moves); Cells played(child.moves.end()-static_cast<std::ptrdiff_t>(child.moves.size()-game.moves.size()),child.moves.end());
        auto proof=std::make_shared<Proof>(Proof{played,{}});
        if (child.done) { if (child.winner==attacker) return proof; continue; }
        if (child.player==attacker) continue;
        auto threats=child.threats(attacker,game.stones);
        if (threats.empty() || !child.threats(child.player,child.left).empty()) continue;
        auto replies=defenses(child,threats,budget); bool complete=true;
        for (const auto& reply : replies) {
            budget.tick(); auto position=apply_turn(child,reply);
            if (position.done || position.player!=attacker) { complete=false; break; }
            auto continuation=discover(position,attacker,turns-1,width,maximum,budget);
            if (!continuation) { complete=false; break; }
            proof->responses.emplace_back(reply,std::move(continuation));
        }
        if (complete) return proof;
    }
    return {};
}
bool verify(const Position& game, int attacker, int turns, const Proof& proof, Budget& budget) {
    budget.tick();
    Position child=game;
    try { child=apply_turn(game,proof.moves,true); } catch (const std::invalid_argument&) { return false; }
    if (child.done) return child.winner==attacker && proof.responses.empty();
    if (child.player==attacker || turns<2) return false;
    auto threats=child.threats(attacker,game.stones);
    if (threats.empty() || !child.threats(child.player,child.left).empty()) return false;
    auto replies=verified_defenses(child,threats,budget);
    if (replies.size()!=proof.responses.size()) return false;
    for (std::size_t i=0; i<replies.size(); ++i) {
        budget.tick(); const auto& row=proof.responses[i]; if (row.first!=replies[i] || !row.second) return false;
        try {
            auto position=apply_turn(child,row.first,true);
            if (position.done || position.player!=attacker || !verify(position,attacker,turns-1,*row.second,budget)) return false;
        } catch (const std::invalid_argument&) { return false; }
    }
    return true;
}
static void encode_node(const Proof& proof, Cells& out) {
    out.push_back(static_cast<int>(proof.moves.size())); out.insert(out.end(),proof.moves.begin(),proof.moves.end());
    out.push_back(static_cast<int>(proof.responses.size()));
    for (const auto& [moves,node] : proof.responses) {
        out.push_back(static_cast<int>(moves.size())); out.insert(out.end(),moves.begin(),moves.end()); encode_node(*node,out);
    }
}
static std::shared_ptr<Proof> decode_node(const int*& cursor, const int* end, int depth) {
    if (depth>20 || cursor==end) throw std::invalid_argument("Invalid proof.");
    int count=*cursor++; if (count<1 || count>2 || end-cursor<count+1) throw std::invalid_argument("Invalid proof moves.");
    auto proof=std::make_shared<Proof>(); proof->moves.assign(cursor,cursor+count); cursor+=count;
    int replies=*cursor++; if (replies<0 || replies>200000) throw std::invalid_argument("Invalid proof replies.");
    for (int i=0; i<replies; ++i) {
        if (cursor==end) throw std::invalid_argument("Truncated proof.");
        count=*cursor++; if (count<1 || count>2 || end-cursor<count) throw std::invalid_argument("Invalid reply.");
        Cells reply(cursor,cursor+count); cursor+=count; proof->responses.emplace_back(std::move(reply),decode_node(cursor,end,depth+1));
    }
    return proof;
}
static Certificate decode(const int* data, int size) {
    if (size<4 || size>1000000 || data[0]!=1 || (data[1]!=1 && data[1]!=-1) || data[2]<1 || data[2]>20)
        throw std::invalid_argument("Invalid certificate.");
    const int* cursor=data+3; auto tree=decode_node(cursor,data+size,1);
    if (cursor!=data+size) throw std::invalid_argument("Extra proof data.");
    return {data[1],data[2],std::move(tree)};
}
static Cells encode(const Certificate& proof) { Cells out{1,proof.attacker,proof.turns}; if (proof.tree) encode_node(*proof.tree,out); return out; }

// A small per-call worker pool is reused for every simulation and root in that call.
// ponytail: recreate workers between batches; keep a persistent pool only if startup is measured as costly.
class Workers {
    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable start, finished;
    std::function<void(int)> job;
    std::atomic<int> next{0};
    int size=0, running=0, generation=0;
    bool stopping=false;
    std::exception_ptr failure;
public:
    explicit Workers(int count) {
        try { for (int i=1; i<count; ++i) threads.emplace_back([this] {
            int seen=0;
            for (;;) {
                std::unique_lock lock(mutex); start.wait(lock,[&]{return stopping || generation!=seen;});
                if (stopping) return; seen=generation; lock.unlock(); work(); lock.lock();
                if (--running==0) finished.notify_one();
            }
        }); } catch (...) { close(); throw; }
    }
    void work() {
        try { for (int index; (index=next.fetch_add(1))<size;) job(index); }
        catch (...) { std::lock_guard lock(mutex); if (!failure) failure=std::current_exception(); }
    }
    void each(int count, std::function<void(int)> function) {
        { std::lock_guard lock(mutex); job=std::move(function); size=count; next=0; failure=nullptr; running=static_cast<int>(threads.size()); ++generation; }
        start.notify_all(); work();
        { std::unique_lock lock(mutex); finished.wait(lock,[&]{return running==0;}); if (failure) std::rethrow_exception(failure); }
    }
    void close() { { std::lock_guard lock(mutex); stopping=true; } start.notify_all(); for (auto& thread : threads) if (thread.joinable()) thread.join(); }
    ~Workers() { close(); }
};
struct Node {
    Position game;
    int parent=-1, edge=-1;
    Cells actions, visits, children;
    std::vector<double> prior,total;
};
struct Tree {
    std::vector<Node> nodes;
    explicit Tree(Position game) { nodes.push_back(Node{std::move(game)}); }
    void backup(int id,double value) {
        int player=nodes[id].game.player;
        while (nodes[id].parent>=0) {
            int edge=nodes[id].edge; id=nodes[id].parent;
            ++nodes[id].visits[edge]; nodes[id].total[edge]+=nodes[id].game.player==player ? value : -value;
        }
    }
    int advance() {
        int id=0;
        while (!nodes[id].game.done && !nodes[id].actions.empty()) {
            const auto& node=nodes[id];
            int best=puct_select(node.prior.data(),node.visits.data(),node.total.data(),static_cast<int>(node.actions.size()));
            if (node.children[best]<0) {
                auto child=node.game; child.play(node.actions[best]); int next_id=static_cast<int>(nodes.size());
                nodes.push_back(Node{std::move(child),id,best}); nodes[id].children[best]=next_id; return next_id;
            }
            id=node.children[best];
        }
        return id;
    }
    void expand(int id, const float* logits, std::mt19937_64* rng=nullptr) {
        auto& node=nodes[id]; node.actions=node.game.actions();
        if (node.actions.empty()) throw std::invalid_argument("Cannot expand a terminal game.");
        double maximum=-INFINITY;
        for (int a : node.actions) { if (!std::isfinite(logits[a])) throw std::runtime_error("Non-finite model policy."); maximum=std::max(maximum,double(logits[a])); }
        double sum=0;
        for (int a : node.actions) { node.prior.push_back(std::exp(logits[a]-maximum)); sum+=node.prior.back(); }
        for (double& p : node.prior) p/=sum;
        if (rng) {
            std::gamma_distribution<double> gamma(.1,1); std::vector<double> noise; double total=0;
            do { noise.clear(); total=0; for (auto a : node.actions) { (void)a; noise.push_back(gamma(*rng)); total+=noise.back(); } } while (total==0);
            for (std::size_t i=0; i<noise.size(); ++i) node.prior[i]=.75*node.prior[i]+.25*noise[i]/total;
        }
        node.visits.assign(node.actions.size(),0); node.total.assign(node.actions.size(),0); node.children.assign(node.actions.size(),-1);
    }
    std::vector<float> policy() const {
        const auto& root=nodes[0]; std::vector<float> out(root.game.n*root.game.n,0);
        int sum=std::accumulate(root.visits.begin(),root.visits.end(),0);
        for (std::size_t i=0; i<root.actions.size(); ++i) out[root.actions[i]]=static_cast<float>(sum ? double(root.visits[i])/sum : root.prior[i]);
        return out;
    }
};
static void mcts(const std::vector<Position>& games, void* model, int simulations, std::mt19937_64* rng,
                 Budget& budget, Workers& workers, std::vector<std::vector<float>>& output, double& inference, int& completed) {
    if (!model) throw std::invalid_argument("Model required for PUCT search.");
    std::vector<Tree> trees; for (const auto& game : games) trees.emplace_back(game);
    int area=games[0].n*games[0].n;
    auto evaluate=[&](const Cells& roots,const Cells& ids,bool initial) {
        std::vector<float> features(roots.size()*8*area), policy(roots.size()*area), values(roots.size());
        workers.each(static_cast<int>(roots.size()),[&](int i){ auto planes=trees[roots[i]].nodes[ids[i]].game.features(); std::copy(planes.begin(),planes.end(),features.begin()+i*8*area); });
        auto began=Clock::now(); predict(model,features.data(),static_cast<int>(roots.size()),games[0].n,policy.data(),values.data()); inference+=seconds(began);
        for (std::size_t i=0; i<roots.size(); ++i) {
            if (!std::isfinite(values[i])) throw std::runtime_error("Non-finite model value.");
            auto& tree=trees[roots[i]]; tree.expand(ids[i],policy.data()+i*area,initial ? rng : nullptr);
            if (!initial) tree.backup(ids[i],values[i]);
        }
    };
    Cells roots(games.size()), ids(games.size(),0); std::iota(roots.begin(),roots.end(),0); evaluate(roots,ids,true);
    completed=0;
    for (int step=0; step<simulations; ++step) {
        try { budget.tick(); } catch (const Limit&) { break; }
        workers.each(static_cast<int>(trees.size()),[&](int i){ ids[i]=trees[i].advance(); });
        Cells pending,leaves;
        for (std::size_t i=0; i<trees.size(); ++i) {
            const auto& node=trees[i].nodes[ids[i]];
            if (node.game.done) trees[i].backup(ids[i],node.game.winner*node.game.player);
            else { pending.push_back(static_cast<int>(i)); leaves.push_back(ids[i]); }
        }
        if (!pending.empty()) evaluate(pending,leaves,false);
        ++completed;
    }
    for (const auto& tree : trees) output.push_back(tree.policy());
}
static Budget make_budget(double duration, int nodes, Stop stopped) {
    if (std::isnan(duration) || nodes<1) throw std::invalid_argument("Invalid time/node budget.");
    duration=std::clamp(duration,0.,604800.);
    return {Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(duration)),nodes,stopped};
}
#define FAIL catch (const std::exception& e) { set_error(e.what()); return -1; } catch (...) { set_error("Native operation failed."); return -1; }
EXPORT int game_play(std::int8_t* board, int* meta, const int* moves, int count, int action) {
    try { Position game(board,meta,moves,count); game.play(action); std::copy(game.board.begin(),game.board.end(),board); meta[4]=game.player; meta[5]=game.left; meta[6]=game.winner; meta[7]=game.done; return 0; } FAIL
}
EXPORT int game_query(const std::int8_t* board, const int* meta, const int* moves, int count, int op, int argument, int* cells, float* planes) {
    try {
        Position game(board,meta,moves,count);
        if (op==0) { auto features=game.features(); std::copy(features.begin(),features.end(),planes); return 0; }
        if (op==1) { auto legal=game.actions(); std::copy(legal.begin(),legal.end(),cells); return static_cast<int>(legal.size()); }
        if (op==2) return game.heuristic();
        if (op==3 || op==4) {
            if (argument<0 || argument>2) throw std::invalid_argument("Threat budget must be 0..2.");
            auto edges=game.threats(op==3 ? game.player : -game.player,argument); int offset=0;
            for (const auto& edge : edges) { cells[offset++]=static_cast<int>(edge.size()); for (int cell : edge) cells[offset++]=cell; }
            return offset;
        }
        throw std::invalid_argument("Unknown game query.");
    } FAIL
}
EXPORT int game_start(std::int8_t* board, int* meta, int* history, int* count, std::uint64_t seed, int mode) {
    try {
        int n=meta[0]; if (n<2 || n>25) throw std::invalid_argument("Invalid board size.");
        std::vector<std::int8_t> empty(n*n,0); int state[8]={n,meta[1],meta[2],meta[3],1,meta[3],0,0};
        Position base(empty.data(),state,history,0), game=base; std::mt19937_64 rng(seed);
        if (mode==0) {
            if (std::generate_canonical<double,53>(rng)<.5) {
                Cells pool; for (int r=std::max(0,n/2-1); r<std::min(n,n/2+2); ++r) for (int c=std::max(0,n/2-1); c<std::min(n,n/2+2); ++c) pool.push_back(r*n+c);
                std::shuffle(pool.begin(),pool.end(),rng);
                for (int i=0; i<game.starter && !game.done; ++i) game.play(pool[i]);
                if (game.done) game=base;
            }
        } else if (mode==1) {
            for (int attempt=0; attempt<20; ++attempt) {
                game=base; auto pool=game.empty(); int center=(n/2)*n+n/2; pool.erase(std::find(pool.begin(),pool.end(),center)); std::shuffle(pool.begin(),pool.end(),rng);
                int length=std::min(game.starter+2*game.stones,n*n-1); game.play(center);
                for (int i=1; i<length && !game.done; ++i) game.play(pool[i-1]);
                if (!game.done) break;
            }
            if (game.done) game=base;
        } else if (mode==2) {
            if (*count<0 || *count>n*n) throw std::invalid_argument("Invalid history length.");
            for (int i=0; i<*count; ++i) game.play(history[i]);
        } else throw std::invalid_argument("Unknown opening mode.");
        std::copy(game.board.begin(),game.board.end(),board); std::copy(game.moves.begin(),game.moves.end(),history); *count=static_cast<int>(game.moves.size());
        meta[4]=game.player; meta[5]=game.left; meta[6]=game.winner; meta[7]=game.done; return 0;
    } FAIL
}
EXPORT int tactical(const std::int8_t* board,const int* meta,const int* moves,int count,
                    int operation,const int* input,int length,int turns,int width,int candidates,int nodes,double duration,Stop stopped,int* output,int capacity) {
    try {
        auto budget=make_budget(duration,nodes,stopped); budget.tick(); Position game(board,meta,moves,count);
        if (operation==0) {
            if (turns<1 || turns>20 || width<2 || width>64 || candidates<1) throw std::invalid_argument("Invalid TSS limits.");
            auto tree=discover(game,game.player,turns,width,candidates,budget); if (!tree) return 0;
            budget.tick(); auto encoded=encode({game.player,turns,tree});
            if (encoded.size()>static_cast<std::size_t>(capacity)) throw std::runtime_error("Proof exceeds output limit.");
            std::copy(encoded.begin(),encoded.end(),output); return static_cast<int>(encoded.size());
        }
        if (operation==1) {
            Certificate proof;
            try { proof=decode(input,length); } catch (const std::invalid_argument&) { return 0; }
            return proof.attacker==game.player && verify(game,proof.attacker,proof.turns,*proof.tree,budget) ? 1 : 0;
        }
        if (operation==2) { auto found=fork(game,budget,width,candidates); std::copy(found.begin(),found.end(),output); return static_cast<int>(found.size()); }
        if (operation==3) return length>0 && length<=2 && verifies(game,Cells(input,input+length)) ? 1 : 0;
        throw std::invalid_argument("Unknown tactical operation.");
    } catch (const Limit&) { return -2; } FAIL
}
EXPORT int runtime_search(const std::int8_t* boards,const int* metadata,const int* histories,const int* offsets,int count,
                          void* model,int simulations,int workers,std::uint64_t seed,int noise,double duration,Stop stopped,float* policies,double* timings) {
    try {
        if (count<1 || count>128 || workers<1 || workers>12 || simulations<1 || simulations>100000) throw std::invalid_argument("Invalid search limits.");
        auto began=Clock::now(); int n=metadata[0]; if (n<2 || n>25) throw std::invalid_argument("Invalid board size.");
        int area=n*n;
        std::vector<Position> games;
        for (int i=0; i<count; ++i) {
            if (offsets[i]<0 || offsets[i+1]<offsets[i] || offsets[i+1]-offsets[i]>area) throw std::invalid_argument("Invalid history offsets.");
            games.emplace_back(boards+i*area,metadata+i*8,histories ? histories+offsets[i] : nullptr,offsets[i+1]-offsets[i]);
            if (games.back().done || games.back().n!=games[0].n || games.back().k!=games[0].k || games.back().stones!=games[0].stones || games.back().starter!=games[0].starter) throw std::invalid_argument("Search needs matching live games.");
        }
        Workers pool(std::min(workers,count)); auto budget=make_budget(duration,100001,stopped); std::mt19937_64 rng(seed);
        std::vector<std::vector<float>> result; double inference=0; int completed=0;
        mcts(games,model,simulations,noise ? &rng : nullptr,budget,pool,result,inference,completed);
        for (int i=0; i<count; ++i) std::copy(result[i].begin(),result[i].end(),policies+i*area);
        timings[0]=std::max(0.,seconds(began)-inference); timings[1]=inference; return completed;
    } FAIL
}
EXPORT int runtime_choose(const float* policy,int area,std::uint64_t seed,double temperature) {
    try {
        if (area<4 || area>625 || !std::isfinite(temperature) || temperature<0) throw std::invalid_argument("Invalid sampling settings.");
        std::vector<double> p(area); double sum=0;
        for (int i=0; i<area; ++i) {
            if (!std::isfinite(policy[i]) || policy[i]<0) throw std::invalid_argument("Invalid policy.");
            p[i]=temperature==0 ? policy[i] : std::pow(policy[i],1/temperature); sum+=p[i];
        }
        if (sum<=0) throw std::invalid_argument("Empty policy.");
        if (temperature==0) return static_cast<int>(std::max_element(p.begin(),p.end())-p.begin());
        std::mt19937_64 rng(seed); return std::discrete_distribution<int>(p.begin(),p.end())(rng);
    } FAIL
}

struct Strategy { Cells history; Certificate proof; };
static Strategy read_strategy(const int* data,int size) {
    if (size<1 || data[0]<0 || data[0]>625 || size<data[0]+5) throw std::invalid_argument("Invalid strategy history.");
    int count=data[0]; return {Cells(data+1,data+1+count),decode(data+1+count,size-1-count)};
}
static Cells write_strategy(const Strategy& strategy) {
    if (!strategy.proof.tree) return {};
    Cells out{static_cast<int>(strategy.history.size())}; out.insert(out.end(),strategy.history.begin(),strategy.history.end());
    auto proof=encode(strategy.proof); out.insert(out.end(),proof.begin(),proof.end()); return out;
}
static Certificate resume(const Position& game,const Strategy& saved,Budget& budget) {
    if (!saved.proof.tree || saved.proof.attacker!=game.player || saved.history.size()>game.moves.size()
        || !std::equal(saved.history.begin(),saved.history.end(),game.moves.begin())) return {};
    auto root=game; root.board.assign(game.n*game.n,0); root.moves.clear(); root.player=1; root.left=game.starter; root.winner=root.done=0;
    for (int action : saved.history) root.play(action);
    if (root.player!=game.player || !verify(root,saved.proof.attacker,saved.proof.turns,*saved.proof.tree,budget)) return {};
    auto certificate=saved.proof; const auto& attack=certificate.tree->moves;
    Cells suffix(game.moves.begin()+saved.history.size(),game.moves.end());
    if (suffix.size()<attack.size()) {
        if (!std::equal(suffix.begin(),suffix.end(),attack.begin())) return {};
        auto node=std::make_shared<Proof>(*certificate.tree); node->moves.erase(node->moves.begin(),node->moves.begin()+suffix.size()); certificate.tree=node;
    } else {
        if (!std::equal(attack.begin(),attack.end(),suffix.begin()) || suffix.size()==attack.size() || certificate.turns<2) return {};
        Cells reply(suffix.begin()+attack.size(),suffix.end()); std::shared_ptr<Proof> next;
        for (const auto& row : certificate.tree->responses) if (row.first==reply) { next=row.second; break; }
        if (!next) return {}; certificate.tree=next; --certificate.turns;
    }
    if (!verify(game,game.player,certificate.turns,*certificate.tree,budget)) return {};
    return certificate;
}
EXPORT int runtime_decide(const std::int8_t* boards,const int* metadata,const int* histories,const int* offsets,int count,
                          void* model,int simulations,int workers,std::uint64_t seed,int mode,int bootstrap,int tss,double tactical_ms,
                          double duration,Stop stopped,const int* stored,const int* stored_offsets,const int* plans,
                          float* policies,int* actions,int* sources,int* out_plans,int* out_strategies,int* out_offsets,int capacity,double* timings) {
    try {
        if (count<1 || count>128 || workers<1 || workers>12 || simulations<1 || simulations>100000 || (mode!=0 && mode!=1)
            || tactical_ms<0 || !std::isfinite(tactical_ms)) throw std::invalid_argument("Invalid batch settings.");
        auto began=Clock::now(); auto overall=make_budget(duration,1000001,stopped); overall.tick();
        int n=metadata[0]; if (n<2 || n>25) throw std::invalid_argument("Invalid board size.");
        int area=n*n;
        std::vector<Position> games;
        for (int i=0; i<count; ++i) {
            if (offsets[i]<0 || offsets[i+1]<offsets[i] || offsets[i+1]-offsets[i]>area) throw std::invalid_argument("Invalid history offsets.");
            games.emplace_back(boards+i*area,metadata+i*8,histories ? histories+offsets[i] : nullptr,offsets[i+1]-offsets[i]);
            if (games.back().done || games.back().n!=n || games.back().k!=games[0].k || games.back().stones!=games[0].stones || games.back().starter!=games[0].starter) throw std::invalid_argument("Batch needs matching live games.");
        }
        std::vector<Strategy> strategies(count); std::vector<Cells> chosen_plans(count);
        std::vector<std::vector<float>> results(count,std::vector<float>(area,0)); Cells unresolved;
        std::mt19937_64 rng(seed); Workers pool(std::min(workers,count));
        std::fill_n(sources,count,-1); std::fill_n(out_plans,2*count,-1);
        for (int i=0; i<count; ++i) {
            overall.tick(); const auto& game=games[i]; Cells proof;
            if (stored_offsets[i]<0 || stored_offsets[i+1]<stored_offsets[i] || stored_offsets[i+1]-stored_offsets[i]>1000000) throw std::invalid_argument("Invalid strategy offsets.");
            if (stored_offsets[i+1]>stored_offsets[i]) {
                try { strategies[i]=read_strategy(stored+stored_offsets[i],stored_offsets[i+1]-stored_offsets[i]); }
                catch (const std::invalid_argument&) { strategies[i]={}; }
            }
            for (int j=0; j<2; ++j) if (plans[2*i+j]>=0) proof.push_back(plans[2*i+j]);
            if (!proof.empty() && !verifies(game,proof)) proof.clear();
            auto own=game.threats(game.player,game.left); if (!own.empty()) proof=shortest(own);
            double allowance=mode==0 ? tactical_ms/1000 : (tactical_ms>0 ? std::min(.05,std::max(.002,duration*.1)) : 0);
            if (proof.empty() && tss && strategies[i].proof.tree && allowance>0) {
                auto budget=make_budget(allowance,20000,stopped);
                try {
                    auto resumed=resume(game,strategies[i],budget);
                    if (resumed.tree) { strategies[i]={game.moves,resumed}; chosen_plans[i]=resumed.tree->moves; sources[i]=4; }
                    else strategies[i]={};
                } catch (const Limit&) { strategies[i]={}; }
            }
            if (proof.empty() && sources[i]<0 && allowance>0 && (mode==0 || duration>.02)) {
                auto expires=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(allowance));
                if (tss) {
                    Budget discovery{Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(allowance*(mode==0 ? .9 : .7))),mode==0 ? 10000 : 20000,stopped};
                    Certificate found;
                    try {
                        for (int horizon : (mode==0 ? Cells{2,3} : Cells{3})) {
                            found={game.player,horizon,discover(game,game.player,horizon,24,1000,discovery)}; if (found.tree) break;
                        }
                    } catch (const Limit&) { found={}; }
                    if (found.tree) {
                        Budget checking{expires,mode==0 ? 10000 : 100000,stopped};
                        try {
                            if (!verify(game,game.player,found.turns,*found.tree,checking)) throw std::runtime_error("Rejected an invalid native TSS proof.");
                            strategies[i]={game.moves,found}; chosen_plans[i]=found.tree->moves; sources[i]=4;
                        } catch (const Limit&) { /* Unknown: use normal search, never a value label. */ }
                    }
                }
                if (sources[i]<0) { Budget budget{expires,100000,stopped}; try { proof=fork(game,budget); } catch (const Limit&) {} }
            }
            if (!proof.empty() && sources[i]<0) {
                if (!verifies(game,proof)) throw std::runtime_error("Rejected an invalid native tactical certificate.");
                chosen_plans[i]=proof; sources[i]=3;
            }
            if (sources[i]>=0) results[i][chosen_plans[i][0]]=1;
            else {
                auto legal=game.actions();
                if (legal.size()==1 || bootstrap || (mode==1 && (!model || duration<=.02))) {
                    int action=legal.size()==1 ? legal[0] : game.heuristic(); results[i][action]=1; sources[i]=legal.size()==1 ? 2 : 1;
                } else unresolved.push_back(i);
            }
        }
        double inference=0; int completed=0;
        if (!unresolved.empty()) {
            std::vector<Position> pending; for (int index : unresolved) pending.push_back(games[index]);
            std::vector<std::vector<float>> output;
            auto search_budget=make_budget(std::max(0.,duration-seconds(began)-(mode==1 ? .02 : 0)),100001,stopped);
            mcts(pending,model,simulations,mode==0 ? &rng : nullptr,search_budget,pool,output,inference,completed);
            for (std::size_t i=0; i<unresolved.size(); ++i) { results[unresolved[i]]=std::move(output[i]); sources[unresolved[i]]=0; }
        }
        overall.tick(); int used=0; out_offsets[0]=0;
        for (int i=0; i<count; ++i) {
            std::copy(results[i].begin(),results[i].end(),policies+i*area);
            actions[i]=runtime_choose(results[i].data(),area,rng(),mode==1 ? 0 : (games[i].moves.size()<12 ? 1 : .25));
            if (actions[i]<0) throw std::runtime_error(last_error());
            for (std::size_t j=0; j<chosen_plans[i].size(); ++j) out_plans[2*i+j]=chosen_plans[i][j];
            auto encoded=write_strategy(strategies[i]);
            if (used+encoded.size()>static_cast<std::size_t>(capacity)) throw std::runtime_error("Strategy batch exceeds output limit.");
            std::copy(encoded.begin(),encoded.end(),out_strategies+used); used+=static_cast<int>(encoded.size()); out_offsets[i+1]=used;
        }
        timings[0]=std::max(0.,seconds(began)-inference); timings[1]=inference; return completed;
    } catch (const Limit&) { return -2; } FAIL
}
EXPORT int runtime_turn(const std::int8_t* board,const int* meta,const int* history,int count,void* model,double duration,
                        const int* stored,int stored_count,int* output,int* out_strategy,int* out_count,int capacity,double* elapsed) {
    try {
        if (!std::isfinite(duration) || duration<0 || duration>30 || stored_count<0 || stored_count>capacity) throw std::invalid_argument("Invalid full-turn request.");
        Position game(board,meta,history,count); if (game.done) throw std::invalid_argument("Game has ended.");
        auto began=Clock::now(); int color=game.player, played=0;
        Cells strategy(stored,stored+stored_count); int plan[2]={-1,-1};
        while (!game.done && game.player==color) {
            int state[8]={game.n,game.k,game.stones,game.starter,game.player,game.left,game.winner,game.done};
            int offsets[2]={0,static_cast<int>(game.moves.size())}, saved_offsets[2]={0,static_cast<int>(strategy.size())};
            int chosen[1], source[1], next_plan[2], next_offsets[2]; double timings[2];
            std::vector<float> policy(game.n*game.n); Cells next_strategy(capacity);
            double remaining=std::max(0.,duration-seconds(began))/game.left;
            int result=runtime_decide(game.board.data(),state,game.moves.data(),offsets,1,model,100000,1,0,1,0,1,50,
                remaining,nullptr,strategy.data(),saved_offsets,plan,policy.data(),chosen,source,next_plan,
                next_strategy.data(),next_offsets,capacity,timings);
            if (result==-1) throw std::runtime_error(last_error());
            int action=result==-2 ? game.heuristic() : chosen[0]; game.play(action); output[played++]=action;
            if (result!=-2) {
                strategy.assign(next_strategy.begin(),next_strategy.begin()+next_offsets[1]);
                plan[0]=next_plan[1]; plan[1]=-1;
            }
        }
        std::copy(strategy.begin(),strategy.end(),out_strategy); *out_count=static_cast<int>(strategy.size()); *elapsed=seconds(began); return played;
    } FAIL
}
