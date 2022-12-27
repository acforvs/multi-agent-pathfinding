#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <vector>

using namespace std;

struct Pt {
  int x;
  int y;
  bool operator<(const Pt &other) const {
    if (x == other.x) return y < other.y;
    return x < other.x;
  }
  bool operator==(const Pt &other) const {
    return x == other.x && y == other.y;
  }
  bool operator!=(const Pt &other) const {
    return x != other.x || y != other.y;
  }
};

struct Node {
  vector<Pt> pts;
  int n;
  int d = 1e9;  // real_dist
  int h = 0;    // heur
  Node *prev = NULL;

  Node() { n = 0; }
  Node(vector<Pt> pts_) {
    pts = pts_;
    n = pts.size();
  }
  void calc_h(Node finish) {
    for (int i = 0; i < n; i++) {
      h += abs(pts[i].x - finish.pts[i].x) + abs(pts[i].y - finish.pts[i].y);
    }
  }

  void add_pt(Pt pt) {
    n += 1;
    pts.push_back(pt);
  }
  bool operator<(const Node &other) const { return pts < other.pts; }
};
bool eq(Node &l, Node &r) { return l.pts == r.pts; }

struct Grid {
  int m;
  vector<vector<int>> mp;  // 0 -- free, 1 -- nonfree
  Grid(int m_) {
    m = m_;
    mp.resize(m, vector<int>(m));
  }
  Grid(vector<vector<int>> mp_) {
    mp = mp_;
    m = mp.size();
  }
  vector<Pt> neigh(Pt nd) {
    vector<Pt> neighs;
    if (nd.x > 0 && mp[nd.x - 1][nd.y] == 0) {
      neighs.push_back({nd.x - 1, nd.y});
    }
    if (nd.x + 1 < m && mp[nd.x + 1][nd.y] == 0) {
      neighs.push_back({nd.x + 1, nd.y});
    }
    if (nd.y > 0 && mp[nd.x][nd.y - 1] == 0) {
      neighs.push_back({nd.x, nd.y - 1});
    }
    if (nd.y + 1 < m && mp[nd.x][nd.y + 1] == 0) {
      neighs.push_back({nd.x, nd.y + 1});
    }

    neighs.push_back(nd);
    return neighs;
  }

  bool collision(Node from, Node to) {
    set<Pt> simple_collision;
    set<pair<Pt, Pt>> hard_collisions;
    int n = to.n;
    for (int i = 0; i < n; i++) {
      if (simple_collision.count(to.pts[i])) return true;
      simple_collision.insert(to.pts[i]);

      if (to.pts[i] != from.pts[i]) {
        if (hard_collisions.count(make_pair(to.pts[i], from.pts[i])))
          return true;
        hard_collisions.insert(make_pair(from.pts[i], to.pts[i]));
      }
    }
    return false;
  }

  vector<Node> neigh(Node nd) {
    vector<vector<Pt>> res = {{}};
    for (int i = 0; i < nd.n; i++) {
      vector<vector<Pt>> res2;
      auto neighs = neigh(nd.pts[i]);
      for (auto pref : res) {
        for (auto step : neighs) {
          pref.push_back(step);
          res2.push_back(pref);
          pref.pop_back();
        }
      }
      swap(res, res2);
    }
    vector<Node> neighs;
    for (auto pts : res) {
      Node candidate = Node(pts);
      if (!collision(nd, candidate)) neighs.push_back(Node(pts));
    }
    return neighs;
  }
};

struct cmpByHeurDist {
  bool operator()(const Node &a, const Node &b) const {
    if (a.d + a.h == b.d + b.h) return a < b;
    return a.d + a.h < b.d + b.h;
  }
};

Node choose_node(set<Node, cmpByHeurDist> &nodes) { return *nodes.begin(); }

int id(Pt from, Pt to) {
  if (to.x == from.x + 1) {
    return 2;
  }
  if (to.x == from.x - 1) {
    return 1;
  }
  if (to.y == from.y + 1) {
    return 4;
  }
  if (to.y == from.y - 1) {
    return 3;
  }
  return 0;
}

vector<vector<int>> make_path(Node node, map<Node, Node> pr) {
  vector<vector<int>> res(node.n);
  Node cur = node;
  while (pr.count(cur)) {
    Node prev = pr[cur];
    for (int i = 0; i < node.n; i++) {
      res[i].push_back(id(prev.pts[i], cur.pts[i]));
    }
    cur = prev;
  }
  for (int i = 0; i < node.n; i++) reverse(res[i].begin(), res[i].end());
  return res;
};

void print_node(Node node) {
  for (int i = 0; i < node.n; i++) {
    cout << "(" << node.pts[i].x << ", " << node.pts[i].y << "), ";
  }
  cout << endl;
}

vector<vector<int>> Astar_multi(vector<vector<int>> mp, vector<Pt> starts,
                                vector<Pt> fins) {
  Grid grid(mp);
  Node start(starts);
  Node finish(fins);

  start.d = 0;
  start.calc_h(finish);

  set<Node, cmpByHeurDist> reachable = {start};
  set<Node> explored;

  map<Node, int> dist;
  dist[*reachable.begin()] = 0;
  map<Node, Node> pr;
  while (!reachable.empty()) {
    Node node = choose_node(reachable);

    if (eq(node, finish)) return make_path(node, pr);

    explored.insert(node);
    reachable.erase(node);

    auto neigs_nodes = grid.neigh(node);
    for (auto adjacent : neigs_nodes) {
      if (explored.count(adjacent)) continue;
      adjacent.calc_h(finish);
      if (!dist.count(adjacent)) {
        dist[adjacent] = 1e9;
        reachable.insert(adjacent);
      }
      adjacent.d = dist[adjacent];

      if (dist[node] + 1 < adjacent.d) {
        reachable.erase(adjacent);
        adjacent.d = dist[node] + 1;
        dist[adjacent] = adjacent.d;
        adjacent.prev = &node;
        pr[adjacent] = node;
        reachable.insert(adjacent);
      }
    }
  }

  return {};
}

int main(int argc, char *argv[]) {
  int cur = 1;
  auto next_token = [&]() {
    int x = atoi(argv[cur++]);
    return x;
  };

  int m = next_token();
  int n = next_token();
  vector<vector<int>> mp(m, vector<int>(m));
  for (auto &xx : mp)
    for (auto &x : xx) x = next_token();

  vector<Pt> starts(n);
  for (auto &[x, y] : starts) {
    x = next_token();
    y = next_token();
  }
  vector<Pt> fins(n);

  for (auto &[x, y] : fins) {
    x = next_token();
    y = next_token();
  }
  long double cur_time = clock();
  auto res = Astar_multi(mp, starts, fins);
  long double end_time = clock();
  for (auto x : res) {
    for (auto step : x) {
      cout << step << " ";
    }
    cout << endl;
  }
  auto delta = (end_time - cur_time) / CLOCKS_PER_SEC;
  cout << delta;
  return 0;
}
