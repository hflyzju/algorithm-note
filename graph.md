


# 一、总结
|  类型 | 难度  | 题目 | 题解 | 
|  ----  | ----  | --- | --- |
|图的遍历|中等|797.所有可能的路径|进入节点加入path，遍历完子节点后从path中pop出来|
|二分图判定|中等|785.判断二分图, 886.可能的二分法|遍历图的时候进行着色，没有访问过的，着不同的颜色，访问过的，检查着色是否和当前节点冲突|
|拓扑排序|中等|207.课程表， 210.课程表II|先统计indree为0的节点，然后先处理indree为0的节点，每处理完一个，检查其邻居indgree是否为0，如果是0则添加到后面进行处理，最后统计完成课程的数量进行输出。|
|并查集|hard|765.情侣牵手|最小交换次数：n-并查集的n|
|树，图|其他|其他|树：没有环，图：有环|
|最小生成树|hard|261.以图判树|判断这些边能否组成树（没有环），利用并查集，如果遇到新的两个节点暂时还不属于一个union，那么没问题，如果属于同一个union，代表之前已经加进来了，代表有环。|
|最小生成树|hard|1135.最低成本联通所有城市|cost按从小到大排序，遇到的两个节点，如果两个节点不属于同一个union（没有环），则该条边是最小生成树的一部分，将它加入mst集合；否则，不是，不要加入，最终判断并查集n的个树|
|最短路径|mid|743.网络延迟时间|从起点出发，每次拿到当前最短的路径，然后更新起邻居的cost，如果邻居的路径有变短，则将邻居作为候选加入优先队列，时间复杂度（mlogn）|

# 二、模板


## 2.1 图的dfs遍历模板

```python
n = len(graph)
res = []
path = []
def search(cur):
    path.append(cur)
    if cur == n - 1:
        res.append(path[:])
    elif cur < n:
        for child in graph[cur]:
            search(child)
    path.pop()
search(0)
return res

```

## 2.2 二分图着色模板

```c++
visited[cur] = true;
for(int i=0; i < graph[cur].size(); i++) {
    int child = graph[cur][i];
    if(!visited[child]) {
        color[child] = ~color[cur];
        if(!dfs(visited, color, graph, child)) {
            return false;
        }
    } else {
        if(color[child] == color[cur]) {
            return false;
        }
    }
}
return true;
```

## 2.3 并查集模板

```c++
class UnionFind {

    private:
        vector<int> parent;
        int n;
    public:
        UnionFind(int size) {
            n = size;
            for(int i=0; i<size; i++) {
                parent.push_back(i);
            }
        }
        int getRoot(int x) {
            while(x != parent[x]) {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        }
        void add(int x, int y) {
            int root_x = getRoot(x);
            int root_y = getRoot(y);
            if(root_x != root_y) {
                parent[root_x] = root_y;
                n -= 1;
            }
        }
        int getCnt() {
            return n;
        }
};

```


## 2.4 dijstra模板

```python
cost = [float('inf')] * (n+1)
cost[k] = 0
cache = []
heapq.heappush(cache, [0, k])

graph = [{} for _ in range(n+1)]
for x,y,c in times:
    graph[x][y] = c

while cache:
    cur_cost, cur = heapq.heappop(cache)
    if cost[cur] <= cur_cost:
        for nei in graph[cur]:
            nei_cost = cur_cost + graph[cur][nei]
            if nei_cost < cost[nei]:
                cost[nei] = nei_cost
                heapq.heappush(cache, [nei_cost, nei])
```


# 三、题解


### 3.1 图的遍历

#### 797 所有可能的路径

```python
class Solution(object):
    def allPathsSourceTarget(self, graph):
        """
        :type graph: List[List[int]]
        :rtype: List[List[int]]
        """
        n = len(graph)
        res = []
        path = []
        def search(cur):
            path.append(cur)
            if cur == n - 1:
                res.append(path[:])
            elif cur < n:
                for child in graph[cur]:
                    search(child)
            path.pop()
        search(0)
        return res
```

```c++
class Solution {
public:

    vector<int> path;
    vector<vector<int>> res;


    void dfs(vector<vector<int>>& graph, vector<int>& path, int cur) {
        path.push_back(cur);
        if(cur == graph.size()-1) {
            res.push_back(path);
        } else if (cur < graph.size()) {
            for(int i=0; i<graph[cur].size();i++) {
                int child = graph[cur][i];
                dfs(graph, path, child);
            }
        }
        path.pop_back();
    }


    vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph) {

        dfs(graph, path, 0);
        return res;
    }
};
```

### 3.2 二分图

#### 785. 判断二分图
```c++
class Solution {
/*

输入：graph = [[1,2,3],[0,2],[0,1,3],[0,2]]
输出：false
解释：能不能将节点分割成两个独立的子集，以使每条边都连通一个子集中的一个节点与另一个子集中的一个节点。
题解：就是图的遍历，遍历的时候尝试去填颜色，如果已经访问过，检查是否冲突，如果没有访问，填充与当前节点不同的颜色即可。

*/
public:
    bool dfs(vector<bool>& visited, vector<int>& color, vector<vector<int>>& graph, int cur) {
        visited[cur] = true;
        for(int i=0; i < graph[cur].size(); i++) {
            int child = graph[cur][i];
            if(!visited[child]) {
                color[child] = ~color[cur];
                if(!dfs(visited, color, graph, child)) {
                    return false;
                }
            } else {
                if(color[child] == color[cur]) {
                    return false;
                }
            }
        }
        return true;
    }

    bool isBipartite(vector<vector<int>>& graph) {
        vector<bool> visited (100, false);
        vector<int> color(100, 0);
        for(int i=0; i<graph.size(); i++) {
            if(!visited[i]) {
                if(!dfs(visited, color, graph, i)) {
                    return false;
                }
            }
        }
        return true;
    }
};

```


#### 886. 可能的二分法

```c++
class Solution {
/*
886. 可能的二分法
给定一组 n 人（编号为 1, 2, ..., n）， 我们想把每个人分进任意大小的两组。每个人都可能不喜欢其他人，那么他们不应该属于同一组。
给定整数 n 和数组 dislikes ，其中 dislikes[i] = [ai, bi] ，表示不允许将编号为 ai 和  bi的人归入同一组。当可以用这种方法将所有人分进两组时，返回 true；否则返回 false。

输入：n = 4, dislikes = [[1,2],[1,3],[2,4]]
输出：true
解释：group1 [1,4], group2 [2,3]
解释：能不能将节点分割成两个独立的子集，使每组里面没有相互不喜欢的数组
题解：就是图的遍历，遍历的时候尝试去填颜色，如果已经访问过，检查是否冲突，如果没有访问，填充与当前节点不同的颜色即可。

*/
public:
    bool dfs(vector<bool>& visited, vector<int>& color, vector<vector<int>>& graph, int cur) {
        visited[cur] = true;
        for(int i=0; i < graph[cur].size(); i++) {
            int child = graph[cur][i];
            if(!visited[child]) {
                color[child] = ~color[cur];
                if(!dfs(visited, color, graph, child)) {
                    return false;
                }
            } else {
                if(color[child] == color[cur]) {
                    return false;
                }
            }
        }
        return true;
    }

    bool possibleBipartition(int n, vector<vector<int>>& dislikes) {
        vector<vector<int>> graph(n);
        for(int i=0; i<dislikes.size(); i++) {
            graph[dislikes[i][0]-1].push_back(dislikes[i][1]-1);
            graph[dislikes[i][1]-1].push_back(dislikes[i][0]-1);
        }

        vector<bool> visited (n, false);
        vector<int> color(n, 0);
        for(int i=0; i < n; i++) {
            if(!visited[i]) {
                if(!dfs(visited, color, graph, i)) {
                    return false;
                }
            }
        }
        return true;
    }
};

```

### 3.3 环检测/拓扑排序

#### 207. 课程表 python解法

```python
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """

        graph = defaultdict(set)
        indegree = [0] * numCourses
        for prerequisite in prerequisites:
            graph[prerequisite[1]].add(prerequisite[0])
            indegree[prerequisite[0]] += 1
        

        cache = deque()
        for i in range(numCourses):
            if indegree[i] == 0:
                cache.append(i)

        cnt = 0
        while cache:
            # print('cache:', cache)
            cur = cache.popleft()
            cnt += 1
            for child in graph[cur]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    cache.append(child)

        return cnt == numCourses

```

#### 207. 课程表 c++

```c++
class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {

        vector<int> indgree(numCourses, 0);
        vector<vector<int>> graph(numCourses);
        for(int i=0; i<prerequisites.size(); i++) {
            indgree[prerequisites[i][0]] += 1;
            graph[prerequisites[i][1]].push_back(prerequisites[i][0]);
        }
        deque<int> d;
        for(int i=0; i<numCourses; i++) {
            if(indgree[i] == 0) {
                d.push_back(i);
            }
        }
        int finishNum = 0;
        while(!d.empty()) {
            int cur = d.front();
            finishNum += 1;
            d.pop_front();
            for(int i=0; i<graph[cur].size(); i++) {
                int child = graph[cur][i];
                indgree[child] -= 1;
                if(indgree[child] == 0) {
                    d.push_back(child);
                }
            }
        }
        return finishNum == numCourses;
    }
};
```

#### 210. 课程表II 输出顺序

```c++
class Solution {
public:
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        vector<int> indgree(numCourses, 0);
        vector<vector<int>> graph(numCourses);
        for(int i=0; i<prerequisites.size(); i++) {
            indgree[prerequisites[i][0]] += 1;
            graph[prerequisites[i][1]].push_back(prerequisites[i][0]);
        }
        deque<int> d;
        for(int i=0; i<numCourses; i++) {
            if(indgree[i] == 0) {
                d.push_back(i);
            }
        }
        int finishNum = 0;
        vector<int> res;
        while(!d.empty()) {
            int cur = d.front();
            res.push_back(cur);
            finishNum += 1;
            d.pop_front();
            for(int i=0; i<graph[cur].size(); i++) {
                int child = graph[cur][i];
                indgree[child] -= 1;
                if(indgree[child] == 0) {
                    d.push_back(child);
                }
            }
        }
        if(finishNum == numCourses) {
            return res;
        } else {
            res.clear();
            return res;
        }
    }
};
```
### 3.4 并查集

#### 765. 情侣牵手

```c++
class UnionFind {

    private:
        vector<int> parent;
        int n;
    public:
        UnionFind(int size) {
            n = size;
            for(int i=0; i<size; i++) {
                parent.push_back(i);
            }
        }
        int getRoot(int x) {
            while(x != parent[x]) {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        }
        void add(int x, int y) {
            int root_x = getRoot(x);
            int root_y = getRoot(y);
            if(root_x != root_y) {
                parent[root_x] = root_y;
                n -= 1;
            }
        }
        int getCnt() {
            return n;
        }
};


class Solution {
/**
765. 情侣牵手
输入: row = [0,2,1,3]
输出: 1
解释: 只需要交换row[1]和row[2]的位置即可。
题目：n对情侣，位置打散，问让所有情侣可以牵手，问最小交换次数
题解：本来有n组结果，现在可能少于n组，对于0，1，分到0组，2，3分到1组，4，5分到2组，原始有n组，当前的并查集的n减少一个，通过交换一次就可达到。
**/
public:
    int minSwapsCouples(vector<int>& row) {
        int n = row.size();
        UnionFind unionFind(n);
        // 注意这里是每两个人合并一下
        for(int i=0; i< n; i+=2) {
            unionFind.add(row[i] / 2, row[i+1] / 2);
        }
        return n - unionFind.getCnt();
    }
};

```

### 3.5 最小生成树
### 3.6 最短路径

#### 743. 网络延迟时间

```python

class Solution(object):
    def networkDelayTime(self, times, n, k):
        """
        :type times: List[List[int]]
        :type n: int
        :type k: int
        :rtype: int
        """

        cost = [float('inf')] * (n+1)
        cost[k] = 0
        cache = []
        heapq.heappush(cache, [0, k])

        graph = [{} for _ in range(n+1)]
        for x,y,c in times:
            graph[x][y] = c

        while cache:
            cur_cost, cur = heapq.heappop(cache)
            if cost[cur] <= cur_cost:
                for nei in graph[cur]:
                    nei_cost = cur_cost + graph[cur][nei]
                    if nei_cost < cost[nei]:
                        cost[nei] = nei_cost
                        heapq.heappush(cache, [nei_cost, nei])
        max_val = -1
        for i in range(1, n+1):
            if cost[i] == float('inf'):
                return -1
            max_val = max(cost[i], max_val)
        return max_val
```


