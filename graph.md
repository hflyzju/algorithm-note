


# 一、总结
|  类型 | 难度  | 题目 | 题解 | 
|  ----  | ----  | --- | --- |
|图的遍历|中等|797.所有可能的路径|进入节点加入path，遍历完子节点后从path中pop出来|
|二分图判定|中等|785.判断二分图, 886.可能的二分法|遍历图的时候进行着色，没有访问过的，着不同的颜色，访问过的，检查着色是否和当前节点冲突|

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
### 3.4 并查集
### 3.5 最小生成树
### 3.6 最短路径


