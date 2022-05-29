

#### 743 网络上从k节点到达其他所有节点的最短时间，无法到达返回-1


```python
class Solution:
    def networkDelayTime(self, times, n, k):
        """网络上从k节点到达其他所有节点的最短时间，无法到达返回-1
        Example:
            # Input: times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
            # Output: 2
        解法：
            Dijkstra
        """

        import heapq
        # 1. 建图 - 邻接表
        mp = [{} for i in range(n + 1)]
        for u, v, t in times:
            mp[u][v] = t
        # 2. Dijkstra
        # 记录结点最早收到信号的时间, 设置最短路标记
        min_dis = [float('inf')] * (n + 1)
        min_dis[k] = 0
        cache = []
        heapq.heappush(cache, [0, k]) # dis, cur_pos
        while cache:
            # 查找最短路径
            cur_dis, cur = heapq.heappop(cache)
            # Nodes can get added to the priority queue multiple times. We only
            # process a vertex the first time we remove it from the priority queue.
            if cur_dis > min_dis[cur]:
                continue
            for neighboor, weight in mp[cur].items():
                if cur_dis + weight < min_dis[neighboor]:
                    min_dis[neighboor] =  cur_dis + weight
                    heapq.heappush(cache, [cur_dis + weight, neighboor])

        # print('min_dis:',min_dis)
        # 3 返回最大值
        min_val = -1
        for i in range(1, n + 1):
            if min_dis[i] == float('inf'):
                return -1
            min_val = max(min_val, min_dis[i])
        return min_val
```


#### 1514. 求start->end的最大概率，边的概率是相乘

```python
class Solution(object):
    def maxProbability(self, n, edges, succProb, start, end):
        """求start->end的最大概率，边的概率是相乘
        :type n: int
        :type edges: List[List[int]]
        :type succProb: List[float]
        :type start: int
        :type end: int
        :rtype: float
        Example:
            # Input: n = 3, edges = [[0,1],[1,2],[0,2]], succProb = [0.5,0.5,0.2], start = 0
            # , end = 2
            # Output: 0.25000

        方法：
            Dijkstra
        """
        import heapq
        graph = [{} for _ in range(n)]
        for i in range(len(edges)):
            x, y = edges[i]
            graph[x][y] = succProb[i]
            graph[y][x] = succProb[i]

        max_prob = [-1] * n
        max_prob[start] = 1.0
        cache = []
        heapq.heappush(cache, [-1.0, start]) # 最小堆
        while cache:
            cur_prob, cur_node = heapq.heappop(cache)
            # 目的是为了找到一个更大的，前面已经比较大了，直接跳过
            if max_prob[cur_node] > (-cur_prob):
                continue
            for nei, prob in graph[cur_node].items():
                next_prob = (-cur_prob) * prob
                if next_prob > max_prob[nei]:
                    max_prob[nei] = next_prob
                    heapq.heappush(cache, [-next_prob, nei])
        # print('max_prob:', max_prob)
        if max_prob[end] != -1:
            return max_prob[end]
        return 0.0

```

#### 1631 最小的体能消耗


```python
class Solution(object):
    def minimumEffortPath(self, heights):
        """从0,0到达m-1,n-1花费的最小的力气
        :type heights: List[List[int]]
        :rtype: int
        Example:
            Input: heights = [[1,2,2],[3,8,2],[5,3,5]]
            Output: 2
        Solutions:
            dijkstra
            下一个节点更新公式：next_effort = max(cur_effort, abs(heights[x][y] - heights[nx][ny]))
        """
        import heapq
        cache = []
        m, n = len(heights), len(heights[0])
        heapq.heappush(cache, [0, 0, 0])
        min_effort = [float('inf')] * (m * n + 1)
        min_effort[0] = 0
        while cache:
            cur_effort, x, y = heapq.heappop(cache)
            if x == m - 1 and y == n - 1:
                return cur_effort
            if min_effort[x * n + y] < cur_effort:
                continue
            for dx, dy in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n:
                    next_effort = max(cur_effort, abs(heights[x][y] - heights[nx][ny]))
                    if next_effort < min_effort[nx * n + ny]:
                        min_effort[nx * n + ny] = next_effort
                        heapq.heappush(cache, [next_effort, nx, ny])


```




#### 6081. 到达角落需要移除障碍物的最小数目

```python
from heapq import heappop, heappush


def dijkstra(graph, start):
    """ 
        Uses Dijkstra's algortihm to find the shortest path from node start
        to all other nodes in a directed weighted graph.
    """
    n = len(graph)
    dist, parents = [float("inf")] * n, [-1] * n
    dist[start] = 0
    queue = [(0, start)]
    while queue:
        cur_dis, cur_index = heappop(queue)
        if cur_dis == dist[cur_index]:
            for w, edge_len in graph[cur_index]:
                if edge_len + cur_dis < dist[w]:
                    dist[w], parents[w] = edge_len + cur_dis, cur_index
                    heappush(queue, (edge_len + cur_dis, w))
    return dist, parents



class Solution:
    def minimumObstacles(self, grid: List[List[int]]) -> int:
        """
给你一个下标从 0 开始的二维整数数组 grid ，数组大小为 m x n 。每个单元格都是两个值之一：

0 表示一个 空 单元格，
1 表示一个可以移除的 障碍物 。
你可以向上、下、左、右移动，从一个空单元格移动到另一个空单元格。

现在你需要从左上角 (0, 0) 移动到右下角 (m - 1, n - 1) ，返回需要移除的障碍物的 最小 数目。
        
输入：grid = [[0,1,1],[1,1,0],[1,1,0]]
输出：2
解释：可以移除位于 (0, 1) 和 (0, 2) 的障碍物来创建从 (0, 0) 到 (2, 2) 的路径。
可以证明我们至少需要移除两个障碍物，所以返回 2 。
注意，可能存在其他方式来移除 2 个障碍物，创建出可行的路径。
        
        """
        
        m = len(grid)
        n = len(grid[0])
        adj = defaultdict(list)
         
        # 相当于拿到每个点，转移到下一个位置需要的损耗
        # adj[i]:相当于当前位置已经走到了，走到下一个邻居，需要消耗的能量，这里的能量就是下一个位置的grid[i][j]值
        for i in range(m):
            for j in range(n):
                if i > 0:
                    adj[(i-1)*n+j].append([i*n+j, grid[i][j]])
                if j > 0:
                    adj[i*n+j-1].append([i*n+j, grid[i][j]])
                if i < m - 1:
                    adj[(i+1)*n+j].append([i*n+j, grid[i][j]])
                if j < n - 1:
                    adj[i*n+j+1].append([i*n+j, grid[i][j]])
        # for k in range(m*n):
        #     print('='*10)
        #     v = adj[k]
        #     print('k:', k)
        #     print('v:',v)
        """
grid:
[[0,1,1],
 [1,1,0],
 [1,1,0]]
==========
k: 0
v: [[1, 1], [3, 1]]
==========
k: 1
v: [[0, 0], [2, 1], [4, 1]]
==========
k: 2
v: [[1, 1], [5, 0]]
==========
k: 3
v: [[0, 0], [4, 1], [6, 1]]
==========
k: 4
v: [[1, 1], [3, 1], [5, 0], [7, 1]]
==========
k: 5
v: [[2, 1], [4, 1], [8, 0]]
==========
k: 6
v: [[3, 1], [7, 1]]
==========
k: 7
v: [[4, 1], [6, 1], [8, 0]]
==========
k: 8
v: [[5, 0], [7, 1]]

        
        """
        dist, par = dijkstra(adj, 0)
        return dist[m*n-1]
                    

```