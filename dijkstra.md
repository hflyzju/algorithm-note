

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
        heapq.heappush(cache, [0, k])
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
        heapq.heappush(cache, [-1.0, start])
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