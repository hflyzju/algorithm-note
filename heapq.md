

#### 295. 数据流的中位数

```python

class MedianFinder:

    def __init__(self):
        self.max_heapq = []
        self.min_heapq = []


    def addNum(self, num: int) -> None:
        if len(self.max_heapq) > len(self.min_heapq):
            # 需要往最小堆里面添加，先加到最大堆(其实也是以最小堆实现)，然后把最大堆里面的最小值拿出来
            heapq.heappush(self.max_heapq, num)
            min_val = heapq.heappop(self.max_heapq)
            heapq.heappush(self.min_heapq, -min_val)
        else:
            # 需要往最大堆里面提那家，先加到最小堆，然后把最小堆里面最大的元素拿出来
            heapq.heappush(self.min_heapq, -num)
            max_val = -heapq.heappop(self.min_heapq)
            heapq.heappush(self.max_heapq, max_val)
        # print("add:", num)
        # print("max_heapq:", self.max_heapq)
        # print("min_heapq:", self.min_heapq)
    def findMedian(self) -> float:
        """数据流的中位数
addNum(1)
addNum(2)
findMedian() -> 1.5
addNum(3) 
findMedian() -> 2

        题解：
            1. 用最大堆和最小堆维护大的一半和小的一半，例如:max_heapq=[3,4,5], min_heapq=[-2, -2, -1] 
            2. 如何保证大的在大堆，小的在小堆？往最大堆添加数字的时候，要先加入到最小堆，然后从最小堆拿出最大的元素放到最大堆，加入最小堆要反过来
        """
        if len(self.max_heapq) > len(self.min_heapq):
            return self.max_heapq[0]
        elif len(self.max_heapq) < len(self.min_heapq):
            return self.min_heapq[-1]
        else:
            return (self.max_heapq[0] - self.min_heapq[0]) / 2






# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
```

#### 407. 接雨水 II

```python
class Solution:
    def trapRainWater(self, heightMap: List[List[int]]) -> int:
        """二维接雨水

输入: heightMap = [[1,4,3,1,3,2],[3,2,1,3,2,4],[2,3,3,2,3,1]]
输出: 4
解释: 下雨后，雨水将会被上图蓝色的方块中。总的接雨水量为1+2+1=4。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/trapping-rain-water-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

        题解：
        1. 利用优先队列，从最外圈往里面搜索，找到外圈最小的桶，然后计算相邻内部的最大盛水体积。
        2. 注意更新相邻内部节点的高度为max(点前外部节点高度，相邻内部节点高度)
        """

        m, n = len(heightMap), len(heightMap[0])

        cache = []
        visited = set()
        # 1. 最外面一圈入优先队列
        for i in range(m):
            heapq.heappush(cache, [heightMap[i][0], i, 0])
            heapq.heappush(cache, [heightMap[i][n-1], i, n-1])
            visited.add((i, 0))
            visited.add((i,n-1))
        for j in range(n):
            heapq.heappush(cache, [heightMap[0][j], 0, j])
            heapq.heappush(cache, [heightMap[m-1][j], m-1, j])
            visited.add((0,j))
            visited.add((m-1,j))

        s = 0
        while cache:
            cur_h, cur_x, cur_y = heapq.heappop(cache)
            for dx, dy in [[0, -1], [0, 1], [-1, 0], [1, 0]]:
                nx, ny = cur_x + dx, cur_y + dy
                if 0 <= nx < m and 0 <= ny < n and (nx, ny) not in visited:
                    tmp = cur_h - heightMap[nx][ny]
                    if tmp > 0:
                        s += tmp
                    visited.add((nx, ny))
                    # 高度取最高的那个
                    heapq.heappush(cache, [max(heightMap[nx][ny], cur_h), nx, ny])
        return s



```


#### 451. 根据字符出现频率排序


```python
class Solution:
    def frequencySort(self, s: str) -> str:


        word_cnt = defaultdict(int)
        for si in s:
            word_cnt[si] += 1

        cache = []
        for letter, cnt in word_cnt.items():
            heapq.heappush(cache, [-cnt, letter])

        res = []
        while cache:
            cnt, letter = heapq.heappop(cache)
            cnt = abs(cnt)
            while cnt:
                res.append(letter)
                cnt -= 1

        return ''.join(res)

```