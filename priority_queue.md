

#### 1705. 吃苹果的最大数目

```python

class Solution(object):
    def eatenApples(self, apples, days):
        """
        :type apples: List[int]
        :type days: List[int]
        :rtype: int

        s = 2
        [1:1, ]

        cache: [[time. num]] 第time天不能吃
        time = 1, cache:[[1, 3]] -> [] s=1
        time = 2, cache:[[4, 2]] -> [[4, 1]] s = 2
        time = 3, cache:[[4, 5]] -> [[4, 4]] s = 3
        time = 4, cache [[8 ,5]] -> [[8, 4]] s = 4
        time = 5, cache [[7, 2], [8, 4]] -> [[7, 1], [8, 4]] s = 5
        time = 6, [[7, 1], [8, 4]] -> [[8, 4]] s = 6
        time = 7, [[8, 4]] -> [[8, 3]], s = 7

        """

        n = len(apples)
        cache = []
        time = 1
        ans = 0
        while cache or time <= n:
            # 1. 先放入当天的苹果
            if time <= n:
                apple, day = apples[time-1], days[time-1]
                if apple > 0:
                    # time=1, day=1, time=2不能吃
                    heapq.heappush(cache, [time + day, apple])
            # 2.把不符合要求的pop出来
            while cache and cache[0][0] <= time:
                heapq.heappop(cache)
            # 3.查看当天的苹果数量
            # print("before: time=",time, 'cache:', cache)
            if cache:
                cur_time, cur_apple = heapq.heappop(cache)
                ans += 1
                if cur_apple - 1 > 0:
                    heapq.heappush(cache, [cur_time, cur_apple - 1])
            # print("end: time=",time, 'cache:', cache)
            time += 1
        
        return ans

```