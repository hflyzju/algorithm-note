

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



#### 2386. Find the K-Sum of an Array

```

User Accepted:164
User Tried:1156
Total Accepted:185
Total Submissions:3038
Difficulty:Hard
You are given an integer array nums and a positive integer k. You can choose any subsequence of the array and sum all of its elements together.

We define the K-Sum of the array as the kth largest subsequence sum that can be obtained (not necessarily distinct).

Return the K-Sum of the array.

A subsequence is an array that can be derived from another array by deleting some or no elements without changing the order of the remaining elements.

Note that the empty subsequence is considered to have a sum of 0.

Example 1:

Input: nums = [2,4,-2], k = 5
Output: 2
Explanation: All the possible subsequence sums that we can obtain are the following sorted in decreasing order:
- 6, 4, 4, 2, 2, 0, 0, -2.
The 5-Sum of the array is 2.
Example 2:

Input: nums = [1,-2,3,4,-10,12], k = 16
Output: 10
Explanation: The 16-Sum of the array is 10.
```

```python
class Solution(object):
    def kSum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        
        max_s = sum([max(_, 0) for _ in nums])
        
        nums = [abs(_) for _ in nums]
        nums.sort()
        
        heap = []
        heapq.heappush(heap, [nums[0], 0])
        cur = 0
        for i in range(k - 1):
            cur, cur_index = heapq.heappop(heap)
            if cur_index < len(nums) - 1:
                # 下一个最小值的候选
                # 1. 取消上一个，用当前值替换
                heapq.heappush(heap, [cur - nums[cur_index] + nums[cur_index + 1], cur_index+1])
                # 2. 直接加上当前值，放到备选池里面
                heapq.heappush(heap, [cur + nums[cur_index + 1], cur_index+1])
                # 3. 对于每一个数字，我都做到了要或者不要
        # print('max_s:', max_s)
        return max_s - cur

```