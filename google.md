https://jobs.1point3acres.com/companies/google/interview

https://www.1point3acres.com/bbs/thread-901904-1-1.html

1. 建议练习：蠡口 伊芭芭尔  普通难度但是挺容易绕进去。这轮逻辑不清晰，但是应该做对了。感觉除了priority queue也没啥特别好的解法。
2. 建议练习： 蠡口 尔尔凄凄 或者 儿灵久留。这轮秒了。碰到了很友善的同胞。在此给你磕个头。
3. 建议练习： 蠡口 巴陵散。这题没找到原题，但是思路和这个习题很像。写了查并集的解法并且优化了。感觉这轮代码逻辑清晰，讲的很详细，一直引领着面试官来follow我。

系统设计建议练习一下alex xu 第二册4-5章。感觉面试官也不是很懂，前两年那个面试官也问了类似的题，上次被问傻了。。 今年的面试官估计也是看的这个书 哈哈。这轮一般，因为我前面讲的偏多导致后面follow up不够多吧。但是感觉meet the bar应该够了。和面试官交‍‍‌‌‌‍‌‌‍‍‍‌‍‌‍流的不错。




### Google | OA 2020 | Min Amplitude & Ways to Split String
https://leetcode.com/discuss/interview-question/352460/Google-Online-Assessment-Questions

#### 1509. Minimum Difference Between Largest and Smallest Value in Three Moves
```
You are given an integer array nums. In one move, you can choose one element of nums and change it by any value.
Return the minimum difference between the largest and smallest value of nums after performing at most three moves.
Example 1:
Input: nums = [5,3,2,4]
Output: 0
Explanation: Change the array [5,3,2,4] to [2,2,2,2].
The difference between the maximum and minimum is 2-2 = 0.
Example 2:
Input: nums = [1,5,0,10,14]
Output: 1
Explanation: Change the array [1,5,0,10,14] to [1,1,0,1,1]. 
The difference between the maximum and minimum is 1-0 = 1.
```

```python

class Solution(object):
    def minDifference(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        k = 3 + 1
        n = len(nums)
        if n <= k:
            return 0
        min_heap = []
        max_heap = []
        for i, num in enumerate(nums):
            if i < k or num > min_heap[0]:
                heapq.heappush(min_heap, num)
            if i < k or num < -max_heap[0]:
                heapq.heappush(max_heap, -num)
            if len(min_heap) > k:
                heapq.heappop(min_heap)
            if len(max_heap) > k:
                heapq.heappop(max_heap)
        
        max_nums = []
        min_nums = []
        for i in range(k):
            # [5,4,3,2]
            max_nums.append(heapq.heappop(min_heap))
            # [1,2,3,4]
            min_nums.append(-heapq.heappop(max_heap))

        min_diff = float('inf')
        for i in range(k):
            min_diff = min(min_diff, max_nums[i] - min_nums[k-i-1])
        return min_diff
```


#### 1525. Number of Good Ways to Split a String

```
You are given a string s.

A split is called good if you can split s into two non-empty strings sleft and sright where their concatenation is equal to s (i.e., sleft + sright = s) and the number of distinct letters in sleft and sright is the same.

Return the number of good splits you can make in s.

 

Example 1:

Input: s = "aacaba"
Output: 2
Explanation: There are 5 ways to split "aacaba" and 2 of them are good. 
("a", "acaba") Left string and right string contains 1 and 3 different letters respectively.
("aa", "caba") Left string and right string contains 1 and 3 different letters respectively.
("aac", "aba") Left string and right string contains 2 and 2 different letters respectively (good split).
("aaca", "ba") Left string and right string contains 2 and 2 different letters respectively (good split).
("aacab", "a") Left string and right string contains 3 and 1 different letters respectively.
Example 2:

Input: s = "abcd"
Output: 1
Explanation: Split the string as follows ("ab", "cd").
```

```python
class Solution(object):
    def numSplits(self, s):
        """
        :type s: str
        :rtype: int
        """
        n = len(s)
        lc = [0] * n
        rc = [0] * n
        
        ls = set()
        rs = set()
        for i in range(n):
            ls.add(s[i])
            lc[i] = len(ls)
            rs.add(s[n-i-1])
            rc[n-i-1] = len(rs)
        
        cnt = 0
        for i in range(n-1):
            if lc[i] == rc[i+1]:
                cnt += 1
        return cnt

```