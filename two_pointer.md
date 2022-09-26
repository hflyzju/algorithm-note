|类型|  题号 | 难度  | 题目 | 题解 | 
| ---- |  ----  | ----  | --- | --- |
|位运算+双指针| 2411 | 中等| 求以每个位置为开始的最大连续或(Bitwise OR)的最小长度|或(Bitwise OR)，全部加起来肯定最大，那么需要统计全部加起来有多少个位满了，考虑从左往右加，如果当前的位和能加满的位一致，则输出最小连续长度|



#### 2411. Smallest Subarrays With Maximum Bitwise OR

```
You are given a 0-indexed array nums of length n, consisting of non-negative integers. For each index i from 0 to n - 1, you must determine the size of the minimum sized non-empty subarray of nums starting at i (inclusive) that has the maximum possible bitwise OR.

In other words, let Bij be the bitwise OR of the subarray nums[i...j]. You need to find the smallest subarray starting at i, such that bitwise OR of this subarray is equal to max(Bik) where i <= k <= n - 1.
The bitwise OR of an array is the bitwise OR of all the numbers in it.

Return an integer array answer of size n where answer[i] is the length of the minimum sized subarray starting at i with maximum bitwise OR.

A subarray is a contiguous non-empty sequence of elements within an array.

 

Example 1:

Input: nums = [1,0,2,1,3]
Output: [3,3,2,2,1]
Explanation:
The maximum possible bitwise OR starting at any index is 3. 
- Starting at index 0, the shortest subarray that yields it is [1,0,2].
- Starting at index 1, the shortest subarray that yields the maximum bitwise OR is [0,2,1].
- Starting at index 2, the shortest subarray that yields the maximum bitwise OR is [2,1].
- Starting at index 3, the shortest subarray that yields the maximum bitwise OR is [1,3].
- Starting at index 4, the shortest subarray that yields the maximum bitwise OR is [3].
Therefore, we return [3,3,2,2,1]. 
Example 2:

Input: nums = [1,2]
Output: [2,1]
Explanation:
Starting at index 0, the shortest subarray that yields the maximum bitwise OR is of length 2.
Starting at index 1, the shortest subarray that yields the maximum bitwise OR is of length 1.
Therefore, we return [2,1].
```

```python
from collections import defaultdict

class Solution(object):
    
    def zero_cnt(self, cnt_dict):
        """最大的or sum为0，即所有位都为空"""
        s = sum([_ for k, _ in cnt_dict.items()])
        return s == 0
        
    
    def match(self, left_cnt, right_cnt):
        for k in range(32):
            if left_cnt[k] <= 0 and right_cnt[k] > 0:
                return False
        return True
    
    def smallestSubarrays(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
    
        right_cnt = defaultdict(int)
        for num in nums:
            for k in range(32):
                if (num >> k) & 1:
                    right_cnt[k] += 1
        
        n = len(nums)
        left_cnt = defaultdict(int)
        r = 0
        res = []
        for l in range(n):
            if not self.match(left_cnt, right_cnt):
                while r < n:
                    num = nums[r]
                    for k in range(32):
                        if (num >> k) & 1:
                            left_cnt[k] += 1
                    if self.match(left_cnt, right_cnt):
                        r += 1
                        break
                    else:
                        r += 1
            if self.zero_cnt(right_cnt):
                res.append(1)
            else:
                res.append(r - l)
            num = nums[l]
            for k in range(32):
                if (num >> k) & 1:
                    right_cnt[k] -= 1
                    left_cnt[k] -= 1
                    
        return res

```

#### 532. 数组中的 k-diff 数对

```python
class Solution(object):
    def findPairs(self, nums, k):
        """532. 数组中的 k-diff 数对
        :type nums: List[int]
        :type k: int
        :rtype: int
给你一个整数数组 nums 和一个整数 k，请你在数组中找出 不同的 k-diff 数对，并返回不同的 k-diff 数对 的数目。

输入：nums = [3, 1, 4, 1, 5], k = 2
输出：2
解释：数组中有两个 2-diff 数对, (1, 3) 和 (3, 5)。
尽管数组中有两个 1 ，但我们只应返回不同的数对的数量。

        """

        nums.sort()
        n, y, res = len(nums), 1, 0
        for x in range(n):
            if x == 0 or nums[x] != nums[x - 1]:
                while y < n and (nums[y] - nums[x] < k or y <= x):
                    y += 1
                if y < n and nums[y] - nums[x] == k:
                    res += 1
        return res

# 作者：LeetCode-Solution
# 链接：https://leetcode.cn/problems/k-diff-pairs-in-an-array/solution/shu-zu-zhong-de-k-diff-shu-dui-by-leetco-ane6/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```