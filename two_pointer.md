


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