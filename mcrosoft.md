#### 15 三数之和

```python
class Solution(object):
    def threeSum(self, nums):
        """找出所有三数之和为0的数字, 注意要去重
        # https://www.1point3acres.com/bbs/thread-728439-1-1.html
        :type nums: List[int]
        :rtype: List[List[int]]
        Example:
            #  Example 1:
            #  Input: nums = [-1,0,1,2,-1,-4]
            #  Output: [[-1,-1,2],[-1,0,1]]
        Solution:
            O(n**2)
        """

        n = len(nums)
        if n < 3:
            return []
        nums.sort() # [0,0,0,0,0]
        result = []
        for i in range(n):
            if i >= 1 and nums[i] == nums[i-1]:
                continue
            r = n - 1
            target = -nums[i]
            l = i + 1
            while l < r:
                # 跳过相同数字
                if l > i + 1 and nums[l] == nums[l - 1]:
                    l += 1
                else:
                    # 缩小范围
                    while l < r and nums[l] + nums[r] > target:
                        r -= 1
                    if l == r:
                        break
                    if nums[l] + nums[r] == target:
                        result.append([nums[i], nums[l], nums[r]])
                    l += 1
        return result

```

#### 42 接雨水

```python
class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        n = len(height)
        if n <= 2:
            return 0
        left_max_height = [0] * n
        right_max_height = [0] * n
        left_max = height[0]
        for i in range(1, n):
            left_max_height[i] = left_max
            left_max = max(height[i], left_max)
        right_max = height[n-1]
        for i in range(n-2, -1, -1):
            right_max_height[i] = right_max
            right_max = max(height[i], right_max)

        s = 0
        for i in range(n):
            s += max(min(left_max_height[i], right_max_height[i]) - height[i], 0)
        return s

```