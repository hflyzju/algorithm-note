

#### 136. 只出现一次的数字

```python

class Solution:
    def singleNumber(self, nums: List[int]) -> int:

        if not nums:
            return -1

        if len(nums) == 1:
            return nums[0]

        """
        2:10 
        """
        v = 0
        for num in nums:
            v ^= num

        return v

```