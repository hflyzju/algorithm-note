
#### 560 子数组和为k的个数

```python
class Solution(object):
    def subarraySum(self, nums, k):
        """子数组和为k的个数
        :type nums: List[int]
        :type k: int
        :rtype: int
        Example:
            #  Input: nums = [1,1,1], k = 2
            # Output: 2
        Solution:
            前缀和+哈希
        """
        from collections import defaultdict
        # [1,1,1]
        # 1
        # cur_sum=3
        # pre_sum_cnt=[1:1, 2:1]
        # diff = 3-2=1
        # pre_sum = 1
        # cur_sum - pre_sum = k
        # pre_sum = cur_sum - k
        pre_sum_cnt = defaultdict(int)
        pre_sum_cnt[0] = 1
        pre_sum = 0
        cnt = 0
        for num in nums:
            cur_sum = pre_sum + num
            diff = cur_sum - k
            cnt += pre_sum_cnt[diff]
            pre_sum_cnt[cur_sum] += 1
            pre_sum = cur_sum
        return cnt

```


#### 791 将s中的字母按order的顺序进行排序

```python
class Solution(object):
    def customSortString(self, order, s):
        """将s中的字母按order的顺序进行排序
        面经地址：https://www.1point3acres.com/bbs/thread-650769-1-1.html
        :type order: str
        :type s: str
        :rtype: str
        Example:
            # Input: order = "cba", s = "abcd"
            # Output: "cbad"

            # Input: order = "cba", s = "abad"
            # Output: "baad" (要注意s中字符重复的问题)

        Solution:
            hash表
        """
        from collections import defaultdict
        letter_cnt = defaultdict(int)
        for letter in s:
            letter_cnt[letter] += 1
        order_letter_set = set()
        for letter in order:
            order_letter_set.add(letter)

        res = []
        for letter in order:
            if letter in letter_cnt:
                while letter_cnt[letter] > 0:
                    res.append(letter)
                    letter_cnt[letter] -= 1

        for letter in s:
            if letter not in order_letter_set:
                res.append(letter)

        return ''.join(res)

```