


#### 668. 乘法表中第k小的数-二分

```python

class Solution(object):
    def findKthNumber(self, m, n, k):
        """
        :type m: int
        :type n: int
        :type k: int
        :rtype: int

668. 乘法表中第k小的数

输入: m = 3, n = 3, k = 5
输出: 3
解释: 
乘法表:
1	2	3
2	4	6
3	6	9

第5小的数字是 3 (1, 2, 2, 3, 3).


        题解：二分法，最开始的区间为（1，m*n)，不断缩小区间，找到第一个L，满足小于等于L的个数为k个。

        """

        l, r = 1, m * n
        if m < n:
            m, n = n, m
        def get_small_cnt(mid):
            cnt = 0
            for i in range(1, n+1):
                if i * m < mid:
                    cnt += m
                else:
                    # i*1, i*2, .., i*m
                    # i=3, -> 3, 6, 9, ...
                    cnt += mid // i
            return cnt
        while l < r:
            mid = (l+r) >> 1
            cnt = get_small_cnt(mid)
            if cnt >= k:
                r = mid
            else:
                # 二分找的是第一个满足左边数的数量大于等于 k 的数，必然是存在的。
                l = mid + 1
        # 二分找的是第一个满足左边数的数量大于等于 kk 的数，必然是存在的。
        return l
```