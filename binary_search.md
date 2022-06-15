### 一、总结

|  类型   | 编号  | 题目 | 题解 |
|  ----  | ----  | --- | --- |
| 1d数组中找数字  | leetcode33, leetcode81 | 旋转数组搜索 | 找到有序的那一半，与当前数字进行比较，然后搜索 |
| 2d数组中找第k小  | leetcode 668 | 乘法表中第k小的数 | 首先确定边界[1, m*n]，每次找到mid，对于每个1d的行，可以常数时间得到该行小于等于mid的个数，这样可以逐渐缩小边界，输出r即可。|
| 1d数组找出第 K 小的数对距离 | leetcode719 |找出第 K 小的数对距离 | 数对距离边界[0, 10e9], 对于每一个mid, 可以O(n)时间找到nums中小于mid的间隔的个数(nums已经排序，可以用双指针在O(n)时间拿到结果)，这样可以逐渐缩小边界，输出r即可。|



### 二、题解

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

        l, r = 1, m * n
        if m < n:
            m, n = n, m
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


#### Shifted Array Search

input:  shiftArr = [9, 12, 17, 2, 4, 5], num = 2 # shiftArr is the
                                                 # outcome of shifting
                                                 # [2, 4, 5, 9, 12, 17]
                                                 # three times to the left

output: 3 # since it’s the index of 2 in arr



```python
def shifted_arr_search(shiftArr, num):
  
  l, r = 0, len(shiftArr)-1
  while l <= r:
    # [1,2]
    # mid=1
    mid_index = (l+r) >> 1
    mid_val = shiftArr[mid_index]
    if mid_val == num:
      return mid_index
    # means left part are sorted
    if shiftArr[l] <= mid_val:
      if shiftArr[l] <= num < mid_val:
        r = mid_index - 1
      else:
        l = mid_index + 1
    else:
      #     >  mid
      # [9, 2, 1, ]
      # right part are sorted
      if mid_val < num < shiftArr[r]:
        l = mid_index + 1
      else:
        r = mid_index - 1
  return -1
      

```


#### 719. 找出第 K 小的数对距离

```python
class Solution:
    def smallestDistancePair(self, nums: List[int], k: int) -> int:
        """
数对 (a,b) 由整数 a 和 b 组成，其数对距离定义为 a 和 b 的绝对差值。
给你一个整数数组 nums 和一个整数 k ，数对由 nums[i] 和 nums[j] 组成且满足 0 <= i < j < nums.length 。返回 所有数对距离中 第 k 小的数对距离。
输入：nums = [1,3,1], k = 1
输出：0
解释：数对和对应的距离如下：
(1,3) -> 2
(1,1) -> 0
(3,1) -> 2
距离第 1 小的数对是 (1,1) ，距离为 0 。

题解：
1. 暴力，至少是n**2 * log(n**2)级别，会超时。
2. 二分+双指针：已知距离的最大diff范围为[l=0, r=10e6]，直接用二分法猜测第k的距离是mid，如果整个nums数组中，间隔小于等于mid的个数大于等于k，那么可以将r置为mid，这个时候mid可能就是最终输出，因为可能有多个间隔为mid的值，如果小于等于mid的个数小于k，那么可以将l置为mid+1，因为，需要找第k大。最终输出r就行
参考：https://leetcode.cn/problems/find-k-th-smallest-pair-distance/solution/by-ac_oier-o4if/
        """
        def check(nums, mid):
            """use two pointer to find the number of sep that slower than mid for a sorted nums"""
            l, r = 0 ,1
            cnt = 0 # 小于mid的间隔的个数，因为是要找第k小的间隔
            n = len(nums)
            while l < n: # 对于每个起点，计算他[l,r]之间的小于mid的间隔的个数
                while r < n and nums[r] - nums[l] <= mid:
                    r += 1
                cnt += (r - l - 1)
                l += 1
            return cnt

        nums.sort()
        l, r = 0, 10**6
        
        while l < r:
            # print('l:', l, 'r:', r)
            mid = (l + r) >> 1
            if check(nums, mid) >= k:
                r = mid
            else:
                l = mid + 1
        return r


```