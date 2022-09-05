### 一、总结

|  类型   | 编号  | 题目 | 题解 |
|  ----  | ----  | --- | --- |
|二分法变种|287. Find the Duplicate Number|找到1-n中可能重复多次的数字|在[1,n]中搜索，每次搜索统计小于i的个数，然后根据这个个数，决定最终的答案在哪个方向，最终答案必须存在，返回l或者r都可以|
| 1d数组中找数字  | leetcode33, leetcode81 | 旋转数组搜索 | 找到有序的那一半，与当前数字进行比较，然后搜索 |
| 2d数组中找第k小  | leetcode 668 | 乘法表中第k小的数(隐藏2分) | 首先确定边界[1, m*n]，每次找到mid，对于每个1d的行，可以常数时间得到该行小于等于mid的个数，这样可以逐渐缩小边界，输出r即可。|
| 1d数组找出第 K 小的数对距离 | leetcode719 |找出第 K 小的数对距离(隐藏2分) | 数对距离边界[0, 10e9], 对于每一个mid, 可以O(n)时间找到nums中小于mid的间隔的个数(nums已经排序，可以用双指针在O(n)时间拿到结果)，这样可以逐渐缩小边界，输出r即可。|
|找左右边界|leetcode34| 从数组中找到第一个出现或者最后一个出现的位置|找左边界的话，当nums[mid]==target, 缩短右边界继续找, 让r=mid-1, 同时记录res=mid，找右边界也一样，反过来就行。 |
|找插入得的位置|leetcode35| 从数组中找到插入排序需要插入的最小的位置, 不同点是target可能不存在|还是可以用之前的模板，不断缩小位置，记录可能的候选位置，当nums[mid]==target, 以及nums[mid] > target， 都是可能的插入位置，另外如果target>nums[-1]这里没有合适的插入位置，需要提前处理。|
|最长递增子序列(1d+2d)|leetcode354| 满足w和h都大，才能套进去，问最多能套多少个|w从小到大排序，w相等，h不能套，所以此时h按降序排列，然后求h的最长递增子序列即可, 最长递增子序列最关键的是找到插入的位置，和上面一致|

### 二、模板
#### 2.1 模板1:找数字(simple)

```python
l, r = 0, n - 1
while l <= r:
  mid = l + r >> 1
  if nums[mid] == target:
    return mid
  elif nums[mid] > target:
    r = mid - 1
  else:
    l = mid + 1
```

#### 2.2 模板2:找一定条件(hard)，用res记录可能的候选，l和r不断逼近最终位置

```python
l, r, res = 0, 10**6, -1
while l <= r:
    mid = (l + r) >> 1
    val = check(nums, mid)
    if val == k:
        res = mid
        r = mid - 1
    elif val > k:
        res = mid
        r = mid - 1 
    else:
        l = mid + 1
return res
```

#### 2.3 模板3:找左右边界(mid)

```python
    def find_left_bound(self, nums, target):
        """找左边界"""
        l, r, res = 0, len(nums) - 1, -1
        while l <= r:
            mid = l + r >> 1
            if nums[mid] == target:
                res = mid
                r = mid - 1
            elif nums[mid] > target:
                r = mid - 1
            else:
                l = mid + 1
        return res

    def find_right_bound(self, nums, target):
        """找右边界"""
        l, r, res = 0, len(nums) - 1, -1
        while l <= r:
            mid = l + r >> 1
            if nums[mid] == target:
                res = mid
                l = mid + 1
            elif nums[mid] > target:
                r = mid - 1
            else:
                l = mid + 1
        return res
```

#### 2.4 找插入位置

```python
        if nums[-1] < target:
            return len(nums) + 1
        l, r, res = 0, len(nums) - 1, -1
        while l <= r:
            mid = l + r >> 1
            if nums[mid] == target:
                res = mid # 可能的候选
                r = mid - 1
            elif nums[mid] > target:
                res = mid  # 可能的候选
                r = mid - 1
            else:
                # nums[mid] < target, 一定要插在后面
                l = mid + 1
        return res
```

### 三、题解


#### 287. Find the Duplicate Number
```
Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive.

There is only one repeated number in nums, return this repeated number.

You must solve the problem without modifying the array nums and uses only constant extra space.

 

Example 1:

Input: nums = [1,3,4,2,2]
Output: 2
Example 2:

Input: nums = [3,1,3,4,2]
Output: 3
 

Constraints:

1 <= n <= 105
nums.length == n + 1
1 <= nums[i] <= n
All the integers in nums appear only once except for precisely one integer which appears two or more times.

```


```python
class Solution(object):
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        n = len(nums)
        l, r = 1, n - 1
        while l < r:
            mid = l + r >> 1
            cnt = 0
            for i in range(n):
                if nums[i] <= mid:
                    cnt += 1
            if cnt == mid:
                l = mid + 1
            elif cnt < mid:
                l = mid + 1
            else:
                r = mid
        return r # must have answer

```

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

```
input:  shiftArr = [9, 12, 17, 2, 4, 5], num = 2 # shiftArr is the
                                                 # outcome of shifting
                                                 # [2, 4, 5, 9, 12, 17]
                                                 # three times to the left

output: 3 # since it’s the index of 2 in arr
```



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
        l, r, res = 0, 10**6, -1
        while l <= r:
            mid = (l + r) >> 1
            val = check(nums, mid)
            if val == k:
                res = mid
                r = mid - 1
            elif val > k:
                res = mid
                r = mid - 1 
            else:
                l = mid + 1
        return res


```


#### leetcode34:从数组中找到第一个出现或者最后一个出现的位置

```python


class Solution(object):

    def find_left_bound(self, nums, target):
        """找左边界"""
        l, r, res = 0, len(nums) - 1, -1
        while l <= r:
            mid = l + r >> 1
            if nums[mid] == target:
                res = mid
                r = mid - 1
            elif nums[mid] > target:
                r = mid - 1
            else:
                l = mid + 1
        return res

    def find_right_bound(self, nums, target):
        """找右边界"""
        l, r, res = 0, len(nums) - 1, -1
        while l <= r:
            mid = l + r >> 1
            if nums[mid] == target:
                res = mid
                l = mid + 1
            elif nums[mid] > target:
                r = mid - 1
            else:
                l = mid + 1
        return res

    def searchRange(self, nums, target):
        """leetcode34:从数组中找到第一个出现或者最后一个出现的位置
Example:
nums = [5,7,7,8,8,10], target = 8
[3,4]

题解：利用res记录访问过的边界，然后持续调整l和r缩短范围
        """
        return [self.find_left_bound(nums, target), self.find_right_bound(nums, target)]


# nums = [5,7,7,8,8,10]
# target = 8

nums = [8,8,8,8,8,8,8,8,8,8]
target = 8

s = Solution()
print(s.searchRange(nums, target))

```

#### leetcode35 搜索插入的位置


```python

class Solution(object):

    def find_left_bound(self, nums, target):
        """leetcode35 搜索插入的位置
        nums = [1,3,5,6], target = 5
        result:2
        题解：还是可以用之前的模板，不断缩小位置，记录可能的候选位置，当nums[mid]==target, 以及nums[mid] > target， 都是可能的插入位置，另外如果target>nums[-1]这里没有合适的插入位置，需要提前处理。
        """
        if nums[-1] < target:
            return len(nums) + 1
        l, r, res = 0, len(nums) - 1, -1
        while l <= r:
            mid = l + r >> 1
            if nums[mid] == target:
                res = mid # 可能的候选
                r = mid - 1
            elif nums[mid] > target:
                res = mid  # 可能的候选
                r = mid - 1
            else:
                # nums[mid] < target, 一定要插在后面
                l = mid + 1
        return res


nums = [5,7,7,8,8,10]
target = 5

# nums = [8,8,8,8,8,8,8,8,8,8]
# target = 8

s = Solution()
print(s.find_left_bound(nums, target))

```


#### leetcode354 俄罗斯套娃信封问题

```python


class Solution(object):

    def LIS(self, nums):
        """求1d数组nums的最长递增子序列"""
        n = len(nums)
        dp = [0] * n
        end = -1 # 最长递增数组的右边界
        for i in range(n):
            target = nums[i]
            if end == -1:
                end += 1
                dp[0] = target
            elif target > dp[end]:
                end += 1
                dp[end] = target
            else:
                l, r = 0, end
                target_index = -1
                while l <= r:
                    mid = l + r >> 1
                    if dp[mid] > target:
                        target_index = mid
                        r = mid - 1
                    elif dp[mid] < target:
                        l = mid + 1
                    else:
                        target_index = mid
                        r = mid - 1
                dp[target_index] = target
        return end + 1

    def maxEnvelopes(self, nums):
        """leetcode354 俄罗斯套娃信封问题
        如果信封的w和h都比上一个大，那么上一个可以放到当前这个里面，问做多能套多少个。
        题解：2d最长递增子序列，先对w从小到大排序，由于w相等时候，是不能套进去的，所以对h这个维度计算最长递增子序列的时候，将w相同的h按降序排列即可完成。
        """
        nums.sort(key=lambda x:[x[0], -x[1]])
        heights = [_[1] for _ in nums]
        return self.LIS(heights)


nums = [[5,4],[6,4],[6,7],[2,3]]

s = Solution()
print(s.maxEnvelopes(nums))

```


#### 793. 阶乘函数后 K 个零

```python
class Solution(object):

    def ending_zero_number(self, n):
        """n的阶乘末尾有多少个零"""
        base = 5
        cnt = 0
        while n // base > 0:
            cnt += n // base
            base *= 5
        return cnt

    def right_bound(self, k):
        """末尾有k个零的最大n(右边界)"""
        l, r, res = 0, 1 << 32, -1
        while l <= r:
            mid = l + r >> 1
            cnt = self.ending_zero_number(mid)
            if cnt == k:
                res = mid
                l = mid + 1
            elif cnt > k:
                r = mid - 1
            else:
                l = mid + 1
        return res

    def left_bound(self, k):
        """末尾有k个零的最小n(左边界)"""
        l, r, res = 0, 1 << 32, -1
        while l <= r:
            mid = l + r >> 1
            cnt = self.ending_zero_number(mid)
            if cnt == k:
                res = mid
                r = mid - 1
            elif cnt > k:
                r = mid - 1
            else:
                l = mid + 1
        return res


    def preimageSizeFZF(self, k):
        """阶乘后为k个零的条件的个数
        k=0
        0!, 1!, 2!, 3!, 4!末尾都没有0，共有5个数。
        输出：5
        """
        r = self.right_bound(k)
        l = self.left_bound(k)
        if r == -1:
            return 0
        # print('r:', r, 'l:', l)
        return r - l + 1





# s = Solution()

# print(s.ending_zero_number(0))
# print(s.ending_zero_number(1))
# print(s.ending_zero_number(2))
# print(s.ending_zero_number(3))
# print(s.ending_zero_number(4))
# print(s.ending_zero_number(5))
# print(s.preimageSizeFZF(0))

```