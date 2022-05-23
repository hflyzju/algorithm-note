### 一、模板
#### 1. 左右递增栈模板1
```python

        nums = [0] + nums + [0]        
        # 右边第一个比它小的元素下标
        right_first_smaller = [None] * len(nums)
        stack = []
        for i in range(len(nums)):
            # 如果当前元素比栈顶元素小，弹栈
            while stack and nums[i] < nums[stack[-1]]:
                right_first_smaller[stack.pop()] = i
            stack.append(i)
        # 左边第一个比它小的元素下标
        left_first_smaller = [None] * len(nums)
        stack = []
        for i in range(len(nums)-1,-1,-1):
            # 如果当前元素比栈顶元素小，弹栈
            while stack and nums[i] < nums[stack[-1]]:
                left_first_smaller[stack.pop()] = i
            stack.append(i)
```

#### 2.左右递增栈模板2

```python
        n = len(arr)
        left = [-1] * n
        cache = []
        for i in range(n):
            while cache and arr[cache[-1]] > arr[i]:
                cache.pop()
            if not cache:
                left[i] = 0
            else:
                left[i] = cache[-1] + 1
            cache.append(i)

        right = [-1] * n
        cache = []
        for i in range(n-1, -1, -1):
            while cache and arr[cache[-1]] >= arr[i]: # 需要等于，badcase: [71,55,82,55]
                cache.pop()
            if not cache:
                right[i] = n - 1
            else:
                right[i] = cache[-1] - 1
            cache.append(i)

```


### 二、题解




#### 224. 基本计算器 I
#### 227. 基本计算器 II

```python

from typing import List
from collections import deque

class Solution:
    def calculate(self, s):
        """实现计数器
        Example:
            input: 1 + 1 * (2 + 3) / 3
            output: 2

            input: 0 + 1 * (2 - 7) / 3
            output: -1

        """
        cache = deque()
        for si in s:
            cache.append(si)

        def calculate_bracket(cache):
            """计算当前括号内的值"""
            stack = []
            # 每次添加num的时候，看的是上一个遇到的sign的脸色
            pre_sign = '+'
            num = 0
            # 这里用deque可以一直从左消耗deque里面的元素，并且递归的时候也可以持续消耗
            while cache:
                cur = cache.popleft()
                if cur.isdigit():
                    num = num * 10 + int(cur)
                if cur == '(':
                    # 递归消耗cache
                    num = calculate_bracket(cache) # 递归拿到当前括号内的值
                # 如果不为数字且不为空，或者 到达终点了
                if (not cur.isdigit() and cur !=' ') or len(cache) == 0:
                    if pre_sign == '+':
                        stack.append(num)
                    elif pre_sign == '-':
                        stack.append(-num)
                    elif pre_sign == '*':
                        stack[-1] *= num
                    elif pre_sign == '/':
                        stack[-1] = int(stack[-1] / float(num))
                    num = 0
                    pre_sign = cur
                if cur ==')':
                    break
            return sum(stack)
        return calculate_bracket(cache)

s = Solution()

print(s.calculate("1 + 1 * (2 + 3) / 3")==2)
print(s.calculate("0 + 1 * (2 - 7) / 3")==-1)


```



#### 83 最大的矩形面积-单调栈

```python
class Solution(object):
    def largestRectangleArea(self, heights):
        """计算可能的最大的矩形的面积
        :type heights: List[int]
        :rtype: int
        Example:
            # Input: heights = [2,1,5,6,2,3]
            # Output: 10
            # heights = [2,1,5,6,2,3]
            # [2,5,6,110,111,111,3]
        Solution:
            0. 最核心是找到每个高度最长的左边界和右边界
            1. 单调栈，递增的时候就加栈里面，否者证明前面的高度需要计算了，因为后面没有比前面高的了
            2. 如何计算前面的高度，[2,5,6,110,111,111]，这里111先一直往回收缩，如果遇到比他小的了，那么111的宽度就出来了，这里是2
            3. [2,5,6,110] 计算完111后，还需要计算110，这种case，直到都比三小 -> [2, 3]
            4.[2, ..., 3] 这里代表2还没有被计算过，因为2后面的数都比2大，所以2可以一直往后计算
            5. 最后栈里面是一个递增数组，还需要最终算一下，因为存的是index，所以算的时候可以拿到w和h
        """
        n = len(heights)
        res = 0
        # 记录的是index
        stack = []
        for i in range(n):
            # 当当前stack不为空 并且新来的i小于最后一个index的高度，前面的高度111到头了，需要高为111时候的面积了
            # [2,5,6,110,111,111]  heights[i]=3
            while stack and heights[i] < heights[stack[-1]]:
                # 把最后的index拿出来，这个cur_height的宽度不可能再高了，需要在这个时候被算掉
                cur_height = heights[stack.pop()] # [2,5,6,110,111] cur_height=111
                # 如果还有和当前高度cur_height相等的，继续怼出来
                while stack and cur_height == heights[stack[-1]]:
                    stack.pop() # [2,5,6,110] cur_height=111
                if len(stack) > 0:
                    # [2,5,6,110, [111, 111], i]
                    # 这里相当于计算的是[111, 111]这个单前最高高度的宽度
                    # 后面再记录[100, '111', '111']的宽度
                    # 直到所有比i大的高度都计算完了
                    # 就可以考虑把下一个装进来了
                    # 这里就拿到cur_height的宽度了
                    cur_width = i - stack[-1] - 1
                else:
                    cur_width = i
                # 后面继续，如果有比heights[i]高的，这个时候都要被计算掉
                res = max(res, cur_height * cur_width)
            stack.append(i)

        # 跑完一轮里面可能还有东西
        # [111, 112, 113, 114]
        # 可能是一直递增的，这里还需要处理下
        while stack:
            cur_height = heights[stack.pop()]
            while len(stack) > 0 and cur_height == heights[stack[-1]]:
                stack.pop()
            if len(stack) > 0:
                cur_width = n - stack[-1] - 1
            else:
                cur_width = n
            res = max(res, cur_height * cur_width)
        return res

```


#### 907. 子数组的最小值之和


```python
class Solution(object):
    def sumSubarrayMins(self, arr):
        """907. 子数组的最小值之和
        :type arr: List[int]
        :rtype: int
题目：
给定一个整数数组 arr，找到 min(b) 的总和，其中 b 的范围为 arr 的每个（连续）子数组。
由于答案可能很大，因此 返回答案模 10^9 + 7 。

输入：arr = [3,1,2,4]
输出：17
解释：
子数组为 [3]，[1]，[2]，[4]，[3,1]，[1,2]，[2,4]，[3,1,2]，[1,2,4]，[3,1,2,4]。 
最小值为 3，1，2，4，1，1，2，1，1，1，和为 17。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/sum-of-subarray-minimums
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

题解：
    1. 每个数字影响的范围与它左右两边第一个小于它的位置有关，例如[3, [1], 2 ,4], 这个1，包括1的子数组的个数为2*3=6个，左右两边分别利用单调栈可以解决。
    2. 重复出现的数字，要注意只往一边扩展可以避免重复计算。[71,55,82,55]

        [3]
        [1]
        [1,2]
        [1,2,4]
        """
        mod = 10 ** 9 + 7
        n = len(arr)
        left = [-1] * n
        cache = []
        for i in range(n):
            while cache and arr[cache[-1]] > arr[i]:
                cache.pop()
            if not cache:
                left[i] = 0
            else:
                left[i] = cache[-1] + 1
            cache.append(i)

        right = [-1] * n
        cache = []
        for i in range(n-1, -1, -1):
            while cache and arr[cache[-1]] >= arr[i]: # 需要等于，badcase: [71,55,82,55]
                cache.pop()
            if not cache:
                right[i] = n - 1
            else:
                right[i] = cache[-1] - 1
            cache.append(i)
        # print('cache:', cache)
        # print('left:',left)
        # print('right:',right)

        s = 0
        for i in range(n):
            left_cnt = i - left[i] + 1
            right_cnt = right[i] - i + 1
            cur = left_cnt * right_cnt * arr[i] % mod
            s += cur
            # print('left_cnt:', left_cnt)
            # print('right_cnt:', right_cnt)
            # print('i:', i, 'cur:', cur)
            # print('s:', s)
            # print('='*10)
            s = s % mod
        return s

```


#### 907. 子数组的最小值之和 - 方法二，dp+单调栈


```python

class Solution:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        """

输入：arr = [3,1,2,4]
输出：17
解释：
子数组为 [3]，[1]，[2]，[4]，[3,1]，[1,2]，[2,4]，[3,1,2]，[1,2,4]，[3,1,2,4]。 
最小值为 3，1，2，4，1，1，2，1，1，1，和为 17。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/sum-of-subarray-minimums
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

        解法二：动态规划+单调栈
        result[i]:代表包含i在内的，所有min子数组和。
        result[i] = result[stack[-1]] + (i - stack[-1]) * arr[i]
        """
        stack = []
        result = [0] * len(arr)

        for i in range(len(arr)):
            while stack and arr[stack[-1]] > arr[i]:
                stack.pop()
            # 如果前面没有更小值，则以arr[i]结尾的所有subarray的minimum都是arr[i]，和为(i+1)*arr[i]
            # 如果前面有更小值，则在起点在这个更小值之前的subarray的minimum由这个更小值决定，即为result[stack[-1]]，在两者之间开始的subarray的最小值为arr[i]，和为(i-stack[-1])*arr[i]
            if stack:
                result[i] = result[stack[-1]] + (i-stack[-1]) * arr[i]
            else:
                result[i] = (i+1) * arr[i]
            stack.append(i)
        return sum(result) % (10**9+7)

# 作者：haodong-du
# 链接：https://leetcode.cn/problems/sum-of-total-strength-of-wizards/solution/li-yong-dan-diao-zhan-de-onjie-fa-by-hao-a9i4/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

#### 496. 下一个更大元素 I

```python

class Solution(object):
    def nextGreaterElement(self, nums1, nums2):
        """496. 下一个更大元素 I
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]

输入：nums1 = [4,1,2], nums2 = [1,3,4,2].
输出：[-1,3,-1]
解释：nums1 中每个值的下一个更大元素如下所述：
- 4 ，用加粗斜体标识，nums2 = [1,3,4,2]。不存在下一个更大元素，所以答案是 -1 。
- 1 ，用加粗斜体标识，nums2 = [1,3,4,2]。下一个更大元素是 3 。
- 2 ，用加粗斜体标识，nums2 = [1,3,4,2]。不存在下一个更大元素，所以答案是 -1 

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/next-greater-element-i
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

        解法：倒着单调栈+逆序+递减栈
        """


        cache = []
        next_bigger_num = dict()


        for i in range(len(nums2)-1, -1, -1):
            while cache and cache[-1] < nums2[i]:
                cache.pop()
            if cache:
                next_bigger_num[nums2[i]] = cache[-1]
            cache.append(nums2[i])

        res = []
        for num in nums1:
            if num in next_bigger_num:
                res.append(next_bigger_num[num])
            else:
                res.append(-1)
        return res

```


#### 503. 下一个更大元素 II

```python
class Solution(object):
    def nextGreaterElements(self, nums):
        """503. 下一个更大元素 II
        :type nums: List[int]
        :rtype: List[int]

输入: nums = [1,2,1]
输出: [2,-1,2]
解释: 第一个 1 的下一个更大的数是 2；
数字 2 找不到下一个更大的数； 
第二个 1 的下一个最大的数需要循环搜索，结果也是 2。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/next-greater-element-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

        题解：拼起来使用单调栈
        """

        stack = []
        n = len(nums)
        res = [-1] * n
        for i in range(2*n-1, -1, -1):
            index = i % n
            while stack and stack[-1] <= nums[index]:
                stack.pop()
            if stack and i <= n - 1:
                res[i] = stack[-1]
            stack.append(nums[index])
        return res

                

```



#### 1856. 子数组最小乘积的最大值


```python
class Solution:
    def maxSumMinProduct(self, nums: List[int]) -> int:

        """

输入：nums = [1,2,3,2]
输出：14
解释：最小乘积的最大值由子数组 [2,3,2] （最小值是 2）得到。
2 * (2+3+2) = 2 * 7 = 14 。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/maximum-subarray-min-product
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

        题解：单调栈
        """
        # 左右添加两个哨兵，方便单调栈内的判断
        nums = [0] + nums + [0]
        # 前缀和
        presum = [0]
        for n in nums:
            presum.append(presum[-1] + n)
        
        # 右边第一个比它小的元素下标
        right_first_smaller = [None] * len(nums)
        stack = []
        for i in range(len(nums)):
            # 如果当前元素比栈顶元素小，弹栈
            while stack and nums[i] < nums[stack[-1]]:
                right_first_smaller[stack.pop()] = i
            stack.append(i)

        # 左边第一个比它小的元素下标
        left_first_smaller = [None] * len(nums)
        stack = []
        for i in range(len(nums)-1,-1,-1):
            # 如果当前元素比栈顶元素小，弹栈
            while stack and nums[i] < nums[stack[-1]]:
                left_first_smaller[stack.pop()] = i
            stack.append(i)

        print('left_first_smaller:', left_first_smaller)
        print('right_first_smaller:', right_first_smaller)

        # 打擂台得到答案
        res = 0
        for i in range(1,len(nums)-1):
            left = left_first_smaller[i]
            right = right_first_smaller[i]
            res = max(res, nums[i] * (presum[right] - presum[left+1])) # 因为统计的最大值，所以拿到这两个边界就行了
        return res % (10 ** 9 + 7)

# 作者：musiala
# 链接：https://leetcode.cn/problems/maximum-subarray-min-product/solution/python-qian-zhui-he-dan-diao-zhan-qing-x-gow8/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```




#### 2281. 巫师的总力量和


```python
class Solution:
    def totalStrength(self, n: List[int]) -> int:
        """2281. 巫师的总力量和
输入：strength = [5,4,6]
输出：213
解释：以下是所有连续巫师组：
- [5,4,6] 中 [5] ，总力量值为 min([5]) * sum([5]) = 5 * 5 = 25
- [5,4,6] 中 [4] ，总力量值为 min([4]) * sum([4]) = 4 * 4 = 16
- [5,4,6] 中 [6] ，总力量值为 min([6]) * sum([6]) = 6 * 6 = 36
- [5,4,6] 中 [5,4] ，总力量值为 min([5,4]) * sum([5,4]) = 4 * 9 = 36
- [5,4,6] 中 [4,6] ，总力量值为 min([4,6]) * sum([4,6]) = 4 * 10 = 40
- [5,4,6] 中 [5,4,6] ，总力量值为 min([5,4,6]) * sum([5,4,6]) = 4 * 15 = 60
所有力量值之和为 25 + 16 + 36 + 36 + 40 + 60 = 213 。
。
        题解：https://mp.weixin.qq.com/s?__biz=MzI2NzQ3OTQ1Mw==&mid=2247485030&idx=1&sn=b648c8a3c8a9b8fce625765fb345d233&chksm=eaff7694dd88ff825dd71a9b740dcba6711f192220c718361382b70534718b0d98936b5514c4&token=1764672036&lang=zh_CN#rd
        """
        MOD = 10 ** 9 + 7
        # L，R是左右边界数组
        R = [len(n)] * len(n)
        L = [0] * len(n)
        # 单调栈求得左右边界
        monostack = []
        for i in range(len(n)):
            v = n[i]
            while monostack and monostack[-1][1] > v:
                R[monostack[-1][0]] = i
                monostack.pop()
            monostack.append((i, v))

        monostack = []
        for i in range(len(n) - 1, -1, -1):
            v = n[i]
            while monostack and monostack[-1][1] >= v:
                L[monostack[-1][0]] = i + 1
                monostack.pop()
            monostack.append((i, v))

        # 求前缀和数组，和左右两个三角形前缀和数组
        prefix = [0]
        for v in n:
            prefix.append((prefix[-1] + v) % MOD)
        l_triangle = [0]
        for i, v in enumerate(n):
            l_triangle.append((l_triangle[-1] + v * (i + 1)) % MOD)
        r_triangle = [0]
        for i, v in enumerate(n):
            r_triangle.append((r_triangle[-1] + v * (len(n) - i)) % MOD)

        def prefix_sum(l, r):
            return prefix[r] - prefix[l]

        def l_triangle_sum(l, r):
            return l_triangle[r] - l_triangle[l]

        def r_triangle_sum(l, r):
            return r_triangle[r] - r_triangle[l]

        res = 0
        for i in range(len(n)):
            r = R[i]
            l = L[i]
            # 求左右三角形的面积
            l_sum = l_triangle_sum(0, len(n)) - l_triangle_sum(0, l) - \
                    l_triangle_sum(i + 1, len(n)) - prefix_sum(l, i + 1) * l
            r_sum = r_triangle_sum(0, len(n)) - r_triangle_sum(0, i) - \
                    r_triangle_sum(r, len(n)) - prefix_sum(i, r) * (len(n) - r)
            # 左面积*右行数+右面积*左行数-中间数*左行数*右行数
            l_r_sum = l_sum * (r - i) + r_sum * (i - l + 1) - (r - i) * (i - l + 1) * n[i]
            # 根据题目要求再乘以中间数
            res += l_r_sum * n[i]
            res %= MOD
        return res

```