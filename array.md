hot100数组题目: https://leetcode-cn.com/problem-list/2cktkvj/
### 一、知识点

1. hash，二数之和
2. 双指针
3. 二分
4. 回溯
5. 子数组去重
6. 前缀和
9. 桶排序



### 二、例题
#### 4. Median of Two Sorted Arrays 寻找两个排序数组的中位数-二分
```python
class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """求两个排序数组的中位数
        解法：二分法
        """
        m = len(nums1)
        n = len(nums2)
        # 第一个数组的长度小于第二个数组的长度，这样可以确保k1和k2的值是有效的
        if m > n:
            m, n, nums1, nums2 = n, m, nums2, nums1
        l, r = 0, m
        while l <= r:
            k1 = l + (r - l) // 2
            k2 = (m + n + 1) // 2 - k1
            # k1 [0, m], k2 [0, n]
            # 只需要看k1是否是有效范围内就行了?
            # k2一定是有效范围内
            if k1 < m and nums2[k2-1] > nums1[k1]:
                l = k1 + 1
            elif k1 > 0 and nums1[k1-1] > nums2[k2]:
                r = k1 - 1
            else:
                if k1 == 0:
                    left_max = nums2[k2-1]
                elif k2 == 0:
                    left_max = nums1[k1-1]
                else:
                    left_max = max(nums1[k1-1], nums2[k2-1])
                if (m+n) % 2 == 1:
                    return left_max

                if k1 == m:
                    right_min = nums2[k2]
                elif k2 == n:
                    right_min = nums1[k1]
                else:
                    right_min = min(nums2[k2], nums1[k1])

                return (left_max + right_min) / 2.0
```


#### 11 接雨水-双指针

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        """题目，接雨水，找出两条边，使装的雨水最多
        方法1：双指针
        """
        n = len(height)
        if n <= 1:
            return 0
        # l
        #                 r  
        #[1,8,6,2,5,4,8,3,7]
        l, r = 0, n - 1
        max_area = 0
        while l < r:
            area = (r - l) * min(height[l], height[r])
            if area > max_area:
                max_area = area
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
        return max_area

```


#### 15 三数之和并去重-双指针

```python

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """返回所有三数之和为0并去重的列表
        方法：利用前后判断来去重+双指针缩小范围
        """
        n = len(nums)
        nums.sort()
        ans = []
        for i in range(n):
            # 跳过相同数字
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            k = n - 1
            target = -nums[i]
            j = i + 1
            while j < k:
                # 跳过相同数字
                if j > i + 1 and nums[j] == nums[j - 1]:
                    j += 1
                else:
                    # 缩小范围
                    while j < k and nums[j] + nums[k] > target:
                        k -= 1
                    if j == k:
                        break
                    if nums[j] + nums[k] == target:
                        ans.append([nums[i], nums[j], nums[k]])
                    j += 1
        return ans

# 作者：LeetCode-Solution
# 链接：https://leetcode-cn.com/problems/3sum/solution/san-shu-zhi-he-by-leetcode-solution/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```


#### 15 三数之和并去重-排序+二分

```python
class Solution:

    def binary_search(self, l, r, nums, target):
        while l <= r:
            m = l + (r - l) // 2
            if nums[m] == target:
                return True
            elif nums[m] > target:
                r = m - 1
            else:
                l = m + 1
        return False

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        res = []
        for i in range(n):
            if i > 0 and nums[i-1] == nums[i]:
                continue
            for j in range(i+1, n):
                if j > i + 1 and nums[j-1] == nums[j]:
                    continue
                diff = -nums[i] - nums[j]
                if self.binary_search(j+1, n - 1, nums, diff):
                    res.append([nums[i], nums[j], diff])
        return res
```

#### 31 下一个排列-双指针

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """下一个排列，给一个数组，找出按字典序排列的下一个数组
        方法：
            158476531
            1.因为4<7，所以这里把4换掉，可以找到一个比当前数字大的结果
            2.找到5>4，刚好大于4，这里用5替换4，就会比当前大了
            3.翻转后面的数字，因为后面的数字都是从大到小排列的，翻过来就是从小到大排列了
        """
        i = len(nums) - 2
        # 1. 从后往前，找到第一个逆序的数字
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        # 2. 找到i后面第一个比i大的数字，并替换
        if i >= 0:
            j = len(nums) - 1
            while j >= 0 and nums[i] >= nums[j]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        # 3. 反转后面部分，因为后面部分都是从大到小排列的，所以这里全部swap反过来就是从小到大排列了
        left, right = i + 1, len(nums) - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1

# 作者：LeetCode-Solution
# 链接：https://leetcode-cn.com/problems/next-permutation/solution/xia-yi-ge-pai-lie-by-leetcode-solution/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```

#### 33. 搜索旋转排序数组-二分

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        """从旋转的排序数组中找到target
        example:
        输入：nums = [4,5,6,7,0,1,2], target = 0
        输出：4
        解法：二分法，确定有序的部分，如果target在有序的部分，直接在这里找，否则在另外的区间找
        1. 利用nums[0] <= nums[mid]可以将数组分成左边有序的数组和右边部分有序的数组
            1.1 有序的数组直接二分法找
        2. 部分有序的右边部分，比较nums[mid]和nums[-1]的大小，如果有序并且target在这里面，就在这里面找
        """
        if not nums:
            return -1
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] == target:
                return mid
            # 1. 前面这一部分是有序的
            if nums[0] <= nums[mid]:
                # 1.1 检查是否在前面部分
                if nums[0] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:
                # 右边这一部分也是有序的，在这里面找
                if nums[mid] < target <= nums[len(nums) - 1]:
                    l = mid + 1
                else:
                    r = mid - 1
        return -1

# 作者：LeetCode-Solution
# 链接：https://leetcode-cn.com/problems/search-in-rotated-sorted-array/solution/sou-suo-xuan-zhuan-pai-xu-shu-zu-by-leetcode-solut/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```


#### 34 查找排序数组的左右边界

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)
        def search_left_bound():
            l, r = 0, n - 1
            while l <= r:
                m = (r - l) // 2 + l
                if nums[m] == target:
                    # 收缩右边界
                    r = m - 1
                elif nums[m] > target:
                    r = m - 1
                elif nums[m] <  target:
                    l = m + 1
            # 校验最终的边界
            if l < n and nums[l] == target:
                return l
            return -1

        def search_right_bound():
            l, r = 0, n - 1
            while l <= r:
                m = (r - l) // 2 + l
                if nums[m] == target:
                    # 收缩左边界
                    l = m + 1
                elif nums[m] > target:
                    r = m - 1
                elif nums[m] <  target:
                    l = m + 1
            # 校验最终的边界
            if r >= 0 and nums[r] == target:
                return r
            return -1
        
        return [search_left_bound(), search_right_bound()]

```

#### 39 Combination Sum 返回和为target的子序列-回溯，如何去重的？

- 要当前位置 or 不要当前位置回溯

```python
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """

        n = len(candidates)
        res = []
        def search(i, cur_path, cur_sum):
            if cur_sum == target:
                res.append(cur_path[:])
                return
            if i >= n or cur_sum > target:
                return
            # 1. 直接跳过当前位置
            search(i+1, cur_path, cur_sum)
            # 2. 考虑加上当前位置
            cur_sum += candidates[i]
            cur_path.append(candidates[i])
            search(i, cur_path, cur_sum) # 因为可以一直使用，所以还可以用这个
            cur_sum -= candidates[i]
            cur_path.pop()
        search(0, [], 0)
        return res
``

- 每次都重新选择，以start为起点的下一个

```python
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        # 目的是为了去重
        # candidates.sort() #居然都可以对，测试用例没考虑去重
        n = len(candidates)
        res = []
        def find_sum(start_i, cur_sum, cur_path):
            if cur_sum > target or start_i == n:
                return
            if cur_sum == target:
                res.append(cur_path[:])
                return 
            # 遍历选择下一个， 可以重复使用，包括当前节点
            for j in range(start_i, n):
                cur_path.append(candidates[j])
                find_sum(j, cur_sum+candidates[j], cur_path)
                cur_path.pop()
        find_sum(0,0,[])
        return res

``


#### 347 前 K 个高频元素- 最小堆解决

```python

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        """找到出现频率最大的k个数
        
        方法1：
            1. 先统计每个数组出现的次数。
            2. 利用最小堆，如果当前频次>堆顶最小频次，那么则替换他
        """
        count = collections.Counter(nums)
        heap = []
        for key, cnt in count.items():
            if len(heap) >= k:
                # 最小堆
                if cnt > heap[0][0]:
                    heapq.heappop(heap)
                    heapq.heappush(heap, (cnt, key))
                    # heapq.heapreplace(heap, (cnt, key))
            else:
                heapq.heappush(heap, (cnt, key))
        return [item[1] for item in heap]

# 作者：edelweisskoko
# 链接：https://leetcode-cn.com/problems/top-k-frequent-elements/solution/347-qian-k-ge-gao-pin-yuan-su-zhi-jie-pa-wemd/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

#### 42 接雨水-前缀max

```python
class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        n = len(height)
        max_height_from_left_side = [0] * n
        max_height_from_right_side = [0] * n
        left_pre_height_max, right_pre_height_max = 0, 0
        for i in range(n):
            max_height_from_left_side[i] = left_pre_height_max
            if height[i] > left_pre_height_max:
                left_pre_height_max = height[i]
            max_height_from_right_side[n-i-1] = right_pre_height_max
            if height[n-i-1] > right_pre_height_max:
                right_pre_height_max = height[n-i-1]
        
        s = 0
        for i in range(n):
            bound = min(max_height_from_left_side[i], max_height_from_right_side[i])
            if height[i] < bound:
                s += (bound - height[i])

        return s

```

#### 46 全排列

```python
class Solution(object):
    def permute(self, nums):
        """返回所有全排列
        :type nums: List[int]
        :rtype: List[List[int]]
        Example:
            #  Input: nums = [1,2,3]
            # Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
        Solution:
            1. 每次遍历从0-n，没有访问过就访问。
        """
        result = []
        n = len(nums)
        def search(path, visited):
            if len(path) == n:
                result.append(path[:])
                return
            for i in range(n):
                if 1 << i & visited == 0:
                    path.append(nums[i])
                    visited = visited | 1 << i
                    search(path, visited)
                    path.pop()
                    visited = visited ^ (1 << i)
        search([], 0)
        return result

```

#### 48 旋转图像

```python
class Solution(object):
    def rotate(self, matrix):
        """旋转图像
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.

        Example:
            # Input: matrix = [[1,2,3],
                               [4,5,6],
                               [7,8,9]]
            # Output:         [[7,4,1],
                               [8,5,2],
                               [9,6,3]]

        题解：
            对于左上1/4的区域内的点，与其他3个1/4区域的对应的点，共4个点做一次旋转即可。
        """
        n = len(matrix)
        # 一次换四个点
        # 总共只要换1/4的区域就行了,这1/4的区域按下面情况选出来就行
        for i in range(n//2):
            for j in range(i, n - i - 1):
                matrix[i][j],matrix[j][n-i-1],matrix[n-i-1][n-j-1],matrix[n-j-1][i] = \
                matrix[n-j-1][i], matrix[i][j],matrix[j][n-i-1],matrix[n-i-1][n-j-1]

```



#### 53 最大子数组和

```python
class Solution(object):
    def maxSubArray(self, nums):
        """最大连续数组和
        :type nums: List[int]
        :rtype: int

        Example:
            # Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
            # Output: 6
            # Explanation: [4,-1,2,1] has the largest sum = 6.
            注意：[-2] -> [-2]
                 [-3,-2] -> [-2]
        Solution:
            dp[i]:包含i在内，可以取得的最大的子数组和
        """
        n = len(nums)
        max_subarray_sum = nums[0]
        if nums[0] < 0:
            pre_sum = 0
        else:
            pre_sum = nums[0]
        for i in range(1, n):
            num = nums[i]
            cur_sum = pre_sum + num
            max_subarray_sum = max(max_subarray_sum, cur_sum)
            if cur_sum > 0:
                pre_sum = cur_sum
            else:
                pre_sum = 0
        return max_subarray_sum

```

#### 55 跳跃游戏

```python
class Solution(object):
    def canJump(self, nums):
        """能否jump到最后的位置
        :type nums: List[int]
        :rtype: bool

        Example:
            # Input: nums = [2,3,1,1,4]
            # Output: true
            # Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
        """
        n = len(nums)
        max_jump_position = 0
        for i in range(n):
            # 当前能跳到
            if max_jump_position >= i:
                now_jump = i + nums[i]
                if now_jump > max_jump_position:
                    max_jump_position = now_jump
        return max_jump_position >= n - 1
```


#### 56 合并区间

```python
class Solution(object):
    def merge(self, intervals):
        """合并相同的区域
        :type intervals: List[List[int]]
        :rtype: List[List[int]]

        Example:
            # Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
            # Output: [[1,6],[8,10],[15,18]]
            # Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].
        """

        intervals.sort(key=lambda x:x[0])
        cache = []
        l = 0
        while l < len(intervals):
            if not cache:
                cache.append(intervals[l])
                l += 1
            else:
                cur = intervals[l]
                pre = cache.pop()
                if cur[0] <= pre[1]:
                    # 注意[[1,4],[2,3]]这种合并
                    cache.append([pre[0], max(cur[1], pre[1])])
                    l += 1
                else:
                    cache.append(pre)
                    cache.append(cur)
                    l += 1
        return cache

```


#### 64 不同路径

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        """从左上到右下的路径的条数
        Example:
            输入：m = 3, n = 7
            输出：28
        Solution:
            dp
        """

        cache = dict()
        def count_path_number(i, j):
            if i < 0 or i >= m or j < 0 or j >= n:
                return 0
            if i == 0 and j == 0:
                return 1
            if (i, j) not in cache:
                cache[(i, j)] = count_path_number(i - 1, j) + count_path_number(i, j - 1)
            return cache[(i, j)]
    
        return count_path_number(m-1, n-1)

```

#### 83 最大的矩形面积

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

#### 164 求排序后数组的最大相邻gap，必须用线性时间-桶排序

```python
class Solution(object):
    def maximumGap(self, nums):
        """求排序后数组的最大相邻gap，必须用线性时间
        example：
            Input: nums = [3,6,9,1]
            Output: 3
        方法：桶排序
        1. 如果能将每个数字放到一个个桶中，只计算桶间的最大gap就行了的话，时间可以到O(n)
        2. 如果桶间的gap始终要大于桶内gap的话，那么只计算桶间gap就行了。
        3. 如果桶的intervals >= (max_val - min_val) / (n - 1)的话，桶内的gap最大为intervals，桶间的距离要 >= intervals，这时候，只计算桶间gap就行了。
        4. 注意有empty桶情况下的桶间gap的计算
        """
        n = len(nums)
        if n <= 1:
            return 0
        max_val, min_val = max(nums), min(nums)
        if max_val - min_val == 0:
            return 0
        
        interval = (max_val - min_val) // (n - 1)
        if (max_val - min_val) % (n - 1) > 0:
            interval += 1

        buket_max = [-1] * (n - 1)
        buket_min = [float('inf')] * (n - 1)

        for i in range(n):
            if nums[i] == max_val or nums[i] == min_val:
                continue
            index = (nums[i] - min_val) // interval
            buket_max[index] = max(buket_max[index], nums[i])
            buket_min[index] = min(buket_min[index], nums[i])

        max_gap = 0
        pre_max = min_val
        for index in range(n - 1):
            if buket_max[index] == -1:
                continue
            max_gap = max(max_gap, buket_min[index] - pre_max)
            pre_max = buket_max[index]
        
        max_gap = max(max_gap, max_val - pre_max)
        return max_gap

```

#### 220. Contains Duplicate III 满足index_diff < k , val_diff < t的邻居是否存在 - 桶排序

```python
class Solution(object):
    def containsNearbyAlmostDuplicate(self, nums, k, t):
        """
        :type nums: List[int]
        :type k: int
        :type t: int
        :rtype: bool
        1. 以t+1为桶的intervals
        2. 求所在桶的index，如果桶存在，则代表存在该邻居
        3. 依次删除index_diff>k的邻居，确保查询的邻居index_diff在k以内
        """

        if t < 0:
            return False
        n = len(nums)
        cache = dict()
        buket_width = t + 1

        def get_index(num, w):
            if num > 0:
                return num // w
            else:
                # 0/2 -> index=0
                # 1/2 -> index=0
                # -1/2 -> 0? -> -1
                return (num) // w - 1

        for i in range(n):
            index = get_index(nums[i], buket_width)
            # 有数字在同一个桶中，直接返回True
            if index in cache:
                return True
            if index - 1 in cache and abs(nums[i] - cache[index - 1]) < buket_width:
                return True
            if index + 1 in cache and abs(nums[i] - cache[index + 1]) < buket_width:
                return True
            cache[index] = nums[i]
            if i >= k:
                cache.pop(get_index(nums[i - k], buket_width))
        return False

```

#### 436. Find Right Interval- 二分
```python
class Solution(object):
    def findRightInterval(self, intervals):
        """
        题目：给一堆区间[[x,y], ...]，找到每个区间的下一个区间，注意返回最小的x的区间
        题解：先对x排序，对于每个y, 找到x>y的最小的结果即可，x已经排好序了，找到比y大的下一个x就行了
        Example:
            # Input: intervals = [[3,4],[2,3],[1,2]]
            # Output: [-1,0,1]
            # Explanation: There is no right interval for [3,4].
            # The right interval for [2,3] is [3,4] since start0 = 3 is the smallest start t
            # hat is >= end1 = 3.
            # The right interval for [1,2] is [2,3] since start1 = 2 is the smallest start t
            # hat is >= end2 = 2.
        """

        def binarySearch(nums, target):  # 二分查找函数
            """找到最小的大于target的位置"""
            n = len(nums)
            if nums[n - 1] < target:
                return float('inf')  # 刨除特例，返回inf，返回-1遇见区间为负的用例会踩坑
            l, r = 0, n - 1
            while l < r:
                mid = l + (r - l) // 2
                if nums[mid] < target:
                    l = mid + 1
                else:
                    r = mid
            # 这里返回nums[l],返回的是starts的值而不是下标
            # 例如starts=[1,2,3] binareSearch(starts,2)=2 而不是下标1
            return nums[l]

        n = len(intervals)
        # 建立哈希表存储start的下标
        starts = []
        for interval in intervals:
            starts.append(interval[0])
        hashmap = {}
        for i in range(n):
            hashmap[starts[i]] = i
        starts.sort()  # 排序，后面二分查找
        #  [[3,4],[2,3],[1,2]]  ->  [[1,2],[2,3],[3,4]]
        res = [0] * n
        for i in range(n):
            val = binarySearch(starts, intervals[i][1])
            res[i] = hashmap[val] if val != float('inf') else -1
        return res

# 作者：wo-zhao-wo-de-bao-zhen
# 链接：https://leetcode-cn.com/problems/find-right-interval/solution/by-wo-zhao-wo-de-bao-zhen-o8m8/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

#### 560 返回和为k的子数组的个数-前缀和+两数之和

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        """返回和为k的子数组的个数
        example:
        Input: nums = [1,1,1], k = 2
        Output: 2
        Input: [1,1,1,3,4,5,0,0,2,0]
        Output: 8
        思路1. 双指针? 不可行，因为需要统计每个可能子区间的个数。双指针容易跳过。
        思路2，前缀和+哈希表，可行，前缀和记录和为s的个数，当前i的前缀和-前j的前缀和=k即可满足要求，这里统计频次就行。
        """
        pre_sum_hash_cache = defaultdict(int)
        pre_sum_hash_cache[0] = 1
        pre_sum = 0
        cnt = 0
        for i in range(len(nums)):
            cur_sum = pre_sum + nums[i]
            diff = cur_sum - k
            cnt += pre_sum_hash_cache.get(diff, 0)
            pre_sum_hash_cache[cur_sum] += 1
            pre_sum = cur_sum            
        return cnt


```


#### 581 最短的未排序子数组-前后遍历


```python
class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        """给出一个数组，找出一个连续的子数组，如果对该子数组进行排序，那么整个数组就是排序的了
        方法（前后遍历）：
            1. 每个前面的数，都要比后面的所有数要小，否则需要替换
            2. 每个后面的数，都需要比前面的数大，否则就要替换
        """
        n = len(nums)
        if n <= 1:
            return 0
        # [2,3,3,2,4]
        # min = 2
        # i=1, 3 > 2 -> x = 1

        # max = 3
        # i=3, 2 < 3 -> y = 3

        min_val_from_the_right = nums[n-1]
        x, y = -1, -1
        for i in range(len(nums)-2, -1, -1):
            if nums[i] > min_val_from_the_right:
                x = i
            if nums[i] < min_val_from_the_right:
                min_val_from_the_right = nums[i]
        max_val_from_the_left = nums[0]
        for i in range(1, n):
            if nums[i] < max_val_from_the_left:
                y = i
            if nums[i] > max_val_from_the_left:
                max_val_from_the_left = nums[i]
        # print('x:', x, 'y:', y)
        if y > x:
            return y - x + 1
        return 0

```

#### 621 完成任务的最少时间（每个任务要一定间隔才能继续做）- 贪心算法

```python
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        """
        题目：做一堆任务，每一个任务做了后需要间隔n才能继续做，问最少需要多少时间才能把任务做完。
        解法：
            1. 统计每个任务的频次
            2. total_time = (max_cnt - 1) * (n + 1)就是前面做的任务的个数
            3. total_time + max_time_cnt
            4. 注意，如果len(tasks)>total_time, 代表有很多其他的任务，最终取len(tasks)
            详情：https://leetcode-cn.com/problems/task-scheduler/solution/tan-xin-tu-jie-dai-ma-jian-ji-by-chen-we-gnqo/
        """
        task_cnt = defaultdict(int)
        for task in tasks:
            task_cnt[task] += 1
        
        max_cnt_task, max_cnt = -1, -1
        for k, v in task_cnt.items():
            if v > max_cnt:
                max_cnt_task = k
                max_cnt = v

        all_time = (max_cnt - 1) * (n + 1)

        for k, v in task_cnt.items():
            if v == max_cnt:
                all_time += 1

        return max(all_time, len(tasks))

```