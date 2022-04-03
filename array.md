hot100数组题目: https://leetcode-cn.com/problem-list/2cktkvj/

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


#### 15 双数之和-双指针

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


#### 347 前 K 个高频元素- 最小堆解决

```python

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        """找到出现频率最小的k个数
        
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