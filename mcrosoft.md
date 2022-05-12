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

### SDE2020面试

#### 1轮 151. 颠倒字符串中的单词

```python
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str

        151. 颠倒字符串中的单词

输入：s = "the sky is blue"
输出："blue is sky the"

        起始和结束位置都可能存在空格

        解法：
        1. 先去除首位空格
        2. 双指针遍历每个word，并push到result里面
        3. 利用首位指针
        """

        n = len(s)
        left, right = 0, n - 1

        while left <= right and s[left] == " ":
            left += 1

        while left <= right and s[right] == " ":
            right -= 1

        result = []
        word = []
        while left <= right:
            if s[left] != ' ':
                word.append(s[left])
            else:
                if word:
                    result.append(''.join(word))
                    word = []
            left += 1

        if word:
            result.append(''.join(word))
        
        l, r = 0, len(result) - 1
        while l < r:
            result[l], result[r] = result[r], result[l]
            l += 1
            r -= 1
        
        return ' '.join(result)
        

```

#### 2轮 顺时针生成矩阵

```python

class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        """给你一个正整数 n ，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的 n x n 正方形矩阵 matrix 。

输入：n = 3
输出：[[1,2,3],[8,9,4],[7,6,5]]

        题解：
        分层模拟，注意用完一层就更新
        """



        l, r, t, b = 0, n - 1, 0, n - 1
        result = [[0 for _ in range(n)] for _ in range(n)]
        num, tar = 1, n * n
        while num <= tar:
            for i in range(l, r + 1): # left to right
                result[t][i] = num
                num += 1
            # 每次用完就更新
            t += 1
            for i in range(t, b + 1): # top to bottom
                result[i][r] = num
                num += 1
            # 每次用完就更新
            r -= 1
            for i in range(r, l - 1, -1): # right to left
                result[b][i] = num
                num += 1
            # 每次用完就更新
            b -= 1
            for i in range(b, t - 1, -1): # bottom to top
                result[i][l] = num
                num += 1
            # 每次用完就更新
            l += 1
        return result

# 作者：jyd
# 链接：https://leetcode.cn/problems/spiral-matrix-ii/solution/spiral-matrix-ii-mo-ni-fa-she-ding-bian-jie-qing-x/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

#### 3轮 307. 区域和检索 - 数组可修改

```python
class NumArray:
    def __init__(self, nums: List[int]):
        self.nums = nums
        # 初始化为0
        self.tree = [0] * (len(nums) + 1)
        for i, num in enumerate(nums):
            # 树状数组的index从1开始
            self.add(i+1, num)

    def lowbit(self, index):
        """求补码+1
        lowbit(index) = index & -index
        -index = ~index + 1, 相当于补码加1
        """
        return index & -index

    def add(self, index: int, val: int):
        """往树状数组里面更新值，index更新后，将其所有父节点相关知识都更新"""
        while index < len(self.tree):
            self.tree[index] += val
            # 树状数组index的父节点为 index + lowbit(index)
            # lowbit(index) = index & -index
            # -index = ~index + 1, 相当于补码加1
            index += self.lowbit(index)


    def prefixSum(self, index) -> int:
        """[1,index]的前缀和
        index该位置包括lowbit(index)个数字的和
        上一个需要加上的和就是index - lowbit(x)
        一直加到0就行
        """
        s = 0
        while index:
            s += self.tree[index]
            index -= self.lowbit(index)
        return s

    def update(self, index: int, val: int) -> None:
        """更新线段树和备份的nums值"""
        # 相当于tree的基础上，当前节点和其父节点加上val-self.nums[index]
        self.add(index + 1, val - self.nums[index])
        # 这里相当于有个nums的备份，也需要更新
        self.nums[index] = val

    def sumRange(self, left: int, right: int) -> int:
        """拿到区间和，right+1才是tree的区间,[left, right]->[left, right+1]"""
        return self.prefixSum(right + 1) - self.prefixSum(left)

# 作者：LeetCode-Solution
# 链接：https://leetcode.cn/problems/range-sum-query-mutable/solution/qu-yu-he-jian-suo-shu-zu-ke-xiu-gai-by-l-76xj/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# obj.update(index,val)
# param_2 = obj.sumRange(left,right)

```

#### 3轮方法2 线段树 307. 区域和检索 - 数组可修改

```python
from typing import List

class Node:

    def __init__(self, left, right, val):
        """构建[left, right]的树"""
        self.left = left
        self.right = right
        self.val = val
        self.sum = 0
        self.left_node = None
        self.right_node = None


class NumArray:

    def __init__(self, nums: List[int]):
        self.root = self.build_tree(nums, 0, len(nums ) -1)


    def build_tree(self, nums, left, right):
        if left > right:
            return None
        # 当前节点
        cur_node = Node(left, right, nums[left])
        if left == right:
            cur_node.sum = nums[left]
            return cur_node
        # 左右节点
        mid = left + (right - left) // 2
        cur_node.left_node = self.build_tree(nums, left, mid)
        cur_node.right_node = self.build_tree(nums, mid + 1, right)
        cur_node.sum = cur_node.left_node.sum + cur_node.right_node.sum
        return cur_node


    def update_node(self, node, index, val):
        """跟新左右孩子中当前节点的值，并根绝左右孩子的sum更新当前sum"""
        if node.left == node.right:
            node.val = val
            # 只有一个节点也需要更新sum
            node.sum = val
            return
        mid = node.left + (node.right - node.left) // 2
        if index > mid:
            self.update_node(node.right_node, index, val)
        else:
            self.update_node(node.left_node, index, val)
        # 根据左右孩子更新当前节点的sum
        node.sum = node.left_node.sum + node.right_node.sum

    def update(self, index: int, val: int) -> None:
        """更新index下节点的值"""
        self.update_node(self.root, index, val)


    def query(self, node, left, right):
        """查询某节点[node.left, node.right]下的[left, right]之间的区域和"""
        # node包括在范围内，直接向上输出
        # print('node.left, node.right, left, right:', node.left, node.right, left, right)
        if left > right or left > node.right or right < node.left:
            return 0
        if left <= node.left and node.right <= right:
            return node.sum
        left_acc = self.query(node.left_node, left, right)
        right_acc = self.query(node.right_node, left, right)
        return left_acc + right_acc




    def sumRange(self, left: int, right: int) -> int:
        """求区域和"""
        return self.query(self.root, left, right)




# Your NumArray object will be instantiated and called as such:

# ["NumArray","sumRange","sumRange","sumRange","update","update","update","sumRange","update","sumRange","update"]
# [[[0,9,5,7,3]],[4,4],[2,4],[3,3],[4,5],[1,7],[0,8],[1,2],[1,9],[4,4],[3,4]]

# nums = [1,2,3,4,5]
# nums = [0,9,5,7,3]
# obj = NumArray(nums)
# print(obj.sumRange(left=4,right=4))
# print(obj.sumRange(left=1,right=2))
# print(obj.sumRange(left=1,right=4))
# obj.update(index=1,val=3)
# print(obj.sumRange(left=1,right=2))





```