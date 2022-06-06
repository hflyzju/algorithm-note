

### 一、初步知识

- 【neko】数据结构 线段树【算法编程#6】:https://www.bilibili.com/video/BV1yF411p7Bt?spm_id_from=333.337.search-card.all.click
- 【专题讲解】 线段树的典型应用 leetcode 307 Range Sum Query - Mutable：https://www.bilibili.com/video/BV1EK4y1D7gT?spm_id_from=333.337.search-card.all.click

### 二、题解

#### 307. 区域和检索 - 数组可修改

```python
from typing import List

class Node:

    def __init__(self, left, right, val):
        """构建[left, right]的树"""
        self.left = left
        self.right = right
        self.val = val
        self.sum = 0 # 区间和
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


#### 732. 我的日程安排表 III

```python

class Node:
    def __init__(self) -> None:
        self.left_node = None
        self.right_node = None
        self.val = 0 # 记录的是最终的权重
        self.add = 0

class SegmentTree:
    def __init__(self):
        self.root = Node()
    
    @staticmethod
    def update(node: Node, left: int, right: int, l: int, r: int, v: int) -> None:
        """
        left：当前节点的左边界
        right：当前节点的右边界
        l: 更新区间的左边界
        r: 更新区间的右边界
        v: 更新区间增加的值
        """
        # 完全覆盖，直接返回结果了，先不处理孩子, 并加上layze标记
        if l <= left and right <= r:
            # 添加layze标记
            node.add += v
            node.val += v
            return # 直接返回，先不处理孩子，节省时间
        # 没有完全覆盖，先处理layze标记，进行下沉，add置为0
        SegmentTree.pushdown(node)
        mid = (left + right) >> 1
        # 更新左右节点，因为是更新的区间，所以两边都进行检查
        if l <= mid:
            SegmentTree.update(node.left_node, left, mid, l, r, v)
        if r > mid:
            SegmentTree.update(node.right_node, mid + 1, right, l, r, v)
        # 合并左右节点的值
        SegmentTree.pushup(node)
 
    @staticmethod
    def query(node: Node, left: int, right: int, l: int, r: int) -> int:
        """
        left：当前节点的左边界
        right：当前节点的右边界
        l: 查询区间的左边界
        r: 查询区间的右边界
        """
        # 完全覆盖，直接返回结果
        if l <= left and right <= r:
            return node.val
        # 没有覆盖，先确保所有关联的懒标记下沉下去
        SegmentTree.pushdown(node)
        mid = (left + right) >> 1
        # 左右区间都查一下极值
        left_val, right_val = 0, 0
        if l <= mid:
            left_val = SegmentTree.query(node.left_node, lc, mid, l, r)
        if r > mid:
            # 同样为不同题目中的更新方式
            right_val = SegmentTree.query(node.right_node, mid + 1, rc, l, r)
        return max(left_val, right_val)
    
    @staticmethod
    def pushdown(node: Node) -> None:
        # 懒标记, 在需要的时候才开拓节点和赋值，如果没有，分裂拿到左右的值
        if node.left_node is None:
            node.left_node = Node()
        if node.right_node is None:
            node.right_node = Node()
        if not node.add:
            return
        node.left_node.add += node.add
        node.right_node.add += node.add
        node.left_node.val += node.add
        node.right_node.val += node.add
        # 去除layze标记
        node.add = 0
    
    @staticmethod
    def pushup(node: Node) -> None:
        # 动态更新方式：此处为最大值
        node.val = max(node.left_node.val, node.right_node.val)


class MyCalendarThree:
    """区间最大
当 k 个日程安排有一些时间上的交叉时（例如 k 个日程安排都在同一时间内），就会产生 k 次预订。
给你一些日程安排 [start, end) ，请你在每个日程安排添加后，返回一个整数 k ，表示所有先前日程安排会产生的最大 k 次预订。

输入：
["MyCalendarThree", "book", "book", "book", "book", "book", "book"]
[[], [10, 20], [50, 60], [10, 40], [5, 15], [5, 10], [25, 55]]
输出：
[null, 1, 1, 2, 3, 3, 3]

题解：区间树，涉及到区间的修改，需要layze标记，layze标记需要在update和query的时候都更新
    """
    def __init__(self):
        self.st = SegmentTree()
        self.max_range = int(1e9)

    def book(self, start: int, end: int) -> int:
        """
        start：更新的起始时间
        end：更新的末尾时间
        SegmentTree定义：
        left：当前节点的左边界
        right：当前节点的右边界
        l: 更新区间的左边界
        r: 更新区间的右边界
        v: 更新区间增加的值
        """
        SegmentTree.update(self.st.root, left=0, right=self.max_range, l=start, r=end - 1, v=1)
        return SegmentTree.query(self.st.root, left=0, right=self.max_range, l=0, r=self.max_range)


# Your MyCalendarThree object will be instantiated and called as such:
# obj = MyCalendarThree()
# param_1 = obj.book(start,end)

# 作者：himymBen
# 链接：https://leetcode.cn/problems/my-calendar-iii/solution/pythonjavatypescriptgo-by-himymben-jb1u/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```