

### 一、初步知识

- 【neko】数据结构 线段树【算法编程#6】:https://www.bilibili.com/video/BV1yF411p7Bt?spm_id_from=333.337.search-card.all.click
- 【专题讲解】 线段树的典型应用 leetcode 307 Range Sum Query - Mutable：https://www.bilibili.com/video/BV1EK4y1D7gT?spm_id_from=333.337.search-card.all.click

### 题解

####

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