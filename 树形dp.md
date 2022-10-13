

#### 337. 打家劫舍 III

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def rob(self, root):
        """
        :type root: TreeNode
        :rtype: int

337. 打家劫舍 III
输入: root = [3,2,3,null,3,null,1]
       3
   2      3
 null 3  null 1
输出: 7 
解释: 小偷一晚能够盗取的最高金额 3 + 3 + 1 = 7

        """


        def rob_node(node):
            """返回抢该节点的最大收益和不抢该节点的最大收益"""
            if node is None:
                return 0, 0
            
            left_state = rob_node(node.left)
            right_state = rob_node(node.right)
            # 抢当前节点的话，孩子只能不抢
            rob_gain = node.val + left_state[1] + right_state[1]
            # 不抢该节点，那么孩子节点，我可以抢或者不抢，取最大的就行
            not_rob_gain = max(left_state) + max(right_state)
            return rob_gain, not_rob_gain


        return max(rob_node(root))
```


#### 968. 监控二叉树 - 方法一，树形dp

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def minCameraCover(self, root):
        """
        :type root: TreeNode
        :rtype: int

题目：
给定一个二叉树，我们在树的节点上安装摄像头。
节点上的每个摄影头都可以监视其父对象、自身及其直接子对象。
计算监控树的所有节点所需的最小摄像头数量。

输入：[0,0,null,0,0]
输出：1
解释：如图所示，一台摄像头足以监控所有节点。

题解：每个节点的状态有，装相机，不装相机，被覆盖，不被覆盖。
1. 当前节点装相机->满足->最小需要多少相机。
2. 当前不装相机，满足的情况下->最小需要多少相机。
3. 当前不装相机，不满足（但是孩子需要满足）情况下->最小需要多少相机

1.左右满足或者不满足都可以
cur_set = min(l_set, l_not_set_but_cover, l_not_set_not_cover) + min(r_set, r_not_set_but_cover, r_not_set_not_cover) + 1
2. 左右一个装了就行
cur_not_set_but_cover = min(l_set + min(r_set, r_not_set_but_cover), r_set + min(l_not_set_but_cover, l_set))
3. 左右都满足就行，当前不满足
cur_not_set_not_cover = l_not_set_but_cover + r_not_set_but_cover
        """
        def dfs(root):
            """
            状态0：当前节点安装相机的时候，需要的最少相机数
            状态1：当前节点不安装相机，但是能被覆盖到的时候，需要的最少相机数
            状态2：当前节点不安装相机，也不能被覆盖到的时候，需要的最少相机数
            """
            if not root:
                return [1, 0, 0]
            l_set, l_not_set_but_cover, l_not_set_not_cover = dfs(root.left)
            r_set, r_not_set_but_cover, r_not_set_not_cover = dfs(root.right)
            # 左右装，不装满足，不装不满足都可以
            cur_set = min(l_set, l_not_set_but_cover, l_not_set_not_cover) + min(r_set, r_not_set_but_cover, r_not_set_not_cover) + 1
            # 左侧装，右侧装或者不装满足 或者右侧装，左侧装或者左侧不装满足
            cur_not_set_but_cover = min(l_set + min(r_set, r_not_set_but_cover), r_set + min(l_not_set_but_cover, l_set))
            # 当前不装不满足，可能由父节点来装，但是孩子需要覆盖到
            cur_not_set_not_cover = l_not_set_but_cover + r_not_set_but_cover
            return [cur_set, cur_not_set_but_cover, cur_not_set_not_cover]

        cur_set, cur_not_set_but_cover, cur_r_not_set_not_cover = dfs(root)
        return min(cur_set, cur_not_set_but_cover)


```