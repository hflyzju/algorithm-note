## 一、总结
|  类型   | 编号  | 题目 | 题解 | 
|  ----  | ----  | --- | --- |
| 非递归遍历  | 144,94 | 二叉树的非递归遍历 |1. 前序遍历：栈，[当前, 右节点, 左节点]的形式入栈。 2. 中序遍历：栈先一直放左孩子，为空后，弹出，此时切换成右孩子，继续尝试放左孩子。  |
| 二叉树构建  | 95,96 |构建所有二叉树，可能二叉搜索树的数量 | 递归，递归或者dp，dp的思想求数量，因为每个树可以由前面的树组合而来 |
| 二叉树搜索  | 98 |验证二叉搜索树 | 1. 范围验证。 2. 中序遍历验证 |
| 两科二叉树关系  | 100，101 | 是否等价，是否镜像 | 1. 检验当前节点，然后递归检验左右孩子。 2. 同1 |
| 序列化和反序列化  | 剑指offer 48序列化反序列化二叉树, 428多叉树,449二叉搜索树 | 序列化和反序列化 | 1. 检验当前节点，然后递归检验左右孩子。 2. 同1 |
| 树形dp  | 100,101 | 是否等价，是否镜像 | 1. 检验当前节点，然后递归检验左右孩子。 2. 同1 |

## 二、模板

### 2.1 非递归遍历
- 先序
```python
if root is None:
    return []
# 中左右
stack = []
res = []
stack.append(root)
while stack:
    # [right, left] 1
    # 栈里面先放右子树，然后放左子树
    cur = stack.pop()
    res.append(cur.val)
    if cur.right:
        stack.append(cur.right)
    if cur.left:
        stack.append(cur.left)
return res
```
- 中序

```python
if root is None:
    return []
stack = []
res = []
cur = root
while cur or stack:
    while cur is not None:
        stack.append(cur)
        cur = cur.left
    cur = stack.pop()
    res.append(cur.val)
    cur = cur.right
return res

```

### 2.2 树的构建

```python
def build_trees(left, right):
    """生成[left, right]区间内的所有可能的二叉搜索树"""
    all_trees = []
    if left > right:
        return [None]
    # 遍历所有左右区间
    for i in range(left, right+1):
        # 1. 先把孩子搞定
        left_trees = build_trees(left, i - 1)
        right_trees = build_trees(i + 1, right)
        # 2. 将所有可能的左孩子+右孩子+当前节点，构成结果进行输出
        for l in left_trees:
            for r in right_trees:
                # 当前节点
                cur_tree = TreeNode(i)
                # 所有的左孩子+右孩子节点
                cur_tree.left = l
                cur_tree.right = r
                all_trees.append(cur_tree)
    return all_trees
return build_trees(1, n)

```
- 统计数量

```python
    def numTrees(self, n):
        """给出n，求1-n所有可能的二进制二叉树的数量
        方法2(从下到上dp)：dp[i]代表1-i的可能的数量，每个位置可以由左边孩子的数量*右边孩子的数量得到
        """
        # 这是一道数学问题
        # G(n)=G(0)∗G(n−1)+G(1)∗(n−2)+...+G(n−1)∗G(0)
        dp = [0 for _ in range(n+1)]
        dp[0] = 1
        dp[1] = 1
        for i in range(2, n + 1):
            for j in range(1, i + 1):
                dp[i] += dp[j-1] * dp[i - j]
        return dp[n]
```

## 三、题解

### 3.1 树的遍历

#### 11111 非递归总结
```c++
        // 栈 先进后出
        // 前序遍历，出栈顺序：根左右; 入栈顺序：右左根
        // 中序遍历，出栈顺序：左根右; 入栈顺序：右根左
        // 后序遍历，出栈顺序：左右根; 入栈顺序：根右左

```

#### 144. 二叉树的前序遍历 - 前序遍历（中左右）非递归

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """144. 二叉树的前序遍历
    1
  2   3
4  5 6  7
先序：1245367
输入：root = [1,null,2,3]
输出：[1,2,3]
        题解：栈模拟，先右入栈，出来的时候左就会优先出来
        """
        if root is None:
            return []
        # 中左右
        stack = []
        res = []
        stack.append(root)
        while stack:
            # [right, left] 1
            # 栈里面先放右子树，然后放左子树
            cur = stack.pop()
            res.append(cur.val)
            if cur.right:
                stack.append(cur.right)
            if cur.left:
                stack.append(cur.left)
        return res

```


#### 94. 二叉树的中序遍历


```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """左中右
        // 栈 先进后出
        // 前序遍历，出栈顺序：根左右; 入栈顺序：右左根
        // 中序遍历，出栈顺序：左根右; 入栈顺序：右根左
        // 后序遍历，出栈顺序：左右根; 入栈顺序：根右左
    1
  2   3
4  5 6  7
中序：4251637
[1,5,]
42
        """
        if root is None:
            return []
        stack = []
        res = []
        cur = root
        while cur or stack:
            while cur is not None:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            res.append(cur.val)
            cur = cur.right
        return res

```


#### 94 中序遍历非递归

```python
class Solution:
    def inorderTraversal(self, root):
        """中序遍历
        Args:
            WHITE=0 代表刚放进来的，不处理先
            GRAY=1 代表第二次遍历了，需要输出了
            栈是先进后出:
                右->中->左进去
                左<-中<-右出来
        """
        WHITE, GRAY = 0, 1
        res = []
        stack = [(WHITE, root)]
        while stack:
            # 栈: list pop 默认是出最后一个
            color, node = stack.pop()
            # 检查时候为空节点
            if node is None:
                continue
            # 栈是先进后出
            # 右->中->左进去
            # 左<-中<-右出来
            if color == WHITE:
                # 先出左(最后一个),在出来中,再出右
                stack.append((WHITE, node.right))
                stack.append((GRAY, node))
                stack.append((WHITE, node.left))
            else:
                res.append(node.val)
        return res

```


#### 94 中序遍历-动态规划写法

```python

class Solution:
    def inorderTraversal(self, root):
        """中序遍历
        动态规划写法，先拿到孩子节点，然后凑成输出
        """
        if root is None:
            return []
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)

```


#### 94 中序遍历-回溯写法

```python
class Solution:

    def __init__(self):
        self.res = []

    def inorderTraversal(self, root):
        """中序遍历-回溯写法
        """
        if root is None:
            return
        self.inorderTraversal(root.left)
        self.res.append(root.val)
        self.inorderTraversal(root.right)
        return self.res

```

#### 102 树的层次遍历


```python
class Solution(object):
    def levelOrder(self, root):
        """层次遍历输出某一层的值
        :param root:
        :return:
        """
        if not root:
            return []
        from collections import deque
        cache = deque()
        cache.append(root)
        rt = []
        while cache:
            rt_tmp = []
            cache_tmp = deque()
            while cache:
                cur = cache.popleft()
                rt_tmp.append(cur.val)
                if cur.left:
                    cache_tmp.append(cur.left)
                if cur.right:
                    cache_tmp.append(cur.right)
            cache = cache_tmp
            rt.append(rt_tmp)
        return rt
```


#### 103 z字形层次遍历


```python

class Solution(object):
    def zigzagLevelOrder(self, root):
        """z字型层次遍历
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        from collections import deque
        cache = deque()
        zflag = False
        cache.append(root)
        rt = []
        while cache:
            tmp_cache = deque()
            tmp_rt = []
            while cache:
                cur = cache.popleft()
                tmp_rt.append(cur.val)
                if cur.left:
                    tmp_cache.append(cur.left)
                if cur.right:
                    tmp_cache.append(cur.right)
            cache = tmp_cache
            if not zflag:
                rt.append(tmp_rt)
            else:
                rt.append(tmp_rt[::-1])
            zflag = not zflag
        return rt
```



### 3.2 构建二叉树

#### 95 构建所有二叉搜索树

```python

class Solution(object):

    def generateTrees(self, n):
        """给你n，构建出所有可能的包含1-n的二叉搜索树
        Args:
            n(int):
        Returns:
            List[TreeNode]:

        解法：
            利用二分+递归的思想，构建l,r之间的树
        """
        if n == 0:
            return []

        def build_trees(left, right):
            """生成[left, right]区间内的所有可能的二叉搜索树"""
            all_trees = []
            if left > right:
                return [None]
            # 遍历所有左右区间
            for i in range(left, right+1):
                # 1. 先把孩子搞定
                left_trees = build_trees(left, i - 1)
                right_trees = build_trees(i + 1, right)
                # 2. 将所有可能的左孩子+右孩子+当前节点，构成结果进行输出
                for l in left_trees:
                    for r in right_trees:
                        # 当前节点
                        cur_tree = TreeNode(i)
                        # 所有的左孩子+右孩子节点
                        cur_tree.left = l
                        cur_tree.right = r
                        all_trees.append(cur_tree)
            return all_trees
        return build_trees(1, n)
```


#### 96. 1-n可能搜索二叉树的数量-从上到下dp

- 递归+从上到下+dp
```python

class Solution(object):
    def numTrees(self, n):
        """给出n，求1-n所有可能的二进制二叉树的数量
        方法1(超时)：遍历当前节点，递归查询左右孩子节点，拿到输出
        """
        if n == 0:
            return 0
        cache = dict()
        def numTreesBetweenLR(l, r):
            """number of trees between [l, r]"""
            if l > r:
                return 0
            if l == r:
                return 1
            if (l, r) in cache:
                return cache[(l, r)]
            cnt = 0
            # 遍历当前节点，统计左右孩子的可能数量，当前节点固定，可能的数量有left*right
            for k in range(l, r + 1): # l=1, r=2, k=1,2,3
                left_child_num = numTreesBetweenLR(l, k - 1) # [1,0], [1,1], [1,2] -> [0, 1, 2]
                right_child_num = numTreesBetweenLR(k + 1, r) # [2, 3], [3, 3], [4,3] -> [2, 1, 0]
                # 左边的可能的数量 * 右边的可能数量
                cnt += max(left_child_num, 1) * max(right_child_num, 1)
            cache[(l, r)] = cnt
            return cnt
        return numTreesBetweenLR(1, n)
```

#### 96 1-n可能二叉搜索树的数量-从下到上dp

```python

class Solution(object):
    def numTrees(self, n):
        """给出n，求1-n所有可能的二进制二叉树的数量
        方法2(从下到上dp)：dp[i]代表1-i的可能的数量，每个位置可以由左边孩子的数量*右边孩子的数量得到
        """
        # 这是一道数学问题
        # G(n)=G(0)∗G(n−1)+G(1)∗(n−2)+...+G(n−1)∗G(0)
        dp = [0 for _ in range(n+1)]
        dp[0] = 1
        dp[1] = 1
        # 对于每个n
        for i in range(2, n + 1):
            # 左中右[0, 1, n-1-0]
            # 左中右[1, 1, n-1-1]
            # 左中右[n-1, 1, 0]
            for j in range(1, i + 1):
                dp[i] += dp[j-1] * dp[i - j]
        return dp[n]
```

#### 654 构建最大的二叉树(每次利用最大值将数组分成两半来构建)


```
给定一个不重复的整数数组 nums 。 最大二叉树 可以用下面的算法从 nums 递归地构建:

创建一个根节点，其值为 nums 中的最大值。
递归地在最大值 左边 的 子数组前缀上 构建左子树。
递归地在最大值 右边 的 子数组后缀上 构建右子树。
返回 nums 构建的 最大二叉树 。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/maximum-binary-tree
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
        def build(l, r):
            """构建[l,r]子区间里面的最大二叉树"""
            if l > r:
                return None
            if l == r:
                return TreeNode(nums[l])
            max_index, max_val = -1, float('-inf')
            for i in range(l, r + 1):
                if nums[i] > max_val:
                    max_index, max_val = i, nums[i]
            left = build(l, max_index - 1)
            right = build(max_index + 1, r)
            cur = TreeNode(nums[max_index])
            cur.left = left
            cur.right = right
            return cur
        return build(0, len(nums) - 1)

```

### 3.3 二叉搜索树

#### 98 有效的二叉搜索树-非递归中序遍历

```python

class Solution(object):
    def isValidBST(self, root):
        """给定一个数，判断是不是二叉搜索树
        Args:
            root: TreeNode
        Returns:
            bool
        解法1：非递归遍历+中序遍历：如果一直都是递减的，那么就是对的
        """
        if root is None:
            return True
        init = False
        small = -(2<<31)
        stack = []
        while( root is not  None or stack != []):
            # 先一直往左走
            while(root is not None):
                stack.append(root)
                root = root.left
            # 左边走完了，拿出一个节点来
            root = stack.pop()
            # 中序遍历是否是递减
            if init and root.val <= small:
                return False
            init = True
            small = root.val
            root = root.right
        return True
```

#### 98 有效的二叉搜索树-递归检查边界

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isValidBST(self, root):
        """验证一棵树是否为搜索二叉树
        :type root: TreeNode
        :rtype: bool
        example:
            Input: root = [2,1,3]
            Output: true

        方法1：二叉搜索树的先序遍历结果是从小到大排列的，用先序遍历一下即可
        方法2：当前节点对左右节点都有限制，限制了左孩子的上限，右孩子的下限，所以可以递归检查边界来解决
        """
        def isValid(node, left_boud, right_boud):
            """检查当前节点是否在给与的边界范围内"""
            if node is None:
                return True
            return left_boud < node.val < right_boud and isValid(node.left, left_boud, node.val) and isValid(node.right, node.val, right_boud)
        return isValid(root, float('-inf'), float('inf'))
        
```

#### 99 复原二叉搜索树: 二叉搜索树有两个节点swap了，叫你复原二叉树


```python

class Solution(object):
    def recoverTree(self, root):
        """复原二叉搜索树: 二叉搜索树有两个节点swap了，叫你复原二叉树
        Args:
            root(Node): 根节点
        Returns:
            None Do not return anything, modify root in-place instead.

        example:
            中序遍历结果：1,2,3,4,5 <-> 1,4,3,2,5
        方法1：中序遍历，遇到值不对的节点，记录下来，这里找到4和2，然后swap值就行
        """
        if not root:
            return
        self.first, self.second = None, None
        self.pre = TreeNode(float("-inf"))

        def in_order(node):
            """中序遍历，递归形式"""
            if not node:
                return
            # 1. 先左边
            in_order(node.left)
            # 2. 当前
            # 比较当前节点和上个位置的大小，如果是逆序，记录下来
            # 第一个逆序是前面那个
            if self.first is None and self.pre.val >= node.val:
                self.first = self.pre
            # 找到第二个逆序的
            # 第二个逆序是后面那个
            if self.first and self.pre.val >= node.val:
                self.second = node
            self.pre = node
            # 3. 右边
            in_order(node.right)

        in_order(root)
        # 交换值,
        self.first.val, self.second.val = self.second.val, self.first.val

```

### 3.4 两棵树比较

#### 100 判断两棵树是否等价


```python
class Solution(object):
    def isSameTree(self, p, q):
        """给定两个根节点，判断两科树是否完全等价
        Args:
            p: TreeNode
            q: TreeNode
        Returns:
            bool
        方法：递归判断当前是否等价，孩子是否等价
        """
        def dfs(cur_p, cur_q):
            """判断两科树是否等价"""
            if cur_p is not None:
                if cur_q is None:
                    return False
                else:
                    # 不为空
                    # 比较左右子树是否等价
                    if cur_p.val == cur_q.val:
                        return dfs(cur_p.left, cur_q.left) & dfs(cur_p.right, cur_q.right)
                    else:
                        return False
            else:
                if cur_q is not None:
                    return False
                else:
                    # 两个都为空，也是等价的
                    return True
        return dfs(p, q)
```


#### 101 是否是镜像树-递归

```python

class Solution(object):
    def isSymmetric(self, root):
        """判断一棵树是否是镜像树
        Args:
            root: TreeNode
        Returns:
            bool
        方法：递归比较孩子的左右孩子是否等价
        """
        if root is None:
            return True
        def isSame(left, right):
            if left is None and right is None:
                return True
            if left is not None and right is None:
                return False
            if right is not None and left is None:
                return False
            return left.val == right.val and isSame(left.left, right.right) and isSame(left.right, right.left)
        return isSame(root.left, root.right)
```



#### 104 树的最大深度

```python
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root is None:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

```

### 3.5 重建二叉树

#### 105 根据先序遍历和中序遍历重建二叉树

```python
class Solution(object):
    def buildTree(self, preorder, inorder):
        """根据先序遍历和中序遍历重建二叉树
        Args:
            preorder: List[int], 先序遍历
            inorder: List[int], 中序遍历
        Returns:
            TreeNode: 根节点
        example:
            # Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
            # Output: [3,9,20,null,null,15,7]

        方法：递归实现，根据先序遍历+中序遍历，可以把做边的孩子和右边的孩子区分出来，递归先构建孩子，再构建当前的数就行了
        """
        def creat_tree_from_preorder_and_inorder(pre_nodes, ino_nodes):
            if not pre_nodes or not ino_nodes:
                return None
            # 拿到当前节点
            cur_val = pre_nodes[0]
            # 找到当前节点在中序遍历的位置
            ino_h_index = ino_nodes.index(cur_val)
            cur = TreeNode(cur_val)
            # 遍历左孩子节点
            childl = creat_tree_from_preorder_and_inorder(pre_nodes[1:ino_h_index+1], ino_nodes[:ino_h_index])
            # 遍历右孩子节点
            childr = creat_tree_from_preorder_and_inorder(pre_nodes[ino_h_index+1:], ino_nodes[ino_h_index+1:])
            cur.left = childl
            cur.right = childr
            return cur
        return creat_tree_from_preorder_and_inorder(preorder, inorder)

```


#### 106 根据后序遍历和中序遍历生成二叉树

```python

class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        Args:
            inorder: List[int], 中序遍历
            postorder: List[int], 后序遍历
        Returns:
            TreeNode
        example:
            # Input: inorder = [9,3,15,20,7], postorder = [9,15,7,20,3]
            # Output: [3,9,20,null,null,15,7]
        方法：
            1. 根据后续遍历，可以找到根节点，根据中序遍历，可以将孩子分为左右两半
        """
        def creat_tree_from_inorder_and_postorder(ino, pos):
            if not ino or not pos:
                return None
            root_val = pos[-1]
            root_val_ino_index = ino.index(root_val)
            root = TreeNode(root_val)
            root.left = creat_tree_from_inorder_and_postorder(ino[:root_val_ino_index], pos[:root_val_ino_index])
            root.right = creat_tree_from_inorder_and_postorder(ino[root_val_ino_index+1:], pos[root_val_ino_index:-1])
            return root
        return creat_tree_from_inorder_and_postorder(inorder, postorder)
```


### 3.5 序列化

#### 剑指 Offer II 048. 序列化与反序列化二叉树 or 297. 二叉树的序列化与反序列化

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        :type root: TreeNode
        :rtype: str
        题解：利用层次遍历，遇到空节点就用#代替
        """
        if root is None:
            return ""
        cache = deque()
        cache.append(root)
        path = []
        while cache:
            node = cache.popleft()
            if node is None:
                path.append("#")
            else:
                path.append(str(node.val))
                cache.append(node.left)
                cache.append(node.right)
        return ','.join(path)
        

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        题解：利用层次遍历，一致消耗节点即可，二叉树一次消耗2个，来构建左右两个节点
        """
        if not data:
            return None
        data_split = data.split(',')
        root = TreeNode(data_split[0])
        cache = deque()
        cache.append(root)
        next_data_index = 1
        while cache:
            node = cache.popleft()
            if data_split[next_data_index] == '#':
                node.left = None
            else:
                left = TreeNode(data_split[next_data_index])
                cache.append(left)
                node.left = left
            next_data_index += 1

            if data_split[next_data_index] == '#':
                node.right = None
            else:
                right = TreeNode(data_split[next_data_index])
                cache.append(right)
                node.right = right
            next_data_index += 1
        return root

        

# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))

```

#### 428 序列化和反序列化多叉树

```python

from collections import deque


class Node:
    def __init__(self, val):
        self.val = val
        self.children = []


def creat_multi_children_node():
    """创建多叉树实例"""
    # https://zhuanlan.zhihu.com/p/109521420
    root = Node(1)
    root.children.append(Node(3))
    root.children.append(Node(2))
    root.children.append(Node(4))
    root.children[0].children.append(Node(5))
    root.children[0].children.append(Node(6))
    return root


class Solution:

    def serialize(self, root: 'Node') -> str:
        """Encodes a tree to a single string.
        :type root: Node
        :rtype: str

        题解：序列化N叉树，可以直接层次遍历，记录当前值和其孩子节点的个数，来进行序列化
        """
        if not root:
            return ""
        queue = deque([root])
        res = []
        while queue:
            node = queue.popleft()
            res.append(str(node.val))
            res.append(str(len(node.children)))
            for children in node.children:
                queue.append(children)
        data = ",".join(res)
        print('serialize data:', data)
        return data

    def deserialize(self, data: str) -> 'Node':
        """Decodes your encoded data to tree.
        :type data: str
        :rtype: Node
        题解：还是层次遍历来实现，当前节点有N个孩子，我可以消耗N个数据，并把孩子节点和其对应的孩子节点的个数存到cache中，为后续消费提供支持。
        """
        from collections import deque
        if not data:
            return
        cache = deque()
        data_list = data.split(',')
        data_list = [int(_) for _ in data_list]
        root = Node(data_list[0])
        next_data_index = 2
        cache.append([root, data_list[1]])
        while cache:
            cur_node, cur_children_num = cache.popleft()
            # 持续消耗当前层的数据，并更新next_data_index, 每次加2，因为记录了每个点的val和孩子的个数
            for _ in range(cur_children_num):
                tmp, tmp_num = data_list[next_data_index], data_list[next_data_index+1]
                tmp_node = Node(tmp)
                cur_node.children.append(tmp_node)
                cache.append([tmp_node, tmp_num])
                # 每次消耗两个数据
                next_data_index += 2
        print("root:", root)
        return root


if __name__ == '__main__':

    s = Solution()
    root = creat_multi_children_node()
    serialize_data = s.serialize(root)
    print('serialize_data:', serialize_data)
    print(s.deserialize(serialize_data))

```


#### 449 序列化和反序列化搜索二叉树

- 最新方法：先序遍历+二分+递归

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return ''
        cache = []
        cache.append(root)
        res = []
        while cache:
            cur = cache.pop()
            res.append(str(cur.val))
            if cur.right:
                cache.append(cur.right)
            if cur.left:
                cache.append(cur.left)
        res = ','.join(res)
        # print('res:', res)
        return res

    def build_from_preorder(self, values, l, r):
        if l > r:
            return None
        if l == r:
            return TreeNode(values[l])
        cand = l
        k1 = l + 1
        k2 = r
        while k1 <= k2:
            mid = k1 + k2 >> 1
            if values[mid] > values[l]:
                k2 = mid - 1
            elif values[mid] < values[l]:
                cand = mid
                k1 = mid + 1
            else:
                cand = mid
                k1 = mid + 1
        cur = TreeNode(values[l])
        # print('values[l]:', values[l])
        # print('values[cand]:', values[cand])
        # print('left:', values[l+1:cand+1])
        # print('right:', values[cand+1:r+1])
        cur.left = self.build_from_preorder(values, l+1, cand)
        cur.right = self.build_from_preorder(values, cand + 1, r)
        return cur

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        :type data: str
        :rtype: TreeNode
        449. 序列化和反序列化二叉搜索树
输入：root = [2,1,3]
输出：[2,1,3]

题解：
1. 搜索二叉树如果是非平衡树，用常规序列二叉树的方法存会浪费比较多的空间，可以利用先序遍历的方式存，然后利用二叉搜索树的性质区分左右子树，递归构建左右子树即可。
        """
        if not data:
            return None
        values = data.split(',')
        values = [int(_) for _ in values]
        return self.build_from_preorder(values, 0, len(values)-1)

        

# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# tree = ser.serialize(root)
# ans = deser.deserialize(tree)
# return ans

```

- 之前的方法
```python
# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:
    def serialize(self, root):
        """序列化一个二叉搜索树
        Encodes a tree to a single string.
        """
        def postorder(root):
            """后序遍历，左->右->当前"""
            return postorder(root.left) + postorder(root.right) + [root.val] if root else []

        return ' '.join(map(str, postorder(root)))

    def deserialize(self, data):
        """二叉搜索树的反序列化
        Decodes your encoded data to tree.
        1. 二叉树可以通过前序序列或后序序列和中序序列构造
        2. 搜索树的中序遍历是从小到大排列的，前序序列或后序序列相当于我们也知道了中序序列，可以通过排序获得。
        3. 后续遍历进行序列化，然后再 当前->右->左进行解码
        """

        def helper(lower=float('-inf'), upper=float('inf')):
            """反序列化
            1. 先后建当前节点
            2. 尝试一直优先构建右节点，知道不满足大小要求了，构建左节点
            3. 最终返回结果

            #  Example 1: 
            #  Input: root = [2,1,3]
            # Output: [2,1,3]
            """
            if not data or data[-1] < lower or data[-1] > upper:
                return None
            # 当前->右->左
            # 拿出最后的元素
            val = data.pop()
            root = TreeNode(val)
            # 右边应该是在(val, uppper)区间
            root.right = helper(val, upper)
            # 左边应该是在(lower, val)区间
            root.left = helper(lower, val)
            return root

        data = [int(x) for x in data.split(' ') if x]
        return helper()

```



### 3.6 二叉树直径&路径和

#### 543 二叉树的直径-简化版本

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def diameterOfBinaryTree(self, root):
        """树的直径
        :type root: TreeNode
        :rtype: int
        题目：给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过也可能不穿过根结点。
        Solution:
            1. 想办法求每个节点的最大深度
            2. 树的直径就是每个节点左右的最大深度之和，遍历每个节点的时候，记录最大值即可
        """
        result = 0
        def search(node):
            """求每个节点node的最大深度"""
            if node is None:
                return 0
            nonlocal result
            left_depth = search(node.left)
            right_depth = search(node.right)
            cur_depth = max(left_depth, right_depth) + 1
            # 更新直径的最大值
            result = max(result, left_depth + right_depth)
            return cur_depth
        search(root)
        # 返回直径最大值
        return result
```


#### 2246 相邻字符不同的最长路径

```python
# 如果没有相邻节点的限制，那么本题求的就是树的直径上的点的个数，见 1245. 树的直径。

# 考虑用树形 DP 求直径。枚举子树 xx 的所有子树 yy，维护从 xx 出发的最长路径 \textit{maxLen}maxLen，那么可以更新答案为从 yy 出发的最长路径加上 \textit{maxLen}maxLen，再加上 11（边 x-yx−y），即合并从 xx 出发的两条路径。递归结束时返回 \textit{maxLen}maxLen。

# 对于本题的限制，我们可以在从子树 yy 转移过来时，仅考虑从满足 s[y]\ne s[x]s[y] 
# 
# ​
#  =s[x] 的子树 yy 转移过来，所以对上述做法加个 if 判断就行了。

# 由于本题求的是点的个数，所以答案为最长路径的长度加一。

# Python3GoC++Java

class Solution:
    def longestPath(self, parent: List[int], s: str) -> int:
        """输入一颗树，求满足相邻节点不相等的最大直径
        Example:
            输入：parent = [-1,0,0,1,1,2], s = "abacbe"
            输出：3
        Solution:
            1. 所有孩子中，top1_depth+top2_depth就是最大长度
            2. 需要加上限制条件：相邻节点不相等
        """
        n = len(parent)
        g = [[] for _ in range(n)]
        for i in range(1, n):
            g[parent[i]].append(i)

        ans = 0
        def dfs(x: int) -> int:
            """记录的是节点的最大深度"""
            nonlocal ans
            max_depth = 0
            for child in g[x]:
                # 当前的最大深度为y孩子的最大深度+1
                cur_depth = dfs(child) + 1
                if s[x] != s[child]:
                    # 满足条件时, top1_depth + top2_depth就是最大长度
                    # 因为max_depth一直在更新，可以确保最终最大值包括top1_depth+top2_depth
                    ans = max(ans, max_depth + cur_depth)
                    # 更新top1_depth
                    max_depth = max(max_depth, cur_depth)
            return max_depth
        dfs(0)
        return ans + 1

# 作者：endlesscheng
# 链接：https://leetcode-cn.com/problems/longest-path-with-different-adjacent-characters/solution/by-endlesscheng-92fw/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```


- 方法2

```python
class Solution:
    def longestPath(self, parent: List[int], s: str) -> int:
        n = len(parent)
        if n == 1:
            return 1
        g = [[] for _ in range(n)]
        for i in range(1, n):
            g[parent[i]].append(i)
        
        self.ans = 0
        def dfs(x: int) -> int:
            """记录的是节点的最大深度"""
            child_results = []
            for child in g[x]:
                # 当前的最大深度为y孩子的最大深度+1
                child_depth = dfs(child)
                if s[x] != s[child]:
                    child_results.append(child_depth)
                else:
                    child_results.append(0)
            # print('x:', x, 'child_results:', child_results)
            if not child_results:
                return 1
            child_results.sort()
            if len(child_results) == 1:
                self.ans = max(self.ans, child_results[0] + 1)
            else:
                self.ans = max(self.ans, child_results[-1] + child_results[-2] + 1)
            return child_results[-1] + 1
        dfs(0)
        return self.ans

```

### 3.7 后继者&祖先

#### 面试题 04.06. 后继者


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def inorderSuccessor(self, root, p):
        """
        :type root: TreeNode
        :type p: TreeNode
        :rtype: TreeNode
设计一个算法，找出二叉搜索树中指定节点的“下一个”节点（也即中序后继）。

如果指定节点没有对应的“下一个”节点，则返回null。

输入: root = [2,1,3], p = 1

  2
 / \
1   3

输出: 2
        方法1：中序遍历，查找前一个节点是否为p即可
        方法2：
        1. 如果存在右节点，那么直接从当前节点往右边找
        2. 如果不存在右节点，可以从根节点往下找
        """

        if  p.right is not None:
            p = p.right
            while p.left is not None:
                p = p.left
            return p
        # 没有右节点，需要从根节点往下找
        successor = None
        node = root
        while node:
            if node.val > p.val:
                # 缓存一个比较大的值
                successor = node
                node = node.left
            else:
                node = node.right
        return successor

# 作者：LeetCode-Solution
# 链接：https://leetcode.cn/problems/successor-lcci/solution/hou-ji-zhe-by-leetcode-solution-6hgc/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
        

        

```


#### 236. Lowest Common Ancestor of a Binary Tree - 迭代方法


```python

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        
        Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
        Output: 3
        Explanation: The LCA of nodes 5 and 1 is 3.
        """
        
        cache = []
        cache.append([root, [root]])
        left_path = None
        while cache:
            cur, cur_path = cache.pop()
            # print("cur:", cur.val)
            # print("cur_path:", cur_path)
            if cur.val == p.val:
                left_path = cur_path
                break
            if cur.right:
                cache.append([cur.right, cur_path + [cur.right]])
            if cur.left:
                cache.append([cur.left, cur_path + [cur.left]])
        right_path = None
        cache = [[root, [root]]]
        while cache:
            cur, cur_path = cache.pop()
            if cur.val == q.val:
                right_path = cur_path
                break
            if cur.right:
                cache.append([cur.right, cur_path + [cur.right]])
            if cur.left:
                cache.append([cur.left, cur_path + [cur.left]])
                
        if left_path is not None and right_path is not None:
            l = 0
            result = None
            while l < len(left_path) and l < len(right_path) and left_path[l].val == right_path[l].val:
                result = left_path[l]
                l += 1
            return result
        return None
```


#### 236. Lowest Common Ancestor of a Binary Tree - 递归方法

```python

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        
        Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
        Output: 3
        Explanation: The LCA of nodes 5 and 1 is 3.

        题解：https://mp.weixin.qq.com/s/njl6nuid0aalZdH5tuDpqQ
        """
        
        if root is None:
            return None
        # 代表该节点下找到了p或者q
        if root.val == p.val or root.val == q.val:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        # 如果要从左右两个节点来找数字，那么返回root
        if left is not None and right is not None:
            return root
        return left if left is not None else right

```

### 3.8 树形dp


#### 437 树的路径和III

```python
class Solution(object):
    def pathSum(self, root, targetSum):
        """树上从上到下的节点中，路径和为targetSum的个数，可以不是从头结点或者叶节点为起始点
        :type root: TreeNode
        :type targetSum: int
        :rtype: int
        #  Example 1:
        #
        #
        # Input: root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
        # Output: 3
        # Explanation: The paths that sum to 8 are shown.

        Solution:
            1. 前缀和频次+树的遍历
        """

        self.target_sum_cnt = 0

        def search(cur_node, prefix_sum, prefix_sum_cnt_dict):
            """搜索每一个节点
            Args:
                cur_node(Node):当前节点
                prefix_sum(int): 当前节点之前的路径上的前缀和
                prefix_sum_cnt_dict(dict):当前节点之前的路径上的前缀和的频次
            """

            # nonlocal target_sum_cnt
            if cur_node is None:
                return
            cur_sum = prefix_sum + cur_node.val
            if cur_sum == targetSum:
                self.target_sum_cnt += 1
            diff = cur_sum - targetSum
            if prefix_sum_cnt_dict[diff] > 0:
                self.target_sum_cnt += prefix_sum_cnt_dict[diff]
            # print('cur_node:', cur_node.val, 'cur_sum:', cur_sum, 'prefix_sum:', prefix_sum, 'diff:', diff, 'diff cnt:', prefix_sum_cnt_dict[diff])
            prefix_sum_cnt_dict[cur_sum] += 1
            search(cur_node.left, cur_sum, prefix_sum_cnt_dict)

            search(cur_node.right, cur_sum, prefix_sum_cnt_dict)
            prefix_sum_cnt_dict[cur_sum] -= 1

        from collections import defaultdict
        prefix_sum_cnt_dict = defaultdict(int)
        search(root, 0, prefix_sum_cnt_dict)

        return self.target_sum_cnt

```

#### 1371 找到二进制树中，节点和最大的二进制搜索树

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxSumBST(self, root: Optional[TreeNode]) -> int:
        """find the maxsum bst in an binaray tree
        example:
            Input: root = [1,4,3,2,4,2,5,null,null,null,null,null,null,4,6]
            Output: 20
        solution:
            use postOrder traversal to check if child is valide, and then check
            cur node if valid, meanwhile, we can get the sum after postOrder traversal
        """


        self.max_binary_search_node_sum = 0
        def isValidBinarySearchTree(node):
            if node is None:
                # for an null child, we need use child.maxval and child.minval to check
                # if fathter is valid, it's min_val = float('inf'), and 
                # max_val = float('-inf'), so the child.max_val < fater.val < child.min_val
                # can be true
                # reuslt = isValidBinarySearchTree, node's sum, cur.min, cur.max
                return True, 0, float('inf'), float('-inf')
            
            left_valid, left_sum, left_l, left_r = isValidBinarySearchTree(node.left)
            right_valid, right_sum, right_l, right_r = isValidBinarySearchTree(node.right)
            if left_r < node.val < right_l and left_valid and right_valid:
                cur_sum = left_sum + right_sum + node.val
                if cur_sum > self.max_binary_search_node_sum:
                    self.max_binary_search_node_sum = cur_sum
                # print('node:',node.val,'valid:',True, 'sum:', cur_sum, 'cur.min:', min(left_l, node.val), 'cur.max:', max(right_r, node.val))
                # reuslt = isValidBinarySearchTree, node's sum, cur.min, cur.max                
                return True, cur_sum, min(left_l, node.val), max(right_r, node.val)
            else:
                return False, -1, -1, -1
        isValidBinarySearchTree(root)
        return self.max_binary_search_node_sum

```




### 3.9 其他

#### 114 flatten二叉树


```python

给你二叉树的根结点 root ，请你将它展开为一个单链表：
展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
展开后的单链表应该与二叉树 先序遍历 顺序相同。
输入：root = [1,2,5,3,4,null,6]
   1
 2  5
3 4 null 6

=>

1->2->3->4->5->6

输出：[1,null,2,null,3,null,4,null,5,null,6]

题解：先序遍历

```
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def flatten(self, root):
        """将二叉树转化成链表，
        :type root: TreeNode
        :rtype: None Do not return anything, modify root in-place instead.
        方法：直接先序遍历，记录pre结果，完成flatten
        """
        if root is None:
            return
        cache = [root]
        pre = TreeNode(-1)
        head = pre
        while cache:
            cur = cache.pop()
            # print('cur:', cur.val)
            if cur.right is not None:
                cache.append(cur.right)
            if cur.left is not None:
                cache.append(cur.left)
            cur.left = None
            pre.right = cur
            pre = cur
        return head.right

```

#### 331 验证二叉树的前序序列化

```python
给定一串以逗号分隔的序列，验证它是否是正确的二叉树的前序序列化。编写一个在不重构树的条件下的可行算法。
输入: preorder = "9,3,4,#,#,1,#,#,2,#,6,#,#"
输出: true
```


```python
class Solution(object):
    def isValidSerialization(self, preorder):
        """给出先序遍历结果，空节点由#表示，问是否是个有效的二叉树。
        :type preorder: str, 先序遍历结果
        :rtype: bool
        example:
            #  Input: preorder = "9,3,4,#,#,1,#,#,2,#,6,#,#"
            # Output: true
               9
           3      2
        4   1   #   6
       # # # #    #   #

       方法： 6 # # 代表一个子节点，利用栈一步一步消除子节点，然后判断是否为空
        """
        preorder = preorder.split(',')
        cache = []
        n = len(preorder)
        cur = 0
        while cur < n:
            if not cache:
                cache.append(preorder[cur])
            else:
                if preorder[cur] != '#':
                    cache.append(preorder[cur])
                else:
                    cache.append('#')
                    while True:
                        if len(cache) >= 3 and cache[-1] == '#' and cache[-2] == "#" and cache[-3] != '#':
                            cache.pop()
                            cache.pop()
                            cache.pop()
                            cache.append('#')
                        else:
                            break
            cur += 1
        return cache == ["#"]

```

#### 341 将嵌套的list转成flat的

```python
将嵌套的list转成flat的

会用如下代码检测：
initialize iterator with nestedList
res = []
while iterator.hasNext()
    append iterator.next() to the end of res
return res


输入：nestedList = [[1,1],2,[1,1]]
输出：[1,1,2,1,1]
解释：通过重复调用 next 直到 hasNext 返回 false，next 返回的元素的顺序应该是: [1,1,2,1,1]。
```

```python

class NestedIterator(object):
    """将一个嵌套的list转成flat的
    方法：利用栈来处理，直到最前面的元素处理好了，就不放入栈里面了
    # Input: nestedList = [[1,1],2,[1,1]]
    # Output: [1,1,2,1,1]
    """
    def __init__(self, nestedList):
        # 先倒着放进stack里面
        self.stack = []
        for i in range(len(nestedList) - 1, -1, -1):
            self.stack.append(nestedList[i])

    def next(self):
        # 取出已经处理好的int值
        cur = self.stack.pop()
        return cur.getInteger()

    def hasNext(self):
        """处理好嵌套的数据"""
        while self.stack:
            # 如果是整数类型，直接返回true
            cur = self.stack[-1]
            if cur.isInteger():
                return True
            # 否则把最后一个元素拿出来处理
            self.stack.pop()
            # 利用栈，把最前面的元素放到栈顶，知道它为整数了才退出，相当于利用栈来flat list
            for i in range(len(cur.getList()) - 1, -1, -1):
                self.stack.append(cur.getList()[i])
        return False
```
