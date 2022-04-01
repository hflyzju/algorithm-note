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


#### 95 构建所有二叉树

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


#### 96. 1-n可能二叉树的数量-从上到下dp

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

#### 96 1-n可能二叉树的数量-从下到上dp

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
        for i in range(2, n + 1):
            for j in range(1, i + 1):
                dp[i] += dp[j-1] * dp[i - j]
        return dp[n]
```


#### 98 有效的二叉树-非递归中序遍历


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
        def search(pre_nodes, ino_nodes):
            if not pre_nodes or not ino_nodes:
                return None
            # 拿到当前节点
            cur_val = pre_nodes[0]
            # 找到当前节点在中序遍历的位置
            ino_h_index = ino_nodes.index(cur_val)
            cur = TreeNode(cur_val)
            # 遍历左孩子节点
            childl = search(pre_nodes[1:ino_h_index+1], ino_nodes[:ino_h_index])
            # 遍历右孩子节点
            childr = search(pre_nodes[ino_h_index+1:], ino_nodes[ino_h_index+1:])
            cur.left = childl
            cur.right = childr
            return cur
        return search(preorder, inorder)

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
        def search(ino, pos):
            if not ino or not pos:
                return None
            root_val = pos[-1]
            root_val_ino_index = ino.index(root_val)
            root = TreeNode(root_val)
            root.left = search(ino[:root_val_ino_index], pos[:root_val_ino_index])
            root.right = search(ino[root_val_ino_index+1:], pos[root_val_ino_index:-1])
            return root
        return search(inorder, postorder)
```


#### 331 验证是否为二叉树的先序遍历


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