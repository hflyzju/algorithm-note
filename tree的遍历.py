# https://zhuanlan.zhihu.com/p/98406357
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


root = TreeNode(1)
b = TreeNode(2)
c = TreeNode(3)
d = TreeNode(4)
e = TreeNode(5)
f = TreeNode(6)
g = TreeNode(7)

root.left = b
root.right = c
b.left = d
b.right = e
c.left = f
c.right = g

print('='*10)
print('先序遍历')
## 先序遍历
### 递归

"""专门存着路径"""
class PreOrderTraverse(object):
    """先序遍历"""
    def __init__(self):
        self.path = []
        pass

    def preOrderTraverse(self, node):
        if node:
            self.path.append(node.val)
            self.preOrderTraverse(node.left)
            self.preOrderTraverse(node.right)

    def run(self, root):
        self.preOrderTraverse(root)
        print(self.path)

P = PreOrderTraverse()
P.run(root)


"""递归返回路径"""
def preOrderTraverse(node):
    if node:
        left_vals, right_vals = [], []
        if node.left:
            left_vals = preOrderTraverse(node.left)
        if node.right:
            right_vals = preOrderTraverse(node.right)
        return [node.val] + left_vals + right_vals


print(preOrderTraverse(root))

"""
        1
    2       3
4     5    6 7

先序遍历 : 1->2->4->5->3->6->7
"""

"""非递归先序遍历1"""
def preOrderTraverseNoRecursion(root):
    """
    根据前序遍历访问的顺序，优先访问根结点，然后再分别访问左孩子和右孩子。即对于任一结点，其可看做是根结点，因此可以直接访问，
    访问完之后，若其左孩子不为空，按相同规则访问它的左子树；当访问其左子树时，再访问它的右子树。因此其处理过程如下：
    https://www.cnblogs.com/dolphin0520/archive/2011/08/25/2153720.html
    :param root:
    :return:
    """

    cur = root
    cache = []
    path = []
    while cur or cache:
        # 一路向左, 并把路径上遇到的node压入栈
        while cur:
            path.append(cur.val)
            cache.append(cur)
            cur = cur.left

        # 从栈里面把最前面的左子树拿出来, 然后从这个节点的右子树开始看, 把任意节点看成根节点
        if cache:
            cur = cache.pop()
            cur = cur.right
    return path

print(preOrderTraverseNoRecursion(root))


""""""

"""非递归先序遍历2"""
def preOrderTraverseNoRecursion2(root):
    """
    :param root:
    :return:
    https://blog.csdn.net/m0_37324740/article/details/82763901
    """
    path = []
    cache = [root]
    cur = root
    while cache:
        path.append(cur.val)
        # 先把右节点放进去
        if cur.right:
            cache.append(cur.right)
        # 再把做节点放进去
        if cur.left:
            cache.append(cur.left)
        # 拿到他孩子的 左节点 --> 右节点, ...
        cur = cache.pop()
    return path

print(preOrderTraverseNoRecursion2(root))



## 中序遍历
"""
        1
    2       3
4     5    6 7

中序遍历 : 4->2->5->1->6->3->7
"""
print('='*10)
print('中序遍历')

### 递归
def inOrderTraverse(node):
    if node:
        left_paths = inOrderTraverse(node.left)
        right_paths = inOrderTraverse(node.right)
        return left_paths + [node.val] + right_paths
    return []
print(inOrderTraverse(root))


### 非递归中序
def inOrderTraverseNoRecursion(node):
    """
    1.左-->中-->右
    根据中序遍历的顺序，对于任一结点，优先访问其左孩子，而左孩子结点又可以看做一根结点，然后继续访问其左孩子结点，
    直到遇到左孩子结点为空的结点才进行访问，然后按相同的规则访问其右子树。因此其处理过程如下：
    https://www.cnblogs.com/dolphin0520/archive/2011/08/25/2153720.html
    :param node:
    :return:
    """
    cur = node
    cache = []
    path = []
    while cur or cache:
        while cur:
            cache.append(cur)
            cur = cur.left
        if cache:
            cur = cache.pop()
            path.append(cur.val)
            cur = cur.right
    return path
print(inOrderTraverseNoRecursion(root))

### 非递归中序
def inOrderTraverseNoRecursion2(node):
    """
    1.左-->中-->右
    两种的思想应该是一样的, 都是先一路向左, 直到没有左了, 把cache里面的值当作当前节点, 然后输出, 并把当前节点更新为其右节点
    :param node:
    :return:
    """
    cache = []
    path = []
    cur = node
    while cache or cur:
        if cur:
            cache.append(cur)
            cur = cur.left
        else:
            cur = cache.pop()
            path.append(cur.val)
            cur = cur.right
    return path
print(inOrderTraverseNoRecursion2(root))

"""
        1
    2       3
4     5    6 7

中序遍历 : 4->5->2->6->7->3->1
"""
## 后序遍历
print('='*10)
print('后序遍历')


def posOrderTraverse(node):
    """左-->右-->中"""
    if node:
        left_path = posOrderTraverse(node.left)
        right_path = posOrderTraverse(node.right)
        return left_path + right_path + [node.val]
    return []
print(posOrderTraverse(root))

def posOrderTraverseNoRecursion(node):
    """左-->右-->中"""
    if root is None:
        return False
    stack1 = []
    stack2 = []
    stack1.append(root)
    while stack1:   # 找出后序遍历的逆序，存放在 stack2中  左->右->中 => 中->右->左
        node = stack1.pop()
        if node.left is not None:
            stack1.append(node.left)
        if node.right is not None:
            stack1.append(node.right)
        stack2.append(node)
    while stack2:  # 将 stack2中的元素出栈，即是后序遍历序列
        print(stack2.pop().value, end=' ')









