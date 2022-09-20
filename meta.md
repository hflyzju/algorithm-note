
# META ML RS过经及timeline 2022-1-8

https://www.1point3acres.com/bbs/thread-837096-1-1.html


## 店面

### 题目1:236. Lowest Common Ancestor of a Binary Tree

#### 1.1 递归思想

```python
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return root
        if root == p or root == q:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        if left:
            return left
        return right
```
#### 1.2 用栈记录path

```python
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        stack = []
        self.p_path = None
        self.q_path = None
        def search(node):
            if not node:
                return
            stack.append(node) # 进入节点
            if node == p:
                self.p_path = stack[:]
            if node == q:
                self.q_path = stack[:]
            search(node.left)
            search(node.right)
            stack.pop() # 出节点
        search(root)
        i = 0
        lowset_root = root
        while i < len(self.p_path) and i < len(self.q_path) and self.p_path[i].val == self.q_path[i].val:
            lowset_root = self.p_path[i]
            i += 1
        return lowset_root
```

### 题目2: 523. Continuous Subarray Sum

#### 题解：
1. (m + k * n) % k = m % k， 如果当前和的余数存在过，那么中间就代表加了n*k，所以就有连续的和为n*k
2. 注意0是任何数的倍数，例如[0, 0]就为True，但是长度要大于2，所以[0]不是。
3. 注意一些badcase，例如：[6, 0，0]也是一个结果
4. multiple倍数，0是任何数的倍数

```python
class Solution(object):
    def checkSubarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        (m + k * n) % k = m % k
        """
        n = len(nums)
        if n <= 1:
            return False
        d = dict()
        d[0] = -1 # corner case1 [2,4] 6,  corner case2 [0, 1] 6
        s = 0
        for i, num in enumerate(nums):
            s += num
            v = s % k
            if v in d:
                if i - d[v] > 1: #corner case2 [0, 1] 6
                    return True
            else:
                # 每次记录第一个出现的就行，这样就可以保证长度大于2，所以不存在的时候才更新进d
                d[v] = i
        return False
```



## on-site

### Round1(coding)：三题

#### 339. Nested List Weight Sum

- problem

```python
You are given a nested list of integers nestedList. Each element is either an integer or a list whose elements may also be integers or other lists.

The depth of an integer is the number of lists that it is inside of. For example, the nested list [1,[2,2],[[3],2],1] has each integer's value set to its depth.

Return the sum of each integer in nestedList multiplied by its depth.

 

Example 1:


Input: nestedList = [[1,1],2,[1,1]]
Output: 10
Explanation: Four 1's at depth 2, one 2 at depth 1. 1*2 + 1*2 + 2*1 + 1*2 + 1*2 = 10.
Example 2:


Input: nestedList = [1,[4,[6]]]
Output: 27
Explanation: One 1 at depth 1, one 4 at depth 2, and one 6 at depth 3. 1*1 + 4*2 + 6*3 = 27.
Example 3:

Input: nestedList = [0]
Output: 0
 

Constraints:

1 <= nestedList.length <= 50
The values of the integers in the nested list is in the range [-100, 100].
The maximum depth of any integer is less than or equal to 50.

```

- code
```python
# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
#class NestedInteger(object):
#    def __init__(self, value=None):
#        """
#        If value is not specified, initializes an empty list.
#        Otherwise initializes a single integer equal to value.
#        """
#
#    def isInteger(self):
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        :rtype bool
#        """
#
#    def add(self, elem):
#        """
#        Set this NestedInteger to hold a nested list and adds a nested integer elem to it.
#        :rtype void
#        """
#
#    def setInteger(self, value):
#        """
#        Set this NestedInteger to hold a single integer equal to value.
#        :rtype void
#        """
#
#    def getInteger(self):
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        :rtype int
#        """
#
#    def getList(self):
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        :rtype List[NestedInteger]
#        """
class Solution(object):
    def depthSum(self, nestedList):
        """
        :type nestedList: List[NestedInteger]
        :rtype: int
        """
        self.depth = 1
        def search(curNestedList):
            s = 0
            for i in range(len(curNestedList)):
                val = curNestedList[i]
                if val.isInteger():
                    s += val.getInteger() * self.depth
                else:
                    self.depth += 1
                    s += search(val.getList())
                    self.depth -= 1
            return s

        return search(nestedList)

```

#### 133. Clone Graph

- 题目
```
Given a reference of a node in a connected undirected graph.

Return a deep copy (clone) of the graph.

Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.

class Node {
    public int val;
    public List<Node> neighbors;
}
 

Test case format:

For simplicity, each node's value is the same as the node's index (1-indexed). For example, the first node with val == 1, the second node with val == 2, and so on. The graph is represented in the test case using an adjacency list.

An adjacency list is a collection of unordered lists used to represent a finite graph. Each list describes the set of neighbors of a node in the graph.

The given node will always be the first node with val = 1. You must return the copy of the given node as a reference to the cloned graph.

 

Example 1:


Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
Output: [[2,4],[1,3],[2,4],[1,3]]
Explanation: There are 4 nodes in the graph.
1st node (val = 1)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
2nd node (val = 2)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
3rd node (val = 3)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
4th node (val = 4)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
Example 2:


Input: adjList = [[]]
Output: [[]]
Explanation: Note that the input contains one empty list. The graph consists of only one node with val = 1 and it does not have any neighbors.
Example 3:

Input: adjList = []
Output: []
Explanation: This an empty graph, it does not have any nodes.
 

Constraints:

The number of nodes in the graph is in the range [0, 100].
1 <= Node.val <= 100
Node.val is unique for each node.
There are no repeated edges and no self-loops in the graph.
The Graph is connected and all nodes can be visited starting from the given node.
```

- dfs题解

```python
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

class Solution(object):
    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        
        self.node_to_clone = dict()
        def search(node):
            if not node:
                return None
            if node in self.node_to_clone:
                return self.node_to_clone[node]
            
            clone = Node(node.val)
            self.node_to_clone[node] = clone
            for child in node.neighbors:
                clone.neighbors.append(search(child))
            return clone
        return search(node)

```

- bfs题解

```python
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

class Solution(object):
    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        if not node:
            return node
        copy = Node(node.val)
        d = deque()
        d.append([node, copy])
        visited = dict()
        visited[node] = copy
        while d:
            cur, copy = d.popleft()
            for child in cur.neighbors:
                if child not in visited:
                    child_copy = Node(child.val)
                    visited[child] = child_copy
                    d.append([child, child_copy])
                copy.neighbors.append(visited[child])
        return visited[node]

```


#### 227. Basic Calculator II

```
Given a string s which represents an expression, evaluate this expression and return its value. 

The integer division should truncate toward zero.

You may assume that the given expression is always valid. All intermediate results will be in the range of [-231, 231 - 1].

Note: You are not allowed to use any built-in function which evaluates strings as mathematical expressions, such as eval().

Example 1:

Input: s = "3+2*2"
Output: 7
Example 2:

Input: s = " 3/2 "
Output: 1
Example 3:

Input: s = " 3+5 / 2 "
Output: 5
```

- 用stack解决
```python
class Solution(object):
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        pre_mark = '+'
        val = 0
        res = []
        for i in range(len(s)):
            if s[i].isdigit():
                val = val * 10 + int(s[i])
            if (not s[i].isdigit() and s[i] != ' ') or i == len(s) - 1 :
                if pre_mark == '+':
                    res.append(val)
                elif pre_mark == '-':
                    res.append(-val)
                elif pre_mark == '*':
                    res[-1] *= val
                elif pre_mark == '/':
                    val_to_fix = 0
                    if res[-1] < 0 and (-res[-1]) % val > 0:
                        val_to_fix = 1
                    res[-1] /= val
                    res[-1] += val_to_fix
                pre_mark = s[i]
                val = 0
        return sum(res)
        
```

- 不用stack，因为只需要看最近的两个元素即可

```python
class Solution(object):
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        
        Input: s = "3+2*2"
        Output: 7
        
        l1=3
        l2=2*2
        
        2
        
        """
        pre_mark = '+'
        val = 0
        
        res = 0
        l1 = 0
        l2 = 0
        for i in range(len(s)):
            if s[i].isdigit():
                val = val * 10 + int(s[i])
            if (not s[i].isdigit() and s[i] != ' ') or i == len(s) - 1 :
                if pre_mark == '+':
                    res += l1
                    l1 = l2
                    l2 = val
                elif pre_mark == '-':
                    res += l1
                    l1 = l2
                    l2 = -val
                elif pre_mark == '*':
                    l2 *= val
                elif pre_mark == '/':
                    val_to_fix = 0
                    if l2 < 0 and (-l2) % val > 0:
                        val_to_fix = 1
                    l2 /= val
                    l2 += val_to_fix
                pre_mark = s[i]
                val = 0
        return res + l1 + l2
```
### Round2(coding)：两题

#### 65. Valid Number

```

A valid number can be split up into these components (in order):

A decimal number or an integer.
(Optional) An 'e' or 'E', followed by an integer.
A decimal number can be split up into these components (in order):

(Optional) A sign character (either '+' or '-').
One of the following formats:
One or more digits, followed by a dot '.'.
One or more digits, followed by a dot '.', followed by one or more digits.
A dot '.', followed by one or more digits.
An integer can be split up into these components (in order):

(Optional) A sign character (either '+' or '-').
One or more digits.
For example, all the following are valid numbers: ["2", "0089", "-0.1", "+3.14", "4.", "-.9", "2e10", "-90E3", "3e+7", "+6e-1", "53.5e93", "-123.456e789"], while the following are not valid numbers: ["abc", "1a", "1e", "e3", "99e2.5", "--6", "-+3", "95a54e53"].

Given a string s, return true if s is a valid number.

 

Example 1:

Input: s = "0"
Output: true
Example 2:

Input: s = "e"
Output: false
Example 3:

Input: s = "."
Output: false
```

```python
class Solution(object):

    def isValidFloat(self, val):
        val_split = val.split('.')
        if len(val_split) != 2:
            return False
        val_split = [_ for _ in val_split if _]
        if len(val_split) == 0:
            return False
        for tmp in val_split:
            if not tmp.isdigit():
                return False
        return True

    def isValidInt(self, val):
        # print('val:', val)
        if not val:
            # print('False')
            return False
        n = len(val)
        # print('val.isdigit():', val.isdigit())
        return val.isdigit()

        """
        "abc"
        "1a"
        "1e"
        "e3"
        "99e2.5"
        "--6"
        "-+3"
        "95a54e53"
        """
    def isNumber(self, s):
        """
        :type s: str
        :rtype: bool
        """

        """
        小数或者整数 + e或者E再加整数

        """
        s = s.replace(' ', '')
        if not s:
            return False
        if s == '.':
            return False
        s = s.lower()
        if s.find('e') == -1:
            if s[0] in ['+', '-']:
                return self.isValidInt(s[1:]) or self.isValidFloat(s[1:])
            else:
                return self.isValidInt(s) or self.isValidFloat(s)
        else:
            ls = s.split('e')
            if len(ls) != 2:
                return False
            ls = [_ for _ in ls if _]
            if len(ls) != 2:
                return False
            if ls[1][0] in ['+', '-']:
                if not self.isValidInt(ls[1][1:]):
                    return False
            else:
                if not self.isValidInt(ls[1]):
                    return False
            if ls[0][0] in ['+', '-']:
                if ls[0][1:]:
                    return self.isValidInt(ls[0][1:]) or self.isValidFloat(ls[0][1:])
                else:
                    return False
            else:
                return self.isValidInt(ls[0]) or self.isValidFloat(ls[0])

```

#### 670. Maximum Swap

```
You are given an integer num. You can swap two digits at most once to get the maximum valued number.

Return the maximum valued number you can get.

 

Example 1:

Input: num = 2736
Output: 7236
Explanation: Swap the number 2 and the number 7.
Example 2:

Input: num = 9973
Output: 9973
Explanation: No swap.

```

```python
class Solution(object):
    def maximumSwap(self, num):
        """
        :type num: int
        :rtype: int
        """

        """
        27

        max_val_index=1
        max_val = 7
        left_index=2
             
           l  
        [2,7,3,6]

        题解：从右到左遍历，找到最左边是否存在一个数，小于右边的最大值，如果有，则把右边的最大值与最左边的这个值替换。
        """
        cache = []
        num = list(str(num))
        n = len(num)
        left_index_to_swap_index = dict()
        max_val_index = n - 1
        max_val = num[n-1]
        left_index = -1
        for i in range(n-2, -1, -1):
            if num[i] > max_val:
                max_val = num[i]
                max_val_index = i
            elif num[i] < max_val:
                left_index = i
                left_index_to_swap_index[left_index] = max_val_index
            # print('max_val_index:', max_val_index)
            # print('left_index:', left_index)
            # print("left_index_to_swap_index:", left_index_to_swap_index)
        
        if left_index != -1 and left_index < left_index_to_swap_index[left_index]:
            swap_index = left_index_to_swap_index[left_index]
            num[left_index], num[swap_index] = num[swap_index], num[left_index]
        
        return int(''.join(num))

```

# 其他


#### 560 子数组和为k的个数

```python
class Solution(object):
    def subarraySum(self, nums, k):
        """子数组和为k的个数
        :type nums: List[int]
        :type k: int
        :rtype: int
        Example:
            #  Input: nums = [1,1,1], k = 2
            # Output: 2
        Solution:
            前缀和+哈希
        """
        from collections import defaultdict
        # [1,1,1]
        # 1
        # cur_sum=3
        # pre_sum_cnt=[1:1, 2:1]
        # diff = 3-2=1
        # pre_sum = 1
        # cur_sum - pre_sum = k
        # pre_sum = cur_sum - k
        pre_sum_cnt = defaultdict(int)
        pre_sum_cnt[0] = 1
        pre_sum = 0
        cnt = 0
        for num in nums:
            cur_sum = pre_sum + num
            diff = cur_sum - k
            cnt += pre_sum_cnt[diff]
            pre_sum_cnt[cur_sum] += 1
            pre_sum = cur_sum
        return cnt

```


#### 791 将s中的字母按order的顺序进行排序

```python
class Solution(object):
    def customSortString(self, order, s):
        """将s中的字母按order的顺序进行排序
        面经地址：https://www.1point3acres.com/bbs/thread-650769-1-1.html
        :type order: str
        :type s: str
        :rtype: str
        Example:
            # Input: order = "cba", s = "abcd"
            # Output: "cbad"

            # Input: order = "cba", s = "abad"
            # Output: "baad" (要注意s中字符重复的问题)

        Solution:
            hash表
        """
        from collections import defaultdict
        letter_cnt = defaultdict(int)
        for letter in s:
            letter_cnt[letter] += 1
        order_letter_set = set()
        for letter in order:
            order_letter_set.add(letter)

        res = []
        for letter in order:
            if letter in letter_cnt:
                while letter_cnt[letter] > 0:
                    res.append(letter)
                    letter_cnt[letter] -= 1

        for letter in s:
            if letter not in order_letter_set:
                res.append(letter)

        return ''.join(res)

```


## 其他资料

1. https://leetcode.com/problem-list/top-facebook-questions/
2. https://www.glassdoor.sg/Interview/Meta-Machine-Learning-Engineer-Interview-Questions-EI_IE40772.0,4_KO5,30.htm