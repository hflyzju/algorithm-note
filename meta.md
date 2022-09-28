# 总结

## 面筋总结
|面试id|  面试类型 | 时间  | 题目 | 类型 | 难度 | 题解 |
| ---- |  ----  | ----  | --- | --- | --- | --- |
| 1.MLE一条龙新鲜面经 |  店面-题目1  |2022.07| 791：Custom Sort String: 让s按order排序输出 | 哈希+排序 | 中等 | 先统计需要排序的字符串，然后按order输出，注意不存在order中的字符 |
| 1.MLE一条龙新鲜面经 |  店面-题目2  |2022.07| 560. Subarray Sum Equals K：子序列和为k的个数，注意数字可能位负数  | 前缀和 | 中等偏难 |统计每个前缀和出现的次数，统计cur_pre_sum - target的次数即可，注意pre_sum_cnt[0]=1, 解决[1,1] k=2 => 1这种问题 |
| 1.MLE一条龙新鲜面经 |  on-site1-Coding 1  |2022.07| 670. Maximum Swap：返回只可以swap一次后最大的数| 数组 | 中等 |从后往前遍历并记录最大值，对每个位置，与替换最大值替换可以取得该位置swap后的最大值，与最前面一个位置swap可以取得全局最大值|
| 1.MLE一条龙新鲜面经 |  on-site1-Coding 2  |2022.07| 394. Decode String：将例如多层嵌套的s=3[a3[c]]进行解码|栈| 中等偏难 |注意只会出现k[]的形式，所以变简单一点，用pre_res记录之前累积的字符串，cnt记录括号前的数字, res记录当前的字符，遇到]可以更新res=pre_res + cnt * res， 遇到[可以将前面的pre_res和数字cnt压入栈|
| 1.MLE一条龙新鲜面经 |  on-site1-Coding 3  |2022.07| 543. diameter of the tree： 树的直径|树|中等|后续遍历+每次统计树的深度+利用后续遍历计算包含当前节点的直径|
| 2. META ML RS过经及timeline |  店面-Coding 1  |2022.01| Lowest Common Ancestor of a Binary Tree： 两个节点的最小公共祖先|树|中等|方法1: 递归：当前节点的输出可以由左孩子或者右孩子能返回结果来定，如果左孩子或者右孩子都不为空，那么返回当前节点。否则返回两者之一。方法2: 栈：跟踪跟节点到当前节点的的path(先序遍历入栈，后续遍历出栈可)，然后比较两个节点的path，找出最低的ancestor|
| 2. META ML RS过经及timeline |  店面-Coding 2  |2022.01| 523. Continuous Subarray Sum：是否存在长度大于2的连续和为k的子数组 |数组+数学|中等通过率30%|前缀和对k取余数，如果余数mod相等，那么代表中间的那段数为k的倍数，要注意[1,1] k=2这种情况，可以另pre_mod_index[0]=-1|
| 2. META ML RS过经及timeline |  on-site1-Coding 1  |2022.01| 339. Nested List Weight Sum：多重嵌套数组和|栈|中等|递归：先序遍历depth+1，后续遍历depth-1，这样输出和|
| 2. META ML RS过经及timeline |  on-site1-Coding 2  |2022.01| 133. Clone Graph：克隆无向图|dfs或者bfs|中等|用字典记录每个节点到其copy节点的映射，然后用bfs求解，比较方便|
| 2. META ML RS过经及timeline |  on-site1-Coding 3  |2022.01| Basic Calculator II: 计算无括号的加减乘初的结果|栈 or 优化版本|中等|1. 只需要记录ans(上个位置之前的结果)，pre_val(上一个位置结果)，cur_val(当前位置结果)这三个数，可以将空间复杂度优化到O(1). 2.遇到+-*/符号或者到结束位置就计算结果|
| 2. META ML RS过经及timeline |  on-site2-Coding 1  |2022.07| 65. Valid Number: 是否为有效的float数的字符串，可能包含.+-eE等符号| 字符串 | 中等 |注意很多badcase|
| 2. META ML RS过经及timeline |  on-site2-Coding 1  |2022.07| 670. Maximum Swap：返回只可以swap一次后最大的数| 数组 | 中等 |从后往前遍历并记录最大值，对每个位置，与替换最大值替换可以取得该位置swap后的最大值，与最前面一个位置swap可以取得全局最大值|

## 高频题总结
|类型|  题号 | 难度  | 题目 | 题解 | 
| ---- |  ----  | ----  | --- | --- |

# [面试经验] FB MLE一条龙新鲜面经
https://www.1point3acres.com/bbs/thread-650769-1-1.html

## 店面

### 题目1：791. Custom Sort String

```
791. Custom Sort String
You are given two strings order and s. All the characters of order are unique and were sorted in some custom order previously.

Permute the characters of s so that they match the order that order was sorted. More specifically, if a character x occurs before a character y in order, then x should occur before y in the permuted string.

Return any permutation of s that satisfies this property.

Example 1:

Input: order = "cba", s = "abcd"
Output: "cbad"
Explanation: 
"a", "b", "c" appear in order, so the order of "a", "b", "c" should be "c", "b", and "a". 
Since "d" does not appear in order, it can be at any position in the returned string. "dcba", "cdba", "cbda" are also valid outputs.
Example 2:

Input: order = "cbafg", s = "abcd"
Output: "cbad"
```

```python
class Solution(object):
    def customSortString(self, order, s):
        """
        :type order: str
        :type s: str
        :rtype: str
        """
        letter_to_freq = defaultdict(int)
        for letter in s:
            letter_to_freq[letter] += 1
        res = []
        order_letter_set = set()
        for order_letter in order:
            order_letter_set.add(order_letter)
            for i in range(letter_to_freq[order_letter]):
                res.append(order_letter)   
        for letter in s:
            if letter not in order_letter_set:
                res.append(letter)
        return ''.join(res)
                

```

### 题目2：560. Subarray Sum Equals K

```
Given an array of integers nums and an integer k, return the total number of subarrays whose sum equals to k.

A subarray is a contiguous non-empty sequence of elements within an array.

Example 1:

Input: nums = [1,1,1], k = 2
Output: 2
Example 2:

Input: nums = [1,2,3], k = 3
Output: 2

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/subarray-sum-equals-k
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

```python

class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        """
        [1,1,1], 2
        {0:1, 1:1, 2:1}
        cur_sum=3
        diff=0
        cnt=1

        """
        pre_sum_cnt = defaultdict(int)
        pre_sum_cnt[0] = 1
        cnt = 0
        cur_sum = 0
        for num in nums:
            cur_sum += num
            diff = cur_sum - k
            cnt += pre_sum_cnt[diff]
            pre_sum_cnt[cur_sum] += 1
            # print('pre_sum_cnt:', pre_sum_cnt)
        return cnt

```

## on-site

### Coding 1: 



#### LC 670. Maximum Swap
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
``


```python
class Solution(object):
    def maximumSwap(self, num):
        """
        :type num: int
        :rtype: int
        """

        num = list(str(num))
        n = len(num)
        swap_index = -1
        target_index = -1
        max_val = -1
        max_val_index = -1

        """
        i
               max_val = 7
               max_val_index = 1
             swap_index = 0
             target_index = 1
        [2,7,3,6]


           i
               max_val = 9
               max_val_index = 1
             swap_index = 2
             target_index = 3
        [3,9,9]
        """

        for i in range(n-1, -1, -1):
            if num[i] > max_val:
                max_val = num[i]
                max_val_index = i
            elif num[i] < max_val:
                swap_index = i
                target_index = max_val_index
        
        num[swap_index], num[target_index] = num[target_index], num[swap_index]
        return int(''.join(num))

```


#### 【没做出来】LC 394. Decode String

```
Given an encoded string, return its decoded string.

The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times. Note that k is guaranteed to be a positive integer.

You may assume that the input string is always valid; there are no extra white spaces, square brackets are well-formed, etc. Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, k. For example, there will not be input like 3a or 2[4].

The test cases are generated so that the length of the output will never exceed 105.

Example 1:

Input: s = "3[a]2[bc]"
Output: "aaabcbc"
Example 2:

Input: s = "3[a2[c]]"
Output: "accaccacc"
Example 3:

Input: s = "2[abc]3[cd]ef"
Output: "abcabccdcdcdef"

```

```python
class Solution(object):
    def decodeString(self, s):
        """
        :type s: str
        :rtype: str
        """

        """

        2[abc]3[4[b2[c]]]

        """
        n = len(s)
        cache = []
        cur = 0 # 当前累计的次数
        res = "" # 当前累计的字符
        for i in range(n):
            if s[i].isdigit():
                cur = cur * 10 + int(s[i])
            elif s[i] == '[':
                cache.append([cur, res]) # 把前面的保护起来
                res = ""
                cur = 0
            elif s[i] == ']':
                last_cur, last_res = cache.pop()
                res = last_res + last_cur * res
            else:
                res += s[i]
        return res

```


### Coding 2:  
#### LC 543. Follow up 如果是general tree呢？

```
Given the root of a binary tree, return the length of the diameter of the tree.

The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.

The length of a path between two nodes is represented by the number of edges between them.


Example 1:

Input: root = [1,2,3,4,5]
Output: 3
Explanation: 3 is the length of the path [4,2,1,3] or [5,2,1,3].
Example 2:

Input: root = [1,2]
Output: 1

```

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        self.diameter = 0

        def search(node):
            """return the depth of the node
            """
            if not node:
                return 0
            l = search(node.left)
            r = search(node.right)
            self.diameter = max(self.diameter, l + r)
            return max(l, r) + 1

        search(root)
        return self.diameter
```




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
#### 题目：
```
Given an integer array nums and an integer k, return true if nums has a continuous subarray of size at least two whose elements sum up to a multiple of k, or false otherwise.

An integer x is a multiple of k if there exists an integer n such that x = n * k. 0 is always a multiple of k.

 

Example 1:

Input: nums = [23,2,4,6,7], k = 6
Output: true
Explanation: [2, 4] is a continuous subarray of size 2 whose elements sum up to 6.
Example 2:

Input: nums = [23,2,6,4,7], k = 6
Output: true
Explanation: [23, 2, 6, 4, 7] is an continuous subarray of size 5 whose elements sum up to 42.
42 is a multiple of 6 because 42 = 7 * 6 and 7 is an integer.
Example 3:

Input: nums = [23,2,6,4,7], k = 13
Output: false

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/continuous-subarray-sum
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

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

### Round3(ML)


### Round3(BQ)


# Meta高频
## 分类与总结

### 总结


### stack

#### 1249. Minimum Remove to Make Valid Parentheses

```
Given a string s of '(' , ')' and lowercase English characters.

Your task is to remove the minimum number of parentheses ( '(' or ')', in any positions ) so that the resulting parentheses string is valid and return any valid string.

Formally, a parentheses string is valid if and only if:

It is the empty string, contains only lowercase characters, or
It can be written as AB (A concatenated with B), where A and B are valid strings, or
It can be written as (A), where A is a valid string.
 

Example 1:

Input: s = "lee(t(c)o)de)"
Output: "lee(t(c)o)de"
Explanation: "lee(t(co)de)" , "lee(t(c)ode)" would also be accepted.
Example 2:

Input: s = "a)b(c)d"
Output: "ab(c)d"
```

```python
class Solution(object):
    def minRemoveToMakeValid(self, s):
        """
        :type s: str
        :rtype: str

        "abc

        left=1
         right=

        a)b(c)d

        ((abc(d

        ['lee', [''], t, ['']]

        """

        left = 0
        left_index = []
        right = 0

        remove_index_set = set()

        for i in range(len(s)):
            if s[i] == '(':
                left_index.append(i)
                left += 1
            elif s[i] == ')':
                if left <= 0:
                    remove_index_set.add(i)
                else:
                    left -= 1
                    left_index.pop()
                # print("left_index:", left_index)

        while left_index:
            remove_index_set.add(left_index.pop())

        res = []
        for i in range(len(s)):
            if i not in remove_index_set:
                res.append(s[i])

        return ''.join(res)
        
```

## 高频题

#### 1249. Minimum Remove to Make Valid Parentheses

```
Given a string s of '(' , ')' and lowercase English characters.

Your task is to remove the minimum number of parentheses ( '(' or ')', in any positions ) so that the resulting parentheses string is valid and return any valid string.

Formally, a parentheses string is valid if and only if:

It is the empty string, contains only lowercase characters, or
It can be written as AB (A concatenated with B), where A and B are valid strings, or
It can be written as (A), where A is a valid string.
 

Example 1:

Input: s = "lee(t(c)o)de)"
Output: "lee(t(c)o)de"
Explanation: "lee(t(co)de)" , "lee(t(c)ode)" would also be accepted.
Example 2:

Input: s = "a)b(c)d"
Output: "ab(c)d"
```

```python
class Solution(object):
    def minRemoveToMakeValid(self, s):
        """
        :type s: str
        :rtype: str

        "abc

        left=1
         right=

        a)b(c)d

        ((abc(d

        ['lee', [''], t, ['']]

        """

        left = 0
        left_index = []
        right = 0

        remove_index_set = set()

        for i in range(len(s)):
            if s[i] == '(':
                left_index.append(i)
                left += 1
            elif s[i] == ')':
                if left <= 0:
                    remove_index_set.add(i)
                else:
                    left -= 1
                    left_index.pop()
                # print("left_index:", left_index)

        while left_index:
            remove_index_set.add(left_index.pop())

        res = []
        for i in range(len(s)):
            if i not in remove_index_set:
                res.append(s[i])

        return ''.join(res)
        
```

#### 680. Valid Palindrome II

```
Given a string s, return true if the s can be palindrome after deleting at most one character from it.

Example 1:

Input: s = "aba"
Output: true
Example 2:

Input: s = "abca"
Output: true
Explanation: You could delete the character 'c'.
Example 3:

Input: s = "abc"
Output: false
```

```python

class Solution(object):
    def validPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """


        def is_valid_palindrom_between_left_and_right(l, r, delete_flag):
            while l <= r:
                if l == r:
                    return True
                if s[l] == s[r]:
                    if r - l <= 1:
                        return True
                    l += 1
                    r -= 1
                else:
                    if not delete_flag:
                        return is_valid_palindrom_between_left_and_right(l, r-1, True) or \
                            is_valid_palindrom_between_left_and_right(l+1, r, True)
                    else:
                        return False
            return False


        return is_valid_palindrom_between_left_and_right(0, len(s) - 1, False)
                

```
#### 953. Verifying an Alien Dictionary


```
In an alien language, surprisingly, they also use English lowercase letters, but possibly in a different order. The order of the alphabet is some permutation of lowercase letters.

Given a sequence of words written in the alien language, and the order of the alphabet, return true if and only if the given words are sorted lexicographically in this alien language.

 

Example 1:

Input: words = ["hello","leetcode"], order = "hlabcdefgijkmnopqrstuvwxyz"
Output: true
Explanation: As 'h' comes before 'l' in this language, then the sequence is sorted.
Example 2:

Input: words = ["word","world","row"], order = "worldabcefghijkmnpqstuvxyz"
Output: false
Explanation: As 'd' comes after 'l' in this language, then words[0] > words[1], hence the sequence is unsorted.
```


```python

class Solution(object):
    def isAlienSorted(self, words, order):
        """
        :type words: List[str]
        :type order: str
        :rtype: bool
        """

        def slower_than(word1, word2, letter_start_index):
            """return True if word1 < word2"""

            for i in range(min(len(word1), len(word2))):
                letter1, letter2 = word1[i], word2[i]
                l1_index = ord(letter1) - ord('a')
                l2_index = ord(letter2) - ord('a')
                # print(letter1, letter2, l1_index, l2_index)
                if letter_start_index[l1_index] > letter_start_index[l2_index]:
                    return False
                elif letter_start_index[l1_index] < letter_start_index[l2_index]:
                    return True
            if len(word1) > len(word2):
                return False            
            return True


        letter_start_index = [float('inf')] * 26
        for i in range(len(order)):
            letter = order[i]
            if letter not in letter_start_index:
                letter_start_index[ord(letter) - ord('a')] = i

        for i in range(len(words) - 1):
            if not slower_than(words[i], words[i+1], letter_start_index):
                return False
        
        return True
```


#### 301. Remove Invalid Parentheses

```

Given a string s that contains parentheses and letters, remove the minimum number of invalid parentheses to make the input string valid.

Return all the possible results. You may return the answer in any order.

 

Example 1:

Input: s = "()())()"
Output: ["(())()","()()()"]
Example 2:

Input: s = "(a)())()"
Output: ["(a())()","(a)()()"]
Example 3:

Input: s = ")("
Output: [""]
```


```python
class Solution(object):
    def removeInvalidParentheses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """

        def is_valid(s1):
            if not s1:
                return True
            l, r = 0, 0
            for i in range(len(s1)):
                if s1[i] == '(':
                    l += 1
                elif s1[i] == ')':
                    if l <= 0:
                        return False
                    else:
                        l -= 1
            return l == r

        res = []
        def search(s1, l, r, start):
            # print("s1, l, r, start:", s1, l, r, start)
            if l == 0 and r == 0 and is_valid(s1):
                res.append(s1)
                # print('res:', res, 's1:' ,s1)
                return
            if r > 0:
                for i in range(start, len(s1)):
                    if i != start and s1[i] == s1[i-1]:
                        continue
                    if s1[i] == ')':
                        search(s1[:i]+s1[i+1:], l, r - 1, i)# 减去第i个元素了，下一个还是从i开始删
            elif l > 0:
                for i in range(start, len(s1)):
                    if i != start and s1[i] == s1[i-1]:
                        continue
                    if s1[i] == '(':
                        search(s1[:i]+s1[i+1:], l - 1, r, i)

        n = len(s)
        l, r = 0, 0
        for i in range(n):
            if s[i] == '(':
                l += 1
            elif s[i] == ')':
                if l <= 0:
                    r += 1
                else:
                    l -= 1

        search(s, l, r, 0)
        return res
```
#### 973. K Closest Points to Origin

```
Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane and an integer k, return the k closest points to the origin (0, 0).

The distance between two points on the X-Y plane is the Euclidean distance (i.e., √(x1 - x2)2 + (y1 - y2)2).

You may return the answer in any order. The answer is guaranteed to be unique (except for the order that it is in).

 

Example 1:


Input: points = [[1,3],[-2,2]], k = 1
Output: [[-2,2]]
Explanation:
The distance between (1, 3) and the origin is sqrt(10).
The distance between (-2, 2) and the origin is sqrt(8).
Since sqrt(8) < sqrt(10), (-2, 2) is closer to the origin.
We only want the closest k = 1 points from the origin, so the answer is just [[-2,2]].
Example 2:

Input: points = [[3,3],[5,-1],[-2,4]], k = 2
Output: [[3,3],[-2,4]]
Explanation: The answer [[-2,4],[3,3]] would also be accepted.

```

```python
class Solution(object):
    def kClosest(self, points, k):
        """
        :type points: List[List[int]]
        :type k: int
        :rtype: List[List[int]]
        """
        def get_dis(x1, y1):
            return x1 ** 2 + y1 ** 2

        dis = [[get_dis(points[i][0], points[i][1]), i] for i in range(len(points))]

        def find_k_closest_pointer(l, r, res_k):
            k1 = l - 1
            for i in range(l, r):
                if dis[i][0] <= dis[r][0]:
                    k1 += 1
                    dis[i], dis[k1] = dis[k1], dis[i]
            k1 += 1
            dis[r], dis[k1] = dis[k1], dis[r]
            if k1 - l + 1 == res_k:
                return [points[dis[i][1]] for i in range(k)]
            if res_k > k1 - l + 1:
                return find_k_closest_pointer(k1 + 1, r, res_k - (k1 - l + 1))
            else:
                return find_k_closest_pointer(l, k1 - 1, res_k)

        return find_k_closest_pointer(0, len(dis) - 1, k)
        
```

#### 67. Add Binary

```
Example 1:

Input: a = "11", b = "1"
Output: "100"
Example 2:

Input: a = "1010", b = "1011"
Output: "10101"
```

```python
class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        res = []
        add = 0
        m, n = len(a), len(b)
        for i in range(max(m, n)):
            if i < m:
                x = a[m - i - 1]
            else:
                x = 0
            if i < n:
                y = b[n - i - 1]
            else:
                y = 0
            s = int(x) + int(y) + add
            add = s // 2
            res.append(s % 2)
        if add:
            res.append(add)
        return ''.join([str(_) for _ in res[::-1]])
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
3. 记录我曾经面试 Facebook（Meta） 的经历：https://sichengingermay.com/facebook-interview/
4. Meta面试经历，被拒，两次！https://zhuanlan.zhihu.com/p/499547331