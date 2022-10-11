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
| mock |  mock2  |2022.10| [691]Stickers to Spell Word: 返回最小的使用stickers组合成target的数量 | dfs+缓存+状态压缩 or bfs | 困难 | 1. 可以用dfs从target开始搜索，每次搜索的时候，尝试使用stickers中的每个结果，最终取最小的。2. 状态比较多，用11111代表每个位置是否已经收集到，这种效率要高一点，相当于利用位运算做了压缩。3. 可以加缓存优化速度。|
| 店面 |  店面  |2022.10| unsorted array 找是否有三个number, a,b,c 满足 a**2 + b**2 = c**2 | 三数之和 | 中等 | 1. 三数之和，先排序，然后双指针。2.c为target，然后利用a+b=c找rarget。|
| 店面 |  店面-onsite1  |2022.1| N（很大的数）个排好序的找有多少unique 数字 | 指针 | 中等 | 比较pre和cur的值是否有有变化，变化计数+1，否则不变，注意pre可以设置为nums[0]|
| 店面 |  店面-onsite2  |2022.1| 286 Walls and Gates 变‍‌‍‌‍‍‌‌‍‍‍‍‍‍‍‍‌‌‌‌种 题目：有墙：-1，门：0，空：INF，给所有空打上离门最近的距离 | bfs | 中等 | bfs先收集门，然后往外拓展|
| 店面 |  店面-onsite3  |2022.1| [863]All Nodes Distance K in Binary Tree：给出root，target，k，找出所有离target距离为k的节点 | bfs | 中等 | 1. 先建图，然后用bfs查找|
| 店面 |  店面-题目1  |2022.10| [987]Vertical Order Traversal of a Binary Tree: 垂直并排序输出结果 | dfs+哈希 | 困难 | 先dfs+hash收集每一列的数字，记录他的位置信息，然后排序输出|
| 店面 |  店面-题目1  |2022.10| [987]Vertical Order Traversal of a Binary Tree: 垂直并排序输出结果 | dfs+哈希 | 困难 | 先dfs+hash收集每一列的数字，记录他的位置信息，然后排序输出|







## 高频题总结
|类别|类型|  题号 | 难度  | 题目 | 题解 | 
| --- | ---- |  ----  | ----  | --- | --- |
|字符串| 字符串-括号删除 |  1249. Minimum Remove to Make Valid Parentheses  | 中等  | 输入括号和英文字母，返回最少删除后valid的字符串 | 1. 只有右边括号优先删除。2.剩余的左边的括号需要删除。 3. 记录需要删除的index_set，返回不在index_set中的字符串即可。4. 如果要返回所有的合理的结果，那么需要从左到右先删除右边的括号，右边的括号删除完了，才能删除剩余的左边的括号，因为先删除左括号，会导致左边的右括号可能更加没法匹配。|
|字符串| 字符串处理 |  67. Add Binary  | 中等  | 字符串相加 | 注意检查结束位置的add|
|回溯| 字符串-括号删除 |  301. Remove Invalid Parentheses  | 困难  | 尽可能删除最小的左右括号数量使字符串匹配，返回所有可能的匹配的结果。 | 1. 先分别统计左右需要删除的括号的数量，然后先删除右括号，再删除左括号。2. 比较难想到的是用递归的方法依次删除括号。 3. 注意start用于标记回溯算法当前删除节点的起点位置。|
|递归| 字符串处理 |  273. Integer to English Words  | 困难  | 数字转英文 |1. 递归思想解决1-10，10-20，20-100，100-1000，1000-1000000， 1000000-1000000000的英文处理。2. 注意four，forty，twenty，thousand，million，billion等写法。|
|双指针| 字符串-回文串-双指针 |  680. Valid Palindrome II  | 中等  | 最多删除一个字符，问字符串是否能为回文串 | 1. 第一反应，尝试对每个字符串来进行删除，验证是否是回文串，或者字符串本身就是回文串，时间复杂度O(n**2)级别。2. 优化：双指针，先找到需要删除的节点，然后删除左边或者右边，来看是否是回文串，时间复杂度可以优化到O(n)级别。|
|双指针| 数组 |  [31]Next Permutation  | 中等  | 求下一个排列（比当前节点大） | 1. 从后往前，找到第一个逆序的数字，逆序代表从后面swap可以找到一个更大的数。 2. swap后，对后面的数字利用双指针倒序排列，即为下一个最大的数字。|
|排序| 排序 |  953. Verifying an Alien Dictionary  | 中等  | 验证words是否按照外星人的字母表进行排序 | 1. 先统计每个外星人的字母的顺序index，然后从左到右验证words中每个字符与前一个字符是否是排好序的。|
|排序| 数组排序 |  [56]Merge Intervals  | 中等  | 合并间隔 | 先排序后合并|
|数组| 类似前缀和 |  [238]Product of Array Except Self  | 中等  | 计算除自身外的product | 分别记录左边和右边的product，然后输出，时间复杂度可以优化到O(n)|
|数组| 数组 |  [1762]Buildings With an Ocean View  | 中等  | 最后一个是海，如果建筑比后面的高就可以看到海，输出所有能看到海的index | 直接倒序搜索就可以|
|树| bfs，dfs|  314. Binary Tree Vertical Order Traversal  | 中等  | 二进制树的垂直搜索 | 1. bfs直接搜索，然后root的坐标为0，左边-1，右边+1，然后用loc_to_vallist记录每个loc的坐标，同时记录min_loc和max_loc最终直接输出即可。|
|树| 树的搜索 |  [938]Range Sum of BST  | 中等  | 给定一个范围和一颗BST树，问在这个范围内的数字的和 | 直接树的二分搜索即可|
|topk快排| 数组排序 |  [215]Kth Largest Element in an Array  | 中等  | 找到第topk大的树 | 快排+递归思想，找前k大或者第k大都可以做 |
|树| 树的搜索 |  [199]Binary Tree Right Side View  | 中等  | 从右边看能看到的树的数字 | 记录每一层是否visited过，先右边，再左边递归遍历，这样就可以收集每一层第一次访问的节点，那么就是结果|
|树| 后续便利搜索 |  236 Lowest Common Ancestor of a Binary Tree  | 中等  | 二叉树最近的父节点 | 1. 递归实现，对于左边或者右边都能找到q或者p，那么root就是结果，否则可以递归检查左边或者右边。 2. 对于root=q或者root=p或者root为None，可以直接返回结果。|
|设计题| 字符串处理 |  [1570]Dot Product of Two Sparse Vectors  | 中等  | 稀疏矩阵点乘 | 1. 储存的时候，只记录不为0的index和val。 2.计算点乘的时候，可以用双指针的思想来match并计算点乘和。|
|设计题| 数组 |  528. Random Pick with Weight  | 中等  | 按给定的权重w，随机选择数据 | 1. 计算权重的累积概率和，然后利用random.ramdom()生成一个随机的数。2. 利用二分找到概率在累积概率和中的区域位置，找到后即可返回结果。|
|设计题| 中序遍历非递归写法改编 |  173. Binary Search Tree Iteratory  | 中等  | 设计个类，可以实现BST的next，hasNext的函数 | 1. 中序遍历非递归方法用栈实现，首先一直往左压入栈，然后弹出，然后改成右节点继续压入。2. init阶段执行往左压栈不走，hasNext检查栈是否为空，next弹出，并切换成右节点。|
|位运算| 数学 |  29 Divide Two Integers  | 中等  | 不用乘除法实现除法 | 1. 一直累加可以实现，但是耗时比较长，例如2000/2要累加1000次，可以考虑累加(2,1)，翻倍后(4,2)，翻倍后(8,4)这种方式来实现。2. 初始化base=2，cnt=1，每次base和cnt翻倍，知道a - b < b就不翻倍了，记录此时的cnt，然后用a -= base，继续重复2，直到a < b。 3.注意输出的范围为int的最大值和最小值。|
|设计题| 树的序列化 |  297 Serialize and Deserialize Binary Tree  | 困难  | 将一个二进制树序列化和非序列化 | 1. 可以利用bfs序列化，注意val为负数的时候，有个负号，所以序列化的时候，最好加一个分隔符。2. 反序列化的时候，还是用bfs思想，用i记录用到那个序列化的数据了，每个数据，有两个孩子，需要消耗2个孩子来实现，对于非空的节点，可以直接加到deque里面去，继续进行bfs。|
|二分| 二分法 |  278 First Bad Version  | 中等  | 已知一个函数，求如何找到第一额bad version | cand+二分思想解决问题。|
|上下车问题| 数组 |  253 Meeting Rooms II  | 中等  | 已知每个会议的开始结束时间，问最少需要多少个会议室。 | 转化成每个会议开始时+1，每个会议结束时-1，处理的时候需要从小到大排序，这样可以记录每个时间结束时候的人员的数量，返回最大的即可。|
|计数| 数学 |  621. Task Scheduler  | 中等  | 一堆字母用cpu来处理，每个字母需要间隔sep才能重新出现，每个时刻可以等一下，问最低需要多少cpu时间 | 二维图，对于最大的频次的字母，乘以间隔，就可能是结果，因为可能其他的字符比较多，但是出现的频次较低，所以需要取两者最大值：max((max_freq_num - 1) * (sep + 1) + num_of_max_freq, len(s))|
|字符串| 字符串处理 |  67. Add Binary  | 中等  | 字符串相加 | 注意检查结束位置的add|
|字符串| 字符串处理 |  67. Add Binary  | 中等  | 字符串相加 | 注意检查结束位置的add|
|字符串| 字符串处理 |  67. Add Binary  | 中等  | 字符串相加 | 注意检查结束位置的add|
|字符串| 字符串处理 |  67. Add Binary  | 中等  | 字符串相加 | 注意检查结束位置的add|
|字符串| 字符串处理 |  67. Add Binary  | 中等  | 字符串相加 | 注意检查结束位置的add|







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


# meta-店面 2022.10
https://www.1point3acres.com/bbs/thread-935367-1-1.html
## 店面
### 题目1: 339. Nested List Weight Sum
```
Input  [1, [2, 3], 4, [5, [6, [7, 8, 9]]]] 求和。 但是每多一个【】， 里面的数字的权重加一。 我就直接写了个recursive function， 把权重带pass 进去。
```

- 伪代码

```python
def get_weight_sum(arr):

    w = 1
    s = 0
    def search(arr):
        for ele in arr:
            if ele.isInterger():
                s += w * ele.val
            else:
                w += 1
                search(ele)
                w -= 1
    return s
```

- 正式代码

```python
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

### 题目2: [347]Top K Frequent Elements

- 题目
```
给一个list of integers， 找前K frequent的数.  应该有很快的linear 算法吧。 我用了Tree跟Hash。

[1,1,1, ]
```

- topk问题

```python

# leetcode submit region begin(Prohibit modification and deletion)
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        from collections import defaultdict
        freq_of_num = defaultdict(int)
        for num in nums:
            freq_of_num[num] += 1
        num_freq_list = []
        for num, freq in freq_of_num.items():
            num_freq_list.append([num, freq])
        # print(num_freq_list)
        def select_topk_freq_nums(num_freq_list, l, r, k):
            k1 = l - 1
            base = num_freq_list[r][1]
            for i in range(l, r):
                if num_freq_list[i][1] > base:
                    k1 += 1
                    num_freq_list[k1], num_freq_list[i] = num_freq_list[i], num_freq_list[k1]
            k1 += 1
            num_freq_list[k1], num_freq_list[r] = num_freq_list[r], num_freq_list[k1]
            if k1 - l + 1 == k:
                return
            elif k1 - l + 1 > k:
                select_topk_freq_nums(num_freq_list, l, k1 - 1, k)
            else:
                select_topk_freq_nums(num_freq_list, k1 + 1, r, k - (k1 - l + 1))
        select_topk_freq_nums(num_freq_list, 0, len(num_freq_list) - 1, k)
        res = []
        for i in range(k):
            res.append(num_freq_list[i][0])
        return res
# leetcode submit region end(Prohibit modification and deletion)

s = Solution()
s.topKFrequent(nums=[1,1,1,2,2,3], k=2)

```

# meta-店面 2022.10.06 mock
## mock
2道题：尔凌凌，刘久以

### 题目1: [200]Number of Islands

```python

# leetcode submit region begin(Prohibit modification and deletion)
class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        visited = set()
        m, n = len(grid), len(grid[0])
        def dfs(x, y):
            for dx, dy in [[-1, 0], [1, 0], [0, 1], [0, -1]]:
                nx, ny = x + dx, y + dy
                new_index = nx * n + ny
                if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == '1' and new_index not in visited:
                    visited.add(new_index)
                    dfs(nx, ny)

        num_of_islands = 0
        for i in range(m):
            for j in range(n):
                index = i * n + j
                if grid[i][j] == '1' and index not in visited:
                    # print(i, j)
                    visited.add(index)
                    dfs(i, j)
                    num_of_islands += 1

        return num_of_islands

a = [["1","0","1","1","1"],["1","0","1","0","1"],["1","1","1","0","1"]]
s = Solution()
print(s.numIslands(a))
# leetcode submit region end(Prohibit modification and deletion)

```

### 题目2:[691]Stickers to Spell Word 【未搞懂】

```
We are given n different types of stickers. Each sticker has a lowercase English word on it.

You would like to spell out the given string target by cutting individual letters from your collection of stickers and rearranging them. You can use each sticker more than once if you want, and you have infinite quantities of each sticker.

Return the minimum number of stickers that you need to spell out target. If the task is impossible, return -1.

Note: In all test cases, all words were chosen randomly from the 1000 most common US English words, and target was chosen as a concatenation of two random words.

Example 1:

Input: stickers = ["with","example","science"], target = "thehat"
Output: 3
Explanation:
We can use 2 "with" stickers, and 1 "example" sticker.
After cutting and rearrange the letters of those stickers, we can form the target "thehat".
Also, this is the minimum number of stickers necessary to form the target string.
Example 2:

Input: stickers = ["notice","possible"], target = "basicbasic"
Output: -1
Explanation:
We cannot form the target "basicbasic" from cutting letters from the given stickers.
```
- dfs+缓存(易于理解+超时)

```python
class Solution(object):
    def minStickers(self, stickers, target):
        """
        :type stickers: List[str]
        :type target: str
        :rtype: int
        """
        best = dict()
        def convert(s):
            res = [0] * 26
            for i in range(len(s)):
                res[ord(s[i]) - ord('a')] += 1
            return res
        
        stickers_list = [convert(_) for _ in stickers]
        target_list = convert(target)

        def sub(a_list, b_list):
            res = [0] * 26
            for i in range(26):
                res[i] = max(0, a_list[i] - b_list[i])
            return res
        
        def get_key(target_list):
            return ','.join([str(_) for _ in target_list])

        def dfs(target_list):
            if sum(target_list) == 0:
                return 0
            key = get_key(target_list)
            if key in best:
                return best[key]
            best[key] = float('inf')
            for sticker_list in stickers_list:
                sub_res = sub(target_list, sticker_list)
                best[key] = min(dfs(sub_res) + 1, best[key])
            return best[key]

        res = dfs(target_list)
        return -1 if res >= float('inf') else res
```

- dfs+缓存+状态压缩优化版本

```python
class Solution(object):
    def minStickers(self, stickers, target):
        """
        :type stickers: List[str]
        :type target: str
        :rtype: int

        time: O((m+n)*n*2^m)
        space:O(2^m)
        """
        best = dict()
        def search(cur):
            if not cur:
                return 0
            if cur in best:
                return best[cur]
            best[cur] = float('inf')
            for sticker in stickers:
                letter_cnt = Counter(sticker)
                base = cur
                for i in range(len(target)):
                    letter = target[i]
                    if letter_cnt[letter] > 0 and base & (1 << i) != 0:
                        letter_cnt[letter] -= 1
                        base ^= (1 << i) # 取反，标记该位置已经有了
                if base < cur:
                    best[cur] = min(best[cur], search(base) + 1)
            return best[cur]
        res = search((1 << len(target)) - 1)
        return -1 if res >= float('inf') else res
```

- 状态压缩+dp
```python

# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def minStickers(self, stickers: List[str], target: str) -> int:
        """
        :type stickers: List[str]
        :type target: str
        :rtype: int
        """
        """
        题目：用stickers来凑target里面的字母，求所需要的最小的stickers的数量
        解法：状态压缩+动态规划
        1. 状态压缩：对于target里面的每个字符相当于一位，每一位都完成了，就代表target完成了。
        2. target的状态共有 1<<len(target)种，对于每种状态，分析添加一个sticker能够到达哪个状态，这样这个状态的个数可以由前一个得来。
        3. 加入一个sticker，可以遍历每一位状态，检查该sticker能不能补充这个状态。
        """
        n = len(target)
        m = len(stickers)
        # 记录每个sticker有多少可用的字符
        sticker2charcnt = [[0] * 26 for _ in range(m)]
        # 每一位对应一个状态
        for i in range(m):
            sticker = stickers[i]
            for c in sticker:
                sticker2charcnt[i][ord(c) - ord('a')] += 1
        # 定义dp，dp[i]代表完成状态i最小需要多少个stickers
        total_num_of_status = (1 << n)
        dp = [float('inf')] * (1 << 15)
        # print("total_num_of_status:", total_num_of_status)
        dp[0] = 0
        # 遍历每一位状态
        for i in range(total_num_of_status):
            # 这个代表这个状态还没搞定，那么也没法演变成其他状态
            if dp[i] == float('inf'):
                continue
            # 添加sticker，看能转移到哪个状态
            for sticker_index, sticker in enumerate(stickers):
                # print('sticker:',sticker)
                # 每个sticker都单独查看一遍
                next_v = i
                # 遍历每一位状态，检查是否需要这个状态，以及sticker是否包含这个状态
                # 注意python这里要copy一下list
                cur_sticker_charcnt = sticker2charcnt[sticker_index].copy()
                for cand_statu_index in range(n):
                    # 代表i已经有这个位了
                    if next_v & (1 << cand_statu_index) != 0:
                        continue
                    # 原来没有这个位，则检查sticker能不能提供这个位
                    need_char = target[cand_statu_index]
                    need_char_index = ord(need_char) - ord('a')
                    if cur_sticker_charcnt[need_char_index] > 0:
                        # print("need_char:", need_char)
                        next_v += (1 << cand_statu_index)
                        # print("next_v:", next_v)
                        cur_sticker_charcnt[need_char_index] -= 1
                dp[next_v] = min(dp[next_v], dp[i] + 1) # dp[i] + 1代表多加了一个sticker可以转移到状态next_v，取最小值。
        # print(dp)
        return dp[total_num_of_status - 1] if (
                    dp[total_num_of_status - 1] and dp[total_num_of_status - 1] != float('inf')) else - 1
# runtime:3432 ms
# memory:15.3 MB
# leetcode submit region end(Prohibit modification and deletion)
```









# meta-店面 2021.
https://www.1point3acres.com/bbs/thread-814892-1-1.html
## 店面
```
第一题 三幺四第二题 unsorted array 找是否有三个number, a,b,c 满足 a**2 + b**2 = c**2
```
### 314. Binary Tree Vertical Order Traversal

```python

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def verticalOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        d = deque()
        d.append([root, 0])
        loc_to_node_list = defaultdict(list)
        min_loc = 0
        while d:
            cur, cur_loc = d.popleft()
            loc_to_node_list[cur_loc].append(cur.val)
            if cur.left:
                d.append([cur.left, cur_loc - 1])
            if cur.right:
                d.append([cur.right, cur_loc + 1])
            min_loc = min(cur_loc, min_loc)
        res = []
        for loc in range(min_loc, min_loc + len(loc_to_node_list)):
            res.append(loc_to_node_list[loc])
        return res
```

### unsorted array 找是否有三个number, a,b,c 满足 a**2 + b**2 = c**2
- 思路
```
这里是OP，第二题可以有负数，我是按照three sum的思路做的，先map成平方的形式，再sort，然后for loop+two pointers

```
```python

def find_three_nums(arr):
    arr = [_**2 for _ in arr]
    arr.sort()
    for k3 in range(len(arr) - 1, 1, -1):
        if k3 != len(arr) - 1 and arr[k3] == arr[k3] + 1:
            continue
        c = arr[k3]
        # a + b = c
        k1, k2 = 0, k3 - 1
        while k1 < k2:
            if arr[k1] + arr[k2] == c:
                return True
            elif arr[k1] + arr[k2] < c:
                k1 += 1
            else:
                k2 -= 1
```
- 三数之和题解

```python
class Solution(object):
    def threeSum(self, nums):
        """找出所有三数之和为0的数字, 注意要去重
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
            target = -nums[i]
            l = i + 1
            r = n - 1
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

# meta-店面 2022.1
https://www.1point3acres.com/bbs/thread-843295-1-1.html
## onsite
```
一月中旬面试的Meta，感觉还不错。
1轮System Design， 1轮ML Infra Design，2轮Coding，1轮Behavior
System： 爬虫， ML：Ads CTR prediction，Coding： 1. N（很大的数）个排好序的找有多少unique 数字。2. Walls and Gates 变‍‌‍‌‍‍‌‌‍‍‍‍‍‍‍‍‌‌‌‌种。3. 二叉树中距离为K的node。 4. 忘了，但是是一个非常Easy的题目。
```
### N（很大的数）个排好序的找有多少unique 数字

"""
pre=1
          l
cnt=2

1,1,1,1,1,2
1,1,2,2

pre=1
 l
cnt=2
1,2

pre=3
cnt=3
       l
1,1,2,3,3
"""

```python
def num_of_unique_numbers(nums):
    n = len(nums)
    if n <= 0:
        return 0
    pre = nums[0]
    i = 0
    cnt = 1
    while i < m:
        if nums[i] != pre:
            cnt += 1
        pre = nums[i]
    return cnt
```


### 286 Walls and Gates 变‍‌‍‌‍‍‌‌‍‍‍‍‍‍‍‍‌‌‌‌种

```
You are given an m x n grid rooms initialized with these three possible values.

-1 A wall or an obstacle.
0 A gate.
INF Infinity means an empty room. We use the value 231 - 1 = 2147483647 to represent INF as you may assume that the distance to a gate is less than 2147483647.
Fill each empty room with the distance to its nearest gate. If it is impossible to reach a gate, it should be filled with INF

Input: rooms = [[2147483647,-1,0,2147483647],[2147483647,2147483647,2147483647,-1],[2147483647,-1,2147483647,-1],[0,-1,2147483647,2147483647]]
Output: [[3,-1,0,1],[2,2,1,-1],[1,-1,2,-1],[0,-1,3,4]]
Example 2:

Input: rooms = [[-1]]
Output: [[-1]]
```

```python


# leetcode submit region begin(Prohibit modification and deletion)
class Solution(object):
    def wallsAndGates(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: None Do not return anything, modify rooms in-place instead.
        -1 A wall or an obstacle.
        0 A gate.
        INF Infinity means an empty room. We use the value 231 - 1 = 2147483647 to represent INF as you may assume that the distance to a gate is less than 2147483647.
        """
        INF = (1 << 31) - 1
        m, n = len(rooms), len(rooms[0])
        visited = set()
        d = deque()
        for i in range(m):
            for j in range(n):
                if rooms[i][j] == 0:
                    d.append([i, j, 0])
        while d:
            size = len(d)
            for i in range(size):
                x, y, dis = d.popleft()
                rooms[x][y] = dis
                for dx, dy in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                    nx, ny = dx + x, dy + y
                    if 0 <= nx < m and 0 <= ny < n and (nx, ny) not in visited and rooms[nx][ny] >= INF:
                        visited.add((nx, ny))
                        d.append([nx, ny, dis + 1])
```


### [863]All Nodes Distance K in Binary Tree

```
Given the root of a binary tree, the value of a target node target, and an integer k, return an array of the values of all nodes that have a distance k from the target node.

You can return the answer in any order.

Example 1:

Input: root = [3,5,1,6,2,0,8,null,null,7,4], target = 5, k = 2
Output: [7,4,1]
Explanation: The nodes that are a distance 2 from the target node (with value 5) have values 7, 4, and 1.
Example 2:

Input: root = [1], target = 1, k = 3
Output: []
```

```python

# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def distanceK(self, root, target, k):
        """
        :type root: TreeNode
        :type target: TreeNode
        :type k: int
        :rtype: List[int]
            0
             1
               2
                 3

                 1
                 2
        """
        child_to_parent = dict()
        def dfs(node):
            if node.left:
                child_to_parent[node.left] = node
                dfs(node.left)
            if node.right:
                child_to_parent[node.right] = node
                dfs(node.right)
        dfs(root)
        d = deque()
        d.append([target, 0])
        visited = set()
        visited.add(target)
        res = []
        while d:
            n = len(d)
            for i in range(n):
                cur, dis = d.popleft()
                if dis == k:
                    res.append(cur.val)
                elif dis < k:
                    if cur.left and cur.left not in visited:
                        visited.add(cur.left)
                        d.append([cur.left, dis + 1])
                    if cur.right and cur.right not in visited:
                        visited.add(cur.right)
                        d.append([cur.right, dis + 1])
                    if cur in child_to_parent:
                        parent = child_to_parent[cur]
                        if parent not in visited:
                            visited.add(parent)
                            d.append([parent, dis + 1])
        return res
```


# meta MLE 店面 2022.10
https://www.1point3acres.com/bbs/thread-932740-1-1.html
## 店面
```
臼捌跂 follow up 问保持原来字符串不在给定序列里的顺序
壹珋弍变体求最小 follow up 问如果第一个和最后一个没有相邻也算满足条件求结果
```
### [987]Vertical Order Traversal of a Binary Tree

```
Given the root of a binary tree, calculate the vertical order traversal of the binary tree.

For each node at position (row, col), its left and right children will be at positions (row + 1, col - 1) and (row + 1, col + 1) respectively. The root of the tree is at (0, 0).

The vertical order traversal of a binary tree is a list of top-to-bottom orderings for each column index starting from the leftmost column and ending on the rightmost column. There may be multiple nodes in the same row and same column. In such a case, sort these nodes by their values.

Return the vertical order traversal of the binary tree.

Example 1:


Input: root = [3,9,20,null,null,15,7]
Output: [[9],[3,15],[20],[7]]
Explanation:
Column -1: Only node 9 is in this column.
Column 0: Nodes 3 and 15 are in this column in that order from top to bottom.
Column 1: Only node 20 is in this column.
Column 2: Only node 7 is in this column.
Example 2:


Input: root = [1,2,3,4,5,6,7]
Output: [[4],[2],[1,5,6],[3],[7]]
Explanation:
Column -2: Only node 4 is in this column.
Column -1: Only node 2 is in this column.
Column 0: Nodes 1, 5, and 6 are in this column.
          1 is at the top, so it comes first.
          5 and 6 are at the same position (2, 0), so we order them by their value, 5 before 6.
Column 1: Only node 3 is in this column.
Column 2: Only node 7 is in this column.
Example 3:


Input: root = [1,2,3,4,6,5,7]
Output: [[4],[2],[1,5,6],[3],[7]]
Explanation:
This case is the exact same as example 2, but with nodes 5 and 6 swapped.
Note that the solution remains the same since 5 and 6 are in the same location and should be ordered by their values.
```

```python

# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def verticalTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        col_to_nodes = defaultdict(list)
        self.min_col, self.max_col = 0, 0
        def search(node, row, col):
            if not node:
                return
            if col < self.min_col:
                self.min_col = col
            if col > self.max_col:
                self.max_col = col
            col_to_nodes[col].append([node.val, row, col])
            search(node.left, row + 1, col - 1)
            search(node.right, row + 1, col + 1)

        search(root, 0, 0)
        res = []
        for col in range(self.min_col, self.max_col + 1):
            col_list = col_to_nodes[col]
            if col_list:
                col_list.sort(key = lambda x:[x[1], x[0]])
                res.append([_[0] for _ in col_list])
        return res

# leetcode submit region end(Prohibit modification and deletion)

```


### TODO：192变体

```shell
cat words.txt | tr -s ' ' '\n' | sort | uniq -c | sort -r | awk '{ print $2, $1 }'
```

- 解析

```

1 切割
tr 命令用于转换或删除文件中的字符
-s：缩减连续重复的字符成指定的单个字符

[]
cat Words.txt| tr -s ' ' '\n'

the
day
is
sunny
the
the
the
sunny
is
is
2 排序单词
[]
cat Words.txt| tr -s ' ' '\n' | sort

day
is
is
is
sunny
sunny
the
the
the
the
3 统计单词出现次数
uniq 命令用于检查及删除文本文件中重复出现的行列，一般与 sort 命令结合使用。
-c：在每列旁边显示该行重复出现的次数。

[]
cat Words.txt| tr -s ' ' '\n' | sort | uniq -c

1 day
3 is
2 sunny
4 the
4 排序单词出现次数
-r：以相反的顺序来排序

[]
cat Words.txt| tr -s ' ' '\n' | sort | uniq -c | sort -r

4 the
3 is
2 sunny
1 day
5 打印
[]
cat Words.txt| tr -s ' ' '\n' | sort | uniq -c | sort -r | awk '{print $2, $1}'

the 4
is 3
sunny 2
```


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
            # 2. 先尝试删除右括号，如果先删除左括号，会导致可能左边更多有括号没法匹配
            if r > 0:
                for i in range(start, len(s1)):
                    # 去重
                    if i != start and s1[i] == s1[i-1]:
                        continue
                    # 删除
                    if s1[i] == ')':
                        search(s1[:i]+s1[i+1:], l, r - 1, i)# 减去第i个元素了，下一个还是从i开始删
            elif l > 0:
                for i in range(start, len(s1)):
                    if i != start and s1[i] == s1[i-1]:
                        continue
                    if s1[i] == '(':
                        search(s1[:i]+s1[i+1:], l - 1, r, i)

        # 1. 先统计左右各需要删除多少个括号
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
#### todolist [273, 301, 314, 125, 238, 938, 56, 215, 1762, 1570, 31, 199]

#### 273. Integer to English Words

```python
# Convert a non-negative integer num to its English words representation. 
# 
#  
#  Example 1: 
# 
#  
# Input: num = 123
# Output: "One Hundred Twenty Three"
#  
# 
#  Example 2: 
# 
#  
# Input: num = 12345
# Output: "Twelve Thousand Three Hundred Forty Five"
#  
# 
#  Example 3: 
# 
#  
# Input: num = 1234567
# Output: "One Million Two Hundred Thirty Four Thousand Five Hundred Sixty 
# Seven"
#  
# 
#  
#  Constraints: 
# 
#  
#  0 <= num <= 2³¹ - 1 
#  
# 
#  Related Topics 递归 数学 字符串 👍 292 👎 0

class Solution(object):
    def numberToWords(self, num):
        """
        :type num: int
        :rtype: str
        """
        if num == 0:
            return "Zero"
        ones = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
        teens = ["Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
        tens = ["Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
        thousands = [" Thousand", " Million", " Billion"]
        def search(num):
            s = ""
            # 1. if num biger than base, use recursion to get the result and return result,
            # be careful when mode == 0
            for i, base in enumerate([1000000000, 1000000, 1000]):
                if num >= base:
                    s += self.numberToWords(num // base) + thousands[2 - i]
                    if num % base != 0:
                        s += " " + self.numberToWords(num % base)
                    return s
            # 2. if num slower than 1000, check if it at range in [0, 10, 20, 100]
            if num < 1000:
                if num >= 100:
                    if num % 100 == 0:
                        s += self.numberToWords(num // 100) + " Hundred"
                    else:
                        s += self.numberToWords(num // 100) + " Hundred " + self.numberToWords(num % 100)
                elif num >= 20:
                    i = num // 10
                    if num % 10 == 0:
                        s += tens[i - 2]
                    else:
                        s += tens[i - 2] + " " + ones[num % 10 - 1]
                elif num >= 10:
                    s += teens[num - 10]
                else:
                    s += ones[num - 1]
            return s
        return search(num)

        
# runtime:24 ms
# memory:13.2 MB

```

#### 301. Remove Invalid Parentheses

```python
# Given a string s that contains parentheses and letters, remove the minimum 
# number of invalid parentheses to make the input string valid. 
# 
#  Return all the possible results. You may return the answer in any order. 
# 
#  
#  Example 1: 
# 
#  
# Input: s = "()())()"
# Output: ["(())()","()()()"]
#  
# 
#  Example 2: 
# 
#  
# Input: s = "(a)())()"
# Output: ["(a())()","(a)()()"]
#  
# 
#  Example 3: 
# 
#  
# Input: s = ")("
# Output: [""]
#  
# 
#  
#  Constraints: 
# 
#  
#  1 <= s.length <= 25 
#  s consists of lowercase English letters and parentheses '(' and ')'. 
#  There will be at most 20 parentheses in s. 
#  
# 
#  Related Topics 广度优先搜索 字符串 回溯 👍 779 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution(object):
    
    def calBracketNumNotMatched(self, s):
        l = 0
        r = 0
        for i in range(len(s)):
            if s[i] == ')':
                if l > 0:
                    l -= 1
                else:
                    r += 1
            elif s[i] == '(':
                l += 1
        return l, r

    def isValid(self, s):
        l, r = self.calBracketNumNotMatched(s)
        return l == 0 and r == 0
    
    def removeInvalidParentheses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """


        total_l_need_remove, total_r_need_remove = self.calBracketNumNotMatched(s)

        res = []
        def search(cur_s, l, r, start):
            # print(cur_s, l, r, start)
            if l == 0 and r == 0 and self.isValid(cur_s):
                res.append(cur_s)
                return
            if r > 0:
                for i in range(start, len(cur_s)):
                    if i != start and cur_s[i] == cur_s[i-1]:
                        continue
                    if cur_s[i] == ')':
                        search(cur_s[:i]+cur_s[i+1:], l, r - 1, i)
            elif l > 0:
                for i in range(start, len(cur_s)):
                    if i != start and cur_s[i] == cur_s[i-1]:
                        continue
                    if cur_s[i] == '(':
                        search(cur_s[:i]+cur_s[i+1:], l - 1, r, i)
        search(s, total_l_need_remove, total_r_need_remove, 0)
        return res
# runtime:20 ms
# memory:13 MB

# leetcode submit region end(Prohibit modification and deletion)

```

#### 314. Binary Tree Vertical Order Traversal

```python
# Given the root of a binary tree, return the vertical order traversal of its 
# nodes' values. (i.e., from top to bottom, column by column). 
# 
#  If two nodes are in the same row and column, the order should be from left 
# to right. 
# 
#  
#  Example 1: 
#  
#  
# Input: root = [3,9,20,null,null,15,7]
# Output: [[9],[3,15],[20],[7]]
#  
# 
#  Example 2: 
#  
#  
# Input: root = [3,9,8,4,0,1,7]
# Output: [[4],[9],[3,0,1],[8],[7]]
#  
# 
#  Example 3: 
#  
#  
# Input: root = [3,9,8,4,0,1,7,null,null,null,2,5]
# Output: [[4],[9,5],[3,0,1],[8,2],[7]]
#  
# 
#  
#  Constraints: 
# 
#  
#  The number of nodes in the tree is in the range [0, 100]. 
#  -100 <= Node.val <= 100 
#  
# 
#  Related Topics 树 深度优先搜索 广度优先搜索 哈希表 二叉树 👍 190 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def verticalOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        d = deque()
        d.append([root, 0])
        loc_to_node_list = defaultdict(list)
        min_loc = 0
        while d:
            cur, cur_loc = d.popleft()
            loc_to_node_list[cur_loc].append(cur.val)
            if cur.left:
                d.append([cur.left, cur_loc - 1])
            if cur.right:
                d.append([cur.right, cur_loc + 1])
            min_loc = min(cur_loc, min_loc)
        res = []
        for loc in range(min_loc, min_loc + len(loc_to_node_list)):
            res.append(loc_to_node_list[loc])
        return res
# runtime:16 ms
# memory:13.3 MB

# leetcode submit region end(Prohibit modification and deletion)


```

#### 125. Valid Palindrome

```python
# A phrase is a palindrome if, after converting all uppercase letters into 
# lowercase letters and removing all non-alphanumeric characters, it reads the same 
# forward and backward. Alphanumeric characters include letters and numbers. 
# 
#  Given a string s, return true if it is a palindrome, or false otherwise. 
# 
#  
#  Example 1: 
# 
#  
# Input: s = "A man, a plan, a canal: Panama"
# Output: true
# Explanation: "amanaplanacanalpanama" is a palindrome.
#  
# 
#  Example 2: 
# 
#  
# Input: s = "race a car"
# Output: false
# Explanation: "raceacar" is not a palindrome.
#  
# 
#  Example 3: 
# 
#  
# Input: s = " "
# Output: true
# Explanation: s is an empty string "" after removing non-alphanumeric 
# characters.
# Since an empty string reads the same forward and backward, it is a palindrome.
# 
#  
# 
#  
#  Constraints: 
# 
#  
#  1 <= s.length <= 2 * 10⁵ 
#  s consists only of printable ASCII characters. 
#  
# 
#  Related Topics 双指针 字符串 👍 578 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        s_list = []
        for i in range(len(s)):
            if 'a'<=s[i]<='z' or 'A'<=s[i]<='Z' or '0'<=s[i]<='9':
                s_list.append(s[i].lower())
        # print('s_list:', s_list)
        l, r = 0, len(s_list) - 1
        while l < r:
            if s_list[l] == s_list[r]:
                l += 1
                r -= 1
            else:
                return False
        return True
        
# runtime:44 ms
# memory:17.9 MB

# leetcode submit region end(Prohibit modification and deletion)


```

#### [238]Product of Array Except Self

```python
# Given an integer array nums, return an array answer such that answer[i] is 
# equal to the product of all the elements of nums except nums[i]. 
# 
#  The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit 
# integer. 
# 
#  You must write an algorithm that runs in O(n) time and without using the 
# division operation. 
# 
#  
#  Example 1: 
#  Input: nums = [1,2,3,4]
# Output: [24,12,8,6]
#  
#  Example 2: 
#  Input: nums = [-1,1,0,-3,3]
# Output: [0,0,9,0,0]
#  
#  
#  Constraints: 
# 
#  
#  2 <= nums.length <= 10⁵ 
#  -30 <= nums[i] <= 30 
#  The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit 
# integer. 
#  
# 
#  
#  Follow up: Can you solve the problem in O(1) extra space complexity? (The 
# output array does not count as extra space for space complexity analysis.) 
# 
#  Related Topics 数组 前缀和 👍 1286 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """

        n = len(nums)
        l = [1] * n
        r = [1] * n

        for i in range(n):
            if i == 0:
                l[i] = nums[i]
                r[n-1-i] = nums[n-1-i]
            else:
                l[i] = l[i-1] * nums[i]
                r[n-1-i] = r[n-i] * nums[n-i-1]

        p = [0] * n
        for i in range(n):
            if i == 0:
                p[i] = r[i+1]
            elif i == n - 1:
                p[i] = l[i-1]
            else:
                p[i] = l[i-1] * r[i+1]
        return p
# runtime:60 ms
# memory:23.8 MB

# leetcode submit region end(Prohibit modification and deletion)


```

#### [938]Range Sum of BST

```python
# Given the root node of a binary search tree and two integers low and high, 
# return the sum of values of all nodes with a value in the inclusive range [low, 
# high]. 
# 
#  
#  Example 1: 
#  
#  
# Input: root = [10,5,15,3,7,null,18], low = 7, high = 15
# Output: 32
# Explanation: Nodes 7, 10, and 15 are in the range [7, 15]. 7 + 10 + 15 = 32.
#  
# 
#  Example 2: 
#  
#  
# Input: root = [10,5,15,3,7,13,18,1,null,6], low = 6, high = 10
# Output: 23
# Explanation: Nodes 6, 7, and 10 are in the range [6, 10]. 6 + 7 + 10 = 23.
#  
# 
#  
#  Constraints: 
# 
#  
#  The number of nodes in the tree is in the range [1, 2 * 10⁴]. 
#  1 <= Node.val <= 10⁵ 
#  1 <= low <= high <= 10⁵ 
#  All Node.val are unique. 
#  
# 
#  Related Topics 树 深度优先搜索 二叉搜索树 二叉树 👍 308 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def rangeSumBST(self, root, low, high):
        """
        :type root: TreeNode
        :type low: int
        :type high: int
        :rtype: int
        """
        self.sum = 0
        def search(node):
            if not node:
                return
            if node.val < low:
                search(node.right)
            elif node.val > high:
                search(node.left)
            else:
                self.sum += node.val
                search(node.left)
                search(node.right)
        search(root)
        return self.sum
# runtime:192 ms
# memory:28.9 MB

        
# leetcode submit region end(Prohibit modification and deletion)

```

#### [56]Merge Intervals
```python
# Given an array of intervals where intervals[i] = [starti, endi], merge all 
# overlapping intervals, and return an array of the non-overlapping intervals that 
# cover all the intervals in the input. 
# 
#  
#  Example 1: 
# 
#  
# Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
# Output: [[1,6],[8,10],[15,18]]
# Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].
#  
# 
#  Example 2: 
# 
#  
# Input: intervals = [[1,4],[4,5]]
# Output: [[1,5]]
# Explanation: Intervals [1,4] and [4,5] are considered overlapping.
#  
# 
#  
#  Constraints: 
# 
#  
#  1 <= intervals.length <= 10⁴ 
#  intervals[i].length == 2 
#  0 <= starti <= endi <= 10⁴ 
#  
# 
#  Related Topics 数组 排序 👍 1672 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        intervals.sort(key = lambda x:[x[0], -x[1]])
        res = []
        for i in range(len(intervals)):
            if not res:
                res.append(intervals[i])
            else:
                if intervals[i][0] > res[-1][1]:
                    res.append(intervals[i])
                else:
                    res[-1][1] = max(res[-1][1], intervals[i][1])
        return res
# runtime:36 ms
# memory:17.5 MB

# leetcode submit region end(Prohibit modification and deletion)

```

#### [215]Kth Largest Element in an Array

```python
# Given an integer array nums and an integer k, return the kᵗʰ largest element 
# in the array. 
# 
#  Note that it is the kᵗʰ largest element in the sorted order, not the kᵗʰ 
# distinct element. 
# 
#  You must solve it in O(n) time complexity. 
# 
#  
#  Example 1: 
#  Input: nums = [3,2,1,5,6,4], k = 2
# Output: 5
#  
#  Example 2: 
#  Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
# Output: 4
#  
#  
#  Constraints: 
# 
#  
#  1 <= k <= nums.length <= 10⁵ 
#  -10⁴ <= nums[i] <= 10⁴ 
#  
# 
#  Related Topics 数组 分治 快速选择 排序 堆（优先队列） 👍 1913 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """

        def search(l, r, k):
            mid = l + r >> 1
            nums[mid], nums[r] = nums[r], nums[mid]
            k1, k2 = l - 1, l
            while k2 < r:
                if nums[k2] > nums[r]:
                    k1 += 1
                    nums[k1], nums[k2] = nums[k2], nums[k1]
                k2 += 1
            k1 += 1
            nums[k1], nums[r] = nums[r], nums[k1]
            if k1 - l + 1 == k:
                return nums[k1]
            elif k1 - l + 1 > k:
                return search(l, k1 - 1, k)
            else:
                return search(k1 + 1, r, k - (k1 - l + 1))

        return search(0, len(nums) - 1, k)

# runtime:140 ms
# memory:25.4 MB

# leetcode submit region end(Prohibit modification and deletion)

```

#### [1762]Buildings With an Ocean View

```python
# There are n buildings in a line. You are given an integer array heights of 
# size n that represents the heights of the buildings in the line. 
# 
#  The ocean is to the right of the buildings. A building has an ocean view if 
# the building can see the ocean without obstructions. Formally, a building has an 
# ocean view if all the buildings to its right have a smaller height. 
# 
#  Return a list of indices (0-indexed) of buildings that have an ocean view, 
# sorted in increasing order. 
# 
#  
#  Example 1: 
# 
#  
# Input: heights = [4,2,3,1]
# Output: [0,2,3]
# Explanation: Building 1 (0-indexed) does not have an ocean view because 
# building 2 is taller.
#  
# 
#  Example 2: 
# 
#  
# Input: heights = [4,3,2,1]
# Output: [0,1,2,3]
# Explanation: All the buildings have an ocean view.
#  
# 
#  Example 3: 
# 
#  
# Input: heights = [1,3,2,4]
# Output: [3]
# Explanation: Only building 3 has an ocean view.
#  
# 
#  
#  Constraints: 
# 
#  
#  1 <= heights.length <= 10⁵ 
#  1 <= heights[i] <= 10⁹ 
#  
# 
#  Related Topics 栈 数组 单调栈 👍 17 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution(object):
    def findBuildings(self, h):
        """
        :type heights: List[int]
        :rtype: List[int]

        h=[2,2]
        m=2
        1
        0
        """
        n = len(h)
        m = h[-1] - 1
        tmp = []
        for i in range(n-1, -1, -1):
            if h[i] > m:
                tmp.append(i)
                m = h[i]
        res = []
        while tmp:
            res.append(tmp.pop())
        return res
# runtime:68 ms
# memory:28.7 MB

# leetcode submit region end(Prohibit modification and deletion)

```

#### [1570]Dot Product of Two Sparse Vectors

```python
# Given two sparse vectors, compute their dot product. 
# 
#  Implement class SparseVector: 
# 
#  
#  SparseVector(nums) Initializes the object with the vector nums 
#  dotProduct(vec) Compute the dot product between the instance of SparseVector 
# and vec 
#  
# 
#  A sparse vector is a vector that has mostly zero values, you should store 
# the sparse vector efficiently and compute the dot product between two SparseVector.
#  
# 
#  Follow up: What if only one of the vectors is sparse? 
# 
#  
#  Example 1: 
# 
#  
# Input: nums1 = [1,0,0,2,3], nums2 = [0,3,0,4,0]
# Output: 8
# Explanation: v1 = SparseVector(nums1) , v2 = SparseVector(nums2)
# v1.dotProduct(v2) = 1*0 + 0*3 + 0*0 + 2*4 + 3*0 = 8
#  
# 
#  Example 2: 
# 
#  
# Input: nums1 = [0,1,0,0,0], nums2 = [0,0,0,0,2]
# Output: 0
# Explanation: v1 = SparseVector(nums1) , v2 = SparseVector(nums2)
# v1.dotProduct(v2) = 0*0 + 1*0 + 0*0 + 0*0 + 0*2 = 0
#  
# 
#  Example 3: 
# 
#  
# Input: nums1 = [0,1,0,0,2,0,0], nums2 = [1,0,0,0,3,0,4]
# Output: 6
#  
# 
#  
#  Constraints: 
# 
#  
#  n == nums1.length == nums2.length 
#  1 <= n <= 10^5 
#  0 <= nums1[i], nums2[i] <= 100 
#  
# 
#  Related Topics 设计 数组 哈希表 双指针 👍 25 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class SparseVector:
    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        data = list()
        for i, num in enumerate(nums):
            if num != 0:
                data.append([i, num])
        self.data = data
        

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec):
        """
        :type vec: 'SparseVector'
        :rtype: int
        """
        s = 0
        m, n = len(self.data), len(vec.data)
        k1, k2 = 0, 0
        while k1 < m and k2 < n:
            if self.data[k1][0] == vec.data[k2][0]:
                s += self.data[k1][1] * vec.data[k2][1]
                k1 += 1
                k2 += 1
            elif self.data[k1][0] > vec.data[k2][0]:
                k2 += 1
            else:
                k1 += 1
        return s
        

# Your SparseVector object will be instantiated and called as such:
# v1 = SparseVector(nums1)
# v2 = SparseVector(nums2)
# ans = v1.dotProduct(v2)
# runtime:1512 ms
# memory:17.1 MB

        

# Your SparseVector object will be instantiated and called as such:
# v1 = SparseVector(nums1)
# v2 = SparseVector(nums2)
# ans = v1.dotProduct(v2)
# leetcode submit region end(Prohibit modification and deletion)

```

#### [31]Next Permutation

```python
# A permutation of an array of integers is an arrangement of its members into a 
# sequence or linear order. 
# 
#  
#  For example, for arr = [1,2,3], the following are all the permutations of 
# arr: [1,2,3], [1,3,2], [2, 1, 3], [2, 3, 1], [3,1,2], [3,2,1]. 
#  
# 
#  The next permutation of an array of integers is the next lexicographically 
# greater permutation of its integer. More formally, if all the permutations of the 
# array are sorted in one container according to their lexicographical order, 
# then the next permutation of that array is the permutation that follows it in the 
# sorted container. If such arrangement is not possible, the array must be 
# rearranged as the lowest possible order (i.e., sorted in ascending order). 
# 
#  
#  For example, the next permutation of arr = [1,2,3] is [1,3,2]. 
#  Similarly, the next permutation of arr = [2,3,1] is [3,1,2]. 
#  While the next permutation of arr = [3,2,1] is [1,2,3] because [3,2,1] does 
# not have a lexicographical larger rearrangement. 
#  
# 
#  Given an array of integers nums, find the next permutation of nums. 
# 
#  The replacement must be in place and use only constant extra memory. 
# 
#  
#  Example 1: 
# 
#  
# Input: nums = [1,2,3]
# Output: [1,3,2]
#  
# 
#  Example 2: 
# 
#  
# Input: nums = [3,2,1]
# Output: [1,2,3]
#  
# 
#  Example 3: 
# 
#  
# Input: nums = [1,1,5]
# Output: [1,5,1]
#  
# 
#  
#  Constraints: 
# 
#  
#  1 <= nums.length <= 100 
#  0 <= nums[i] <= 100 
#  
# 
#  Related Topics 数组 双指针 👍 1937 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution(object):
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """


        """

        l
           r
        3654321

        4654321
        """

        n = len(nums)
        if n == 1:
            return
        i = n - 2
        find = False
        while i >= 0:
            if nums[i] < nums[i+1]:
                find = True
                break
            i -= 1
        
        if not find:
            nums.sort()
            return
        

        def find_first_index_bigger_than_target(target, l, r):
            """find target in num at range [l, r]"""
            cand = l
            while l <= r:
                m = l + r >> 1
                # print(target, l, r, m, nums[m])
                if nums[m] > target:
                    cand = m
                    l = m + 1
                elif nums[m] < target:
                    r = m - 1
                else:
                    r = m - 1
            return cand

        first_index = find_first_index_bigger_than_target(nums[i], i+1, len(nums) - 1)
        nums[i], nums[first_index] = nums[first_index], nums[i]
        # print(nums)
        l, r = i + 1, len(nums) - 1
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l += 1
            r -= 1
        
        return

# runtime:20 ms
# memory:12.9 MB

# leetcode submit region end(Prohibit modification and deletion)

```

#### [199]Binary Tree Right Side View

```python

# Given the root of a binary tree, imagine yourself standing on the right side 
# of it, return the values of the nodes you can see ordered from top to bottom. 
# 
#  
#  Example 1: 
#  
#  
# Input: root = [1,2,3,null,5,null,4]
# Output: [1,3,4]
#  
# 
#  Example 2: 
# 
#  
# Input: root = [1,null,3]
# Output: [1,3]
#  
# 
#  Example 3: 
# 
#  
# Input: root = []
# Output: []
#  
# 
#  
#  Constraints: 
# 
#  
#  The number of nodes in the tree is in the range [0, 100]. 
#  -100 <= Node.val <= 100 
#  
# 
#  Related Topics 树 深度优先搜索 广度优先搜索 二叉树 👍 765 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res = []
        exist_layer = set()
        def dfs(root, layer):
            if not root:
                return
            if layer not in exist_layer:
                exist_layer.add(layer)
                res.append(root.val)
            dfs(root.right, layer + 1)
            dfs(root.left, layer + 1)
        dfs(root, 0)
        return res
# runtime:24 ms
# memory:13.1 MB

        
# leetcode submit region end(Prohibit modification and deletion)

```

#### todolist [528, 173, 415, 1, 29, 297, 278, 236, 253, 621, 91, 88, 543, 71, 50, 124, 10, 1650, 227, 138, 426, 133, 15, 158, 1428]

#### 528. Random Pick with Weight

```python
# You are given a 0-indexed array of positive integers w where w[i] describes 
# the weight of the iᵗʰ index. 
# 
#  You need to implement the function pickIndex(), which randomly picks an 
# index in the range [0, w.length - 1] (inclusive) and returns it. The probability of 
# picking an index i is w[i] / sum(w). 
# 
#  
#  For example, if w = [1, 3], the probability of picking index 0 is 1 / (1 + 3)
#  = 0.25 (i.e., 25%), and the probability of picking index 1 is 3 / (1 + 3) = 0.7
# 5 (i.e., 75%). 
#  
# 
#  
#  Example 1: 
# 
#  
# Input
# ["Solution","pickIndex"]
# [[[1]],[]]
# Output
# [null,0]
# 
# Explanation
# Solution solution = new Solution([1]);
# solution.pickIndex(); // return 0. The only option is to return 0 since there 
# is only one element in w.
#  
# 
#  Example 2: 
# 
#  
# Input
# ["Solution","pickIndex","pickIndex","pickIndex","pickIndex","pickIndex"]
# [[[1,3]],[],[],[],[],[]]
# Output
# [null,1,1,1,1,0]
# 
# Explanation
# Solution solution = new Solution([1, 3]);
# solution.pickIndex(); // return 1. It is returning the second element (index =
#  1) that has a probability of 3/4.
# solution.pickIndex(); // return 1
# solution.pickIndex(); // return 1
# solution.pickIndex(); // return 1
# solution.pickIndex(); // return 0. It is returning the first element (index = 
# 0) that has a probability of 1/4.
# 
# Since this is a randomization problem, multiple answers are allowed.
# All of the following outputs can be considered correct:
# [null,1,1,1,1,0]
# [null,1,1,1,1,1]
# [null,1,1,1,0,0]
# [null,1,1,1,0,1]
# [null,1,0,1,0,0]
# ......
# and so on.
#  
# 
#  
#  Constraints: 
# 
#  
#  1 <= w.length <= 10⁴ 
#  1 <= w[i] <= 10⁵ 
#  pickIndex will be called at most 10⁴ times. 
#  
# 
#  Related Topics 数学 二分查找 前缀和 随机化 👍 271 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
import random


class Solution(object):

    def __init__(self, w):
        """
        :type w: List[int]
        """
        s = float(sum(w))
        for i in range(len(w)):
            if i == 0:
                w[i] = w[i] / s
            else:
                w[i] = w[i-1] + w[i] / s
        self.w = w

    def pickIndex(self):
        """
        :rtype: int
        """
        val = random.random()
        if val <= self.w[0]:
            return 0
        l, r = 0, len(self.w) - 1
        while l <= r:
            m = l + r >> 1
            if self.w[m-1] < val <= self.w[m]:
                return m
            elif val > self.w[m]:
                l = m + 1
            else:
                r = m - 1



# Your Solution object will be instantiated and called as such:
# obj = Solution(w)
# param_1 = obj.pickIndex()
# leetcode submit region end(Prohibit modification and deletion)
```

#### 173. Binary Search Tree Iterator

```python
# Implement the BSTIterator class that represents an iterator over the in-order 
# traversal of a binary search tree (BST): 
# 
#  
#  BSTIterator(TreeNode root) Initializes an object of the BSTIterator class. 
# The root of the BST is given as part of the constructor. The pointer should be 
# initialized to a non-existent number smaller than any element in the BST. 
#  boolean hasNext() Returns true if there exists a number in the traversal to 
# the right of the pointer, otherwise returns false. 
#  int next() Moves the pointer to the right, then returns the number at the 
# pointer. 
#  
# 
#  Notice that by initializing the pointer to a non-existent smallest number, 
# the first call to next() will return the smallest element in the BST. 
# 
#  You may assume that next() calls will always be valid. That is, there will 
# be at least a next number in the in-order traversal when next() is called. 
# 
#  
#  Example 1: 
#  
#  
# Input
# ["BSTIterator", "next", "next", "hasNext", "next", "hasNext", "next", 
# "hasNext", "next", "hasNext"]
# [[[7, 3, 15, null, null, 9, 20]], [], [], [], [], [], [], [], [], []]
# Output
# [null, 3, 7, true, 9, true, 15, true, 20, false]
#  
# 
# Explanation
# BSTIterator bSTIterator = new BSTIterator([7, 3, 15, null, null, 9, 20]);
# bSTIterator.next(); // return 3
# bSTIterator.next(); // return 7
# bSTIterator.hasNext(); // return True
# bSTIterator.next(); // return 9
# bSTIterator.hasNext(); // return True
# bSTIterator.next(); // return 15
# bSTIterator.hasNext(); // return True
# bSTIterator.next(); // return 20
# bSTIterator.hasNext(); // return False
# 
# 
#  
#  Constraints: 
# 
#  
#  The number of nodes in the tree is in the range [1, 10⁵]. 
#  0 <= Node.val <= 10⁶ 
#  At most 10⁵ calls will be made to hasNext, and next. 
#  
# 
#  
#  Follow up: 
# 
#  
#  Could you implement next() and hasNext() to run in average O(1) time and use 
# O(h) memory, where h is the height of the tree? 
#  
# 
#  Related Topics 栈 树 设计 二叉搜索树 二叉树 迭代器 👍 648 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class BSTIterator(object):
    """
    思路：中序遍历
    1. 先将当前节点的所有左子树压入栈，压到没有为止
    2. 将最后一个压入的节点弹出（栈顶元素），加入答案
    3. 将当前弹出的节点作为当前节点，重复步骤一
    """

    def __init__(self, root):
        """
        :type root: TreeNode
        """
        self.head = root
        self.stack = []
        while root:
            self.stack.append(root)
            root = root.left

    def next(self):
        """
        :rtype: int
        """
        cur = self.stack.pop()
        root = cur.right
        while root:
            self.stack.append(root)
            root = root.left
        return cur.val
        

    def hasNext(self):
        """
        :rtype: bool
        """
        return len(self.stack) > 0
        


# Your BSTIterator object will be instantiated and called as such:
# obj = BSTIterator(root)
# param_1 = obj.next()
# param_2 = obj.hasNext()
# leetcode submit region end(Prohibit modification and deletion)


```

#### 415. Add Strings


```pyhton
# Given two non-negative integers, num1 and num2 represented as string, return 
# the sum of num1 and num2 as a string. 
# 
#  You must solve the problem without using any built-in library for handling 
# large integers (such as BigInteger). You must also not convert the inputs to 
# integers directly. 
# 
#  
#  Example 1: 
# 
#  
# Input: num1 = "11", num2 = "123"
# Output: "134"
#  
# 
#  Example 2: 
# 
#  
# Input: num1 = "456", num2 = "77"
# Output: "533"
#  
# 
#  Example 3: 
# 
#  
# Input: num1 = "0", num2 = "0"
# Output: "0"
#  
# 
#  
#  Constraints: 
# 
#  
#  1 <= num1.length, num2.length <= 10⁴ 
#  num1 and num2 consist of only digits. 
#  num1 and num2 don't have any leading zeros except for the zero itself. 
#  
# 
#  Related Topics 数学 字符串 模拟 👍 625 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution(object):
    def addStrings(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        m, n = len(num1), len(num2)
        add = 0
        k1, k2 = m - 1, n - 1
        res = []
        while k1 >= 0 or k2 >= 0:
            a = 0 if k1 < 0 else int(num1[k1])
            b = 0 if k2 < 0 else int(num2[k2])
            k1 -= 1
            k2 -= 1
            cur = a + b + add
            add = cur // 10
            cur = cur % 10
            res.append(str(cur))
        if add:
            res.append(str(add))
        return ''.join(res[::-1])


# leetcode submit region end(Prohibit modification and deletion)


```


#### 29 Divide Two Integers

```python
# Given two integers dividend and divisor, divide two integers without using 
# multiplication, division, and mod operator. 
# 
#  The integer division should truncate toward zero, which means losing its 
# fractional part. For example, 8.345 would be truncated to 8, and -2.7335 would be 
# truncated to -2. 
# 
#  Return the quotient after dividing dividend by divisor. 
# 
#  Note: Assume we are dealing with an environment that could only store 
# integers within the 32-bit signed integer range: [−2³¹, 2³¹ − 1]. For this problem, if 
# the quotient is strictly greater than 2³¹ - 1, then return 2³¹ - 1, and if the 
# quotient is strictly less than -2³¹, then return -2³¹. 
# 
#  
#  Example 1: 
# 
#  
# Input: dividend = 10, divisor = 3
# Output: 3
# Explanation: 10/3 = 3.33333.. which is truncated to 3.
#  
# 
#  Example 2: 
# 
#  
# Input: dividend = 7, divisor = -3
# Output: -2
# Explanation: 7/-3 = -2.33333.. which is truncated to -2.
#  
# 
#  
#  Constraints: 
# 
#  
#  -2³¹ <= dividend, divisor <= 2³¹ - 1 
#  divisor != 0 
#  
# 
#  Related Topics 位运算 数学 👍 991 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution(object):
    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        10 / 3
        2^0, 2^1, 2^2
        3, 3+3, 3+3+3+3,
        base = 3
        cnt = 1
        """
        neg_flag = True if ((dividend > 0 and divisor <0) or (dividend < 0 and divisor > 0)) else False
        dividend = -dividend if dividend < 0 else dividend
        divisor = -divisor if divisor < 0 else divisor
        res = 0
        while dividend >= divisor:
            base = divisor
            cnt = 1
            while dividend - base > base:
                base += base
                cnt += cnt
            dividend -= base
            res += cnt
        if neg_flag:
            res = -res
        max_int = (1 << 31) - 1
        min_int = -(1 << 31)
        if res > max_int:
            return max_int
        if res < min_int:
            return min_int
        return res



# leetcode submit region end(Prohibit modification and deletion)


```

#### 297 Serialize and Deserialize Binary Tree

```python
# Serialization is the process of converting a data structure or object into a 
# sequence of bits so that it can be stored in a file or memory buffer, or 
# transmitted across a network connection link to be reconstructed later in the same or 
# another computer environment. 
# 
#  Design an algorithm to serialize and deserialize a binary tree. There is no 
# restriction on how your serialization/deserialization algorithm should work. You 
# just need to ensure that a binary tree can be serialized to a string and this 
# string can be deserialized to the original tree structure. 
# 
#  Clarification: The input/output format is the same as how LeetCode 
# serializes a binary tree. You do not necessarily need to follow this format, so please be 
# creative and come up with different approaches yourself. 
# 
#  
#  Example 1: 
#  
#  
# Input: root = [1,2,3,null,null,4,5]
# Output: [1,2,3,null,null,4,5]
#  
# 
#  Example 2: 
# 
#  
# Input: root = []
# Output: []
#  
# 
#  
#  Constraints: 
# 
#  
#  The number of nodes in the tree is in the range [0, 10⁴]. 
#  -1000 <= Node.val <= 1000 
#  
# 
#  Related Topics 树 深度优先搜索 广度优先搜索 设计 字符串 二叉树 👍 986 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
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
            return ""
        path = []
        d = deque()
        d.append(root)
        while d:
            cur = d.popleft()
            if cur:
                path.append(str(cur.val))
                d.append(cur.left)
                d.append(cur.right)
            else:
                path.append("#")
        res = ','.join(path)
        # print(res)
        return res

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        if not data:
            return None
        data = data.split(',')
        n = len(data)
        i = 0
        root = TreeNode(data[i])
        i += 1
        d = deque()
        d.append(root)
        while d:
            cur = d.popleft()
            if data[i] != '#':
                left = TreeNode(data[i])
                i += 1
                cur.left = left
                d.append(left)
            else:
                i += 1
            if data[i] != '#':
                right = TreeNode(data[i])
                i += 1
                cur.right = right
                d.append(right)
            else:
                i += 1
        # 1,2,3,null,null,4,5
        return root



        

# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))
# leetcode submit region end(Prohibit modification and deletion)


```

#### 278 First Bad Version

```python
# You are a product manager and currently leading a team to develop a new 
# product. Unfortunately, the latest version of your product fails the quality check. 
# Since each version is developed based on the previous version, all the versions 
# after a bad version are also bad. 
# 
#  Suppose you have n versions [1, 2, ..., n] and you want to find out the 
# first bad one, which causes all the following ones to be bad. 
# 
#  You are given an API bool isBadVersion(version) which returns whether 
# version is bad. Implement a function to find the first bad version. You should 
# minimize the number of calls to the API. 
# 
#  
#  Example 1: 
# 
#  
# Input: n = 5, bad = 4
# Output: 4
# Explanation:
# call isBadVersion(3) -> false
# call isBadVersion(5) -> true
# call isBadVersion(4) -> true
# Then 4 is the first bad version.
#  
# 
#  Example 2: 
# 
#  
# Input: n = 1, bad = 1
# Output: 1
#  
# 
#  
#  Constraints: 
# 
#  
#  1 <= bad <= n <= 2³¹ - 1 
#  
# 
#  Related Topics 二分查找 交互 👍 809 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):

class Solution(object):
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """

        l, r = 1, n
        cand = l
        while l <= r:
            mid = l + r >> 1
            if isBadVersion(mid):
                cand = mid
                r = mid - 1
            else:
                l = mid + 1
        return cand
# leetcode submit region end(Prohibit modification and deletion)


```

#### 236 Lowest Common Ancestor of a Binary Tree

```python
# Given a binary tree, find the lowest common ancestor (LCA) of two given nodes 
# in the tree. 
# 
#  According to the definition of LCA on Wikipedia: “The lowest common ancestor 
# is defined between two nodes p and q as the lowest node in T that has both p 
# and q as descendants (where we allow a node to be a descendant of itself).” 
# 
#  
#  Example 1: 
#  
#  
# Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
# Output: 3
# Explanation: The LCA of nodes 5 and 1 is 3.
#  
# 
#  Example 2: 
#  
#  
# Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
# Output: 5
# Explanation: The LCA of nodes 5 and 4 is 5, since a node can be a descendant 
# of itself according to the LCA definition.
#  
# 
#  Example 3: 
# 
#  
# Input: root = [1,2], p = 1, q = 2
# Output: 1
#  
# 
#  
#  Constraints: 
# 
#  
#  The number of nodes in the tree is in the range [2, 10⁵]. 
#  -10⁹ <= Node.val <= 10⁹ 
#  All Node.val are unique. 
#  p != q 
#  p and q will exist in the tree. 
#  
# 
#  Related Topics 树 深度优先搜索 二叉树 👍 1997 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

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
        l = self.lowestCommonAncestor(root.left, p, q)
        r = self.lowestCommonAncestor(root.right, p, q)
        if l and r:
            return root
        if l:
            return l
        return r
        
# leetcode submit region end(Prohibit modification and deletion)


```

#### 253 Meeting Rooms II

```python
# Given an array of meeting time intervals intervals where intervals[i] = [
# starti, endi], return the minimum number of conference rooms required. 
# 
#  
#  Example 1: 
#  Input: intervals = [[0,30],[5,10],[15,20]]
# Output: 2
#  
#  Example 2: 
#  Input: intervals = [[7,10],[2,4]]
# Output: 1
#  
#  
#  Constraints: 
# 
#  
#  1 <= intervals.length <= 10⁴ 
#  0 <= starti < endi <= 10⁶ 
#  
# 
#  Related Topics 贪心 数组 双指针 前缀和 排序 堆（优先队列） 👍 479 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution(object):
    def minMeetingRooms(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        intervals = [[0,30],[5,10],[15,20]]
        第一个人从0上车，从30下车；
        第二个人从5上车，10下车。。。
        人数 1    2     1     2     1      0
             0----5----10----15----20-----30
        变化 +1   +1    -1    +1    -1    -1
        """
        res = []
        for x, y in intervals:
            res.append([x, 1])
            res.append([y, -1])
        max_cnt = 0
        cnt = 0
        res.sort(key = lambda x:x[0])
        # [[1,1],[13,1],[13,-1],[15,-1]]
        pre_time = res[0][0]
        for time, mark in res:
            if time != pre_time:
                max_cnt = max(max_cnt, cnt)
                pre_time = time
            cnt += mark
        max_cnt = max(max_cnt, cnt)
        return max_cnt




# leetcode submit region end(Prohibit modification and deletion)

```


#### 621. Task Scheduler

```python
# Given a characters array tasks, representing the tasks a CPU needs to do, 
# where each letter represents a different task. Tasks could be done in any order. 
# Each task is done in one unit of time. For each unit of time, the CPU could 
# complete either one task or just be idle. 
# 
#  However, there is a non-negative integer n that represents the cooldown 
# period between two same tasks (the same letter in the array), that is that there 
# must be at least n units of time between any two same tasks. 
# 
#  Return the least number of units of times that the CPU will take to finish 
# all the given tasks. 
# 
#  
#  Example 1: 
# 
#  
# Input: tasks = ["A","A","A","B","B","B"], n = 2
# Output: 8
# Explanation: 
# A -> B -> idle -> A -> B -> idle -> A -> B
# There is at least 2 units of time between any two same tasks.
#  
# 
#  Example 2: 
# 
#  
# Input: tasks = ["A","A","A","B","B","B"], n = 0
# Output: 6
# Explanation: On this case any permutation of size 6 would work since n = 0.
# ["A","A","A","B","B","B"]
# ["A","B","A","B","A","B"]
# ["B","B","B","A","A","A"]
# ...
# And so on.
#  
# 
#  Example 3: 
# 
#  
# Input: tasks = ["A","A","A","A","A","A","B","C","D","E","F","G"], n = 2
# Output: 16
# Explanation: 
# One possible solution is
# A -> B -> C -> A -> D -> E -> A -> F -> G -> A -> idle -> idle -> A -> idle ->
#  idle -> A
#  
# 
#  
#  Constraints: 
# 
#  
#  1 <= task.length <= 10⁴ 
#  tasks[i] is upper-case English letter. 
#  The integer n is in the range [0, 100]. 
#  
# 
#  Related Topics 贪心 数组 哈希表 计数 排序 堆（优先队列） 👍 1032 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution(object):
    def leastInterval(self, tasks, n):
        """
        :type tasks: List[str]
        :type n: int
        :rtype: int
        """
        from collections import defaultdict
        wc = defaultdict(int)
        mc = 0
        for t in tasks:
            wc[t] += 1
            if wc[t] > mc:
                mc = wc[t]

        mcc = 0
        for k, c in wc.items():
            if c == mc:
                mcc += 1

        return max(len(tasks), (mc - 1) * (n + 1) + mcc)

# leetcode submit region end(Prohibit modification and deletion)

```


#### 91. Decode Ways

```python
# A message containing letters from A-Z can be encoded into numbers using the 
# following mapping: 
# 
#  
# 'A' -> "1"
# 'B' -> "2"
# ...
# 'Z' -> "26"
#  
# 
#  To decode an encoded message, all the digits must be grouped then mapped 
# back into letters using the reverse of the mapping above (there may be multiple 
# ways). For example, "11106" can be mapped into: 
# 
#  
#  "AAJF" with the grouping (1 1 10 6) 
#  "KJF" with the grouping (11 10 6) 
#  
# 
#  Note that the grouping (1 11 06) is invalid because "06" cannot be mapped 
# into 'F' since "6" is different from "06". 
# 
#  Given a string s containing only digits, return the number of ways to decode 
# it. 
# 
#  The test cases are generated so that the answer fits in a 32-bit integer. 
# 
#  
#  Example 1: 
# 
#  
# Input: s = "12"
# Output: 2
# Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).
#  
# 
#  Example 2: 
# 
#  
# Input: s = "226"
# Output: 3
# Explanation: "226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2
#  6).
#  
# 
#  Example 3: 
# 
#  
# Input: s = "06"
# Output: 0
# Explanation: "06" cannot be mapped to "F" because of the leading zero ("6" is 
# different from "06").
#  
# 
#  
#  Constraints: 
# 
#  
#  1 <= s.length <= 100 
#  s contains only digits and may contain leading zero(s). 
#  
# 
#  Related Topics 字符串 动态规划 👍 1274 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int

        12322
        11106
        """
        n = len(s)
        if s[0] == '0':
            return 0
        if n == 1:
            return 1
        pre1 = 1
        if s[1] == '0':
            if '1' <= s[0] <= '2':
                pre2 = 1
            else:
                return 0
        else:
            if "11" <= s[:2] <= "26":
                pre2 = 2
            else:
                pre2 = 1

        for i in range(2, n):
            if s[i] == '0':
                if s[i-1] == '0' or s[i-1] > '2':
                    return 0
                pre1, pre2 = pre2, pre1
            else:
                if "11" <= s[i-1:i+1] <= "26":
                    pre1, pre2 = pre2, pre1 + pre2
                else:
                    pre1, pre2 = pre2, pre2
        return pre2




# leetcode submit region end(Prohibit modification and deletion)


```

#### 88. Merge Sorted Array

```python
# You are given two integer arrays nums1 and nums2, sorted in non-decreasing 
# order, and two integers m and n, representing the number of elements in nums1 and 
# nums2 respectively. 
# 
#  Merge nums1 and nums2 into a single array sorted in non-decreasing order. 
# 
#  The final sorted array should not be returned by the function, but instead 
# be stored inside the array nums1. To accommodate this, nums1 has a length of m + 
# n, where the first m elements denote the elements that should be merged, and the 
# last n elements are set to 0 and should be ignored. nums2 has a length of n. 
# 
#  
#  Example 1: 
# 
#  
# Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
# Output: [1,2,2,3,5,6]
# Explanation: The arrays we are merging are [1,2,3] and [2,5,6].
# The result of the merge is [1,2,2,3,5,6] with the underlined elements coming 
# from nums1.
#  
# 
#  Example 2: 
# 
#  
# Input: nums1 = [1], m = 1, nums2 = [], n = 0
# Output: [1]
# Explanation: The arrays we are merging are [1] and [].
# The result of the merge is [1].
#  
# 
#  Example 3: 
# 
#  
# Input: nums1 = [0], m = 0, nums2 = [1], n = 1
# Output: [1]
# Explanation: The arrays we are merging are [] and [1].
# The result of the merge is [1].
# Note that because m = 0, there are no elements in nums1. The 0 is only there 
# to ensure the merge result can fit in nums1.
#  
# 
#  
#  Constraints: 
# 
#  
#  nums1.length == m + n 
#  nums2.length == n 
#  0 <= m, n <= 200 
#  1 <= m + n <= 200 
#  -10⁹ <= nums1[i], nums2[j] <= 10⁹ 
#  
# 
#  
#  Follow up: Can you come up with an algorithm that runs in O(m + n) time? 
# 
#  Related Topics 数组 双指针 排序 👍 1598 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """

        k1, k2 = 0, 0
        for i in range(m-1, -1, -1):
            nums1[i+n] = nums1[i]

        i = 0
        while k1 < m and k2 < n:
            if nums1[k1+n] == nums2[k2]:
                nums1[i] = nums1[k1+n]
                i += 1
                nums1[i] = nums2[k2]
                i += 1
                k1 += 1
                k2 += 1
            elif nums1[k1+n] > nums2[k2]:
                nums1[i] = nums2[k2]
                k2 += 1
                i += 1
            else:
                nums1[i] = nums1[k1+n]
                k1 += 1
                i += 1

        while k1 < m:
            nums1[i] = nums1[k1+n]
            i += 1
            k1 += 1
        while k2 < n:
            nums1[i] = nums2[k2]
            i += 1
            k2 += 1


s = Solution()
a = [1,2,3,0,0,0]
m =			3
b =			[2,5,6]
n =			3
print(s.merge(a, m, b, n))
# leetcode submit region end(Prohibit modification and deletion)


```

#### 543. Diameter of Binary Tree

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


#### 50. Pow(x, n)

```
Implement pow(x, n), which calculates x raised to the power n (i.e., xn).

 

Example 1:

Input: x = 2.00000, n = 10
Output: 1024.00000
Example 2:

Input: x = 2.10000, n = 3
Output: 9.26100
Example 3:

Input: x = 2.00000, n = -2
Output: 0.25000
Explanation: 2-2 = 1/22 = 1/4 = 0.25

```

```python
class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        2**(10) = 2 ** (1010)
        1  2.      4            8
        2, 2*2=4, 2*2*2*2=16, (2*2*2*2)*(2*2*2*2)=16*16
        """
        if n < 0:
            return 1.0 / self.myPow(x, -n)
        i = 0
        s = 1
        while (1 << i) <= n:
            # print('x:',x,"1<<i:",1<<i)
            if (1 << i) & n:
                s *= x
            x *= x
            i += 1
        return s
```

#### 71. Simplify Path

```
Given a string path, which is an absolute path (starting with a slash '/') to a file or directory in a Unix-style file system, convert it to the simplified canonical path.

In a Unix-style file system, a period '.' refers to the current directory, a double period '..' refers to the directory up a level, and any multiple consecutive slashes (i.e. '//') are treated as a single slash '/'. For this problem, any other format of periods such as '...' are treated as file/directory names.

The canonical path should have the following format:

The path starts with a single slash '/'.
Any two directories are separated by a single slash '/'.
The path does not end with a trailing '/'.
The path only contains the directories on the path from the root directory to the target file or directory (i.e., no period '.' or double period '..')
Return the simplified canonical path.

 

Example 1:

Input: path = "/home/"
Output: "/home"
Explanation: Note that there is no trailing slash after the last directory name.
Example 2:

Input: path = "/../"
Output: "/"
Explanation: Going one level up from the root directory is a no-op, as the root level is the highest level you can go.
Example 3:

Input: path = "/home//foo/"
Output: "/home/foo"
Explanation: In the canonical path, multiple consecutive slashes are replaced by a single one.

```

```python

class Solution(object):
    def simplifyPath(self, path):
        """
        :type path: str
        :rtype: str
        """
        path = path.split('/')
        stack = []
        for val in path:
            if val:
                if val == '..':
                    if stack:
                        stack.pop()
                elif val != '.':
                    stack.append(val)
        return '/' + '/'.join(stack)
```

#### 10. Regular Expression Matching 错误解法

```python
class Solution(object):
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool


        ab a.
        a a* dp[i][j] = True if dp[i][j-1]
        aa a* dp[i][j] = True if dp[i-1][j] and p[j-2] == s[i-1]
        aaa a*
        """
        m, n = len(s), len(p)
        dp = [[False] * (n+1) for _ in range(m+1)]
        dp[0][0] = True
        if n >= 2 and p[1] == '*':
            dp[0][1] = True
        for j in range(3, n+1):
            if dp[0][j-2] and p[j-1] == '*':
                dp[0][j] = True
            # else:
                # break
        for i in range(1, m+1):
            for j in range(1, n+1):
                if s[i-1] == p[j-1] or p[j-1] == '.':
                    if dp[i-1][j-1]:
                        dp[i][j] = True
                elif p[j-1] == '*':
                    if j >= 2 and dp[i][j-2]:
                        dp[i][j] = True
                    if dp[i][j-1]:
                        dp[i][j] = True
                    if j >= 2 and dp[i-1][j] and (p[j-2] == s[i-1] or p[j-2] == '.'):
                        dp[i][j] = True

        for dpi in dp:
            print(dpi)
        return dp[m][n]

```

#### 10. Regular Expression Matching 正确解法

```python
class Solution(object):
    def isMatch(self, s, p):
        """正则表达式匹配
        :type s: str
        :type p: str
        :rtype: bool
        Example:
            Input: s = "ab", p = ".*"
            Output: true
        Solution:
            1. 完全相等或者遇到.往下匹配
            2. 没有*不可以匹配了，遇到*，考虑匹配一次或者两次，所以遇到*，还是要看前面的字母
            3. 总体框架，
                1. 是否等价或者为.
                2. 为*
                    匹配0个
                    匹配1个
                    匹配多个
        """

        m = len(s)
        n = len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        # 初始化
        dp[0][0] = True
        for j in range(1, n+1):
            if j-2 >= 0 and p[j-1] == '*' and dp[0][j-2]:
                dp[0][j] = True

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i-1] == p[j-1] or p[j-1] == '.':
                    dp[i][j] = dp[i-1][j-1]
                else:
                    if p[j-1] == "*":
                        # 匹配0个：b -> ba*
                        if j-2 >= 0 and dp[i][j-2]:
                            dp[i][j] = True
                        # 1个：a -> a*
                        if j-1 >= 0 and dp[i][j-1]:
                            dp[i][j] = True
                        # 匹配1个或者多个, 需要前面的相等
                        # baa vs ba*
                        # baaa vs ba*
                        if j-2>=0 and (p[j-2] == s[i-1] or p[j-2] == '.') \
                            and dp[i-1][j]:
                            dp[i][j] = True



        # for dpi in dp:
        #     print(dpi)

        return dp[m][n]

```

#### 10. Regular Expression Matching 简化版本

```python
class Solution(object):
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool


        ab a.
        a a* dp[i][j] = True if dp[i][j-1]
        aa a* dp[i][j] = True if dp[i-1][j] and p[j-2] == s[i-1]
        aaa a*
        """
        m, n = len(s), len(p)
        dp = [[False] * (n+1) for _ in range(m+1)]
        dp[0][0] = True
        for j in range(2, n + 1):
            if p[j-1] == '*':
                dp[0][j] = dp[0][j-2]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if s[i-1] == p[j-1] or p[j-1] == '.':
                    dp[i][j] = dp[i-1][j-1]
                elif p[j-1] == '*':
                    if j >= 2 and s[i-1] != p[j-2] and p[j-2] != '.':
                        dp[i][j] = dp[i][j-2]
                    elif j >= 2:
                        dp[i][j] = dp[i][j-2] | dp[i][j-2] | dp[i-1][j]
        # for dpi in dp:
        #     print(dpi)
        return dp[m][n]

```

#### todolist [76, 269, 139, 523, 23, 200, 283, 65, 211, 347, 34, 721, 42, 339, 2, 987, 636, 121, 146, 162, 282, 17, 986, 43, 140]





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
5. 发一下之前面过的FB E5面经吧：https://www.uscardforum.com/t/topic/28625
6. How I cracked my MLE interview at Facebook：https://towardsdatascience.com/how-i-cracked-my-mle-interview-at-facebook-fe55726f0096
7. 整理最近3个月Facebook面筋(2020):https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=698494&ctid=230547