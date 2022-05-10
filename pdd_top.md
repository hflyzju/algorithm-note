
#### 381 O(1)时间的插入，删除，随机选择数据结构实现

```python
from collections import defaultdict
import random

class RandomizedCollection(object):

    def __init__(self):

        self.cache = []
        self.val_to_indexlist = defaultdict(list)


    def insert(self, val):
        """
        :type val: int
        :rtype: bool
        """
        self.cache.append(val)
        self.val_to_indexlist[val].append(len(self.cache) - 1)
        # print('self.cache:', self.cache)
        return len(self.val_to_indexlist[val]) == 1



    def remove(self, val):
        """
        :type val: int
        :rtype: bool
        """
        if len(self.val_to_indexlist[val]) > 0:
            val_index = self.val_to_indexlist[val][-1]
            last_index = len(self.cache) - 1
            last_val = self.cache[last_index]
            # self.val_to_indexlist[last_val].pop()
            self.val_to_indexlist[last_val].remove(last_index)
            self.val_to_indexlist[last_val].append(val_index)
            try:
                self.cache[val_index], self.cache[last_index] = self.cache[last_index], self.cache[val_index]
            except:
                import pdb;pdb.set_trace()
            self.cache.pop()
            self.val_to_indexlist[val].pop()
            # print('self.cache:', self.cache)
            return True
        # print('self.cache:', self.cache)
        return False


    def getRandom(self):
        """
        :rtype: int
        """
        # print('self.cache:', self.cache)
        index = random.randint(0, len(self.cache)-1)
        # print('index:', index, 'len:', len(self.cache))
        return self.cache[index]



# Your RandomizedCollection object will be instantiated and called as such:
# obj = RandomizedCollection()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()

# runtime:1472 ms
# memory:68.8 MB


```


#### 123 卖两次股票，最大收益

```python
# You are given an array prices where prices[i] is the price of a given stock on
#  the ith day. 
# 
#  Find the maximum profit you can achieve. You may complete at most two transac
# tions. 
# 
#  Note: You may not engage in multiple transactions simultaneously (i.e., you m
# ust sell the stock before you buy again). 
# 
#  
#  Example 1: 
# 
#  
# Input: prices = [3,3,5,0,0,3,1,4]
# Output: 6
# Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit = 
# 3-0 = 3.
# Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 = 3.
#  
# 
#  Example 2: 
# 
#  
# Input: prices = [1,2,3,4,5]
# Output: 4
# Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 
# 5-1 = 4.
# Note that you cannot buy on day 1, buy on day 2 and sell them later, as you ar
# e engaging multiple transactions at the same time. You must sell before buying a
# gain.
#  
# 
#  Example 3: 
# 
#  
# Input: prices = [7,6,4,3,1]
# Output: 0
# Explanation: In this case, no transaction is done, i.e. max profit = 0.
#  
# 
#  
#  Constraints: 
# 
#  
#  1 <= prices.length <= 105 
#  0 <= prices[i] <= 105 
#  
#  Related Topics 数组 动态规划 
#  👍 1112 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int

#  Example 1:
#
#
# Input: prices = [3,3,5,0,0,3,1,4]
# Output: 6
# Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit =
# 3-0 = 3.
# Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 = 3.

       k1  k2
             l1 l2
[3,3,5,0,0,3,1,4]

dp[0][1][0] = 0
dp[0][1][1] = -prices[0]
dp[0][2][0] = 0
dp[0][2][1] = -prices[0]


        无股票: dp[day_i][time][0] : max(dp[day_i - 1][time][0], dp[day_i - 1][time][1] + prices[i]]
        有股票: dp[day_i][time][1] : max(dp[day_i - 1][time][1], dp[day_i - 1][time-1][0] - prices[i]]
        """
        n = len(prices)
        if n <= 1:
            return 0
        dp = [[[0, 0] for __ in range(3)] for _ in range(n)]
        # for dpi in dp:
        #     print(dpi)

        # dp[0][0][0] = 0
        # dp[0][0][1] = -prices[0]
        dp[0][1][0] = 0
        dp[0][1][1] = -prices[0]
        dp[0][2][0] = 0
        dp[0][2][1] = -prices[0]

        for day_i in range(1, n):
            for time in range(1, 3):
                dp[day_i][time][0] = max(dp[day_i - 1][time][0], dp[day_i - 1][time][1] + prices[day_i])
                dp[day_i][time][1] = max(dp[day_i - 1][time][1], dp[day_i - 1][time - 1][0] - prices[day_i])
        # for dpi in dp:
        #     print(dpi)
        return dp[n-1][2][0]


# leetcode submit region end(Prohibit modification and deletion)


prices = [3,3,5,0,0,3,1,4]

s = Solution()
print(s.maxProfit(prices))


```


#### 124 二叉树的最大path和

```python
# A path in a binary tree is a sequence of nodes where each pair of adjacent nod
# es in the sequence has an edge connecting them. A node can only appear in the se
# quence at most once. Note that the path does not need to pass through the root. 
# 
# 
#  The path sum of a path is the sum of the node's values in the path. 
# 
#  Given the root of a binary tree, return the maximum path sum of any non-empty
#  path. 
# 
#  
#  Example 1: 
# 
#  
# Input: root = [1,2,3]
# Output: 6
# Explanation: The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6.
# 
#  
# 
#  Example 2: 
# 
#  
# Input: root = [-10,9,20,null,null,15,7]
# Output: 42
# Explanation: The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7 
# = 42.
#  
# 
#  
#  Constraints: 
# 
#  
#  The number of nodes in the tree is in the range [1, 3 * 104]. 
#  -1000 <= Node.val <= 1000 
#  
#  Related Topics 树 深度优先搜索 动态规划 二叉树 
#  👍 1564 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        cache = dict()
        self.max_sum = float('-inf')
        def max_path_sum(node):
            """返回以node作为头结点的最长的path和"""
            if node not in cache:
                if node is None:
                    return 0
                left_sum = max(max_path_sum(node.left), 0)
                right_sum = max(max_path_sum(node.right), 0)
                max_sum = max(left_sum, right_sum) + node.val
                self.max_sum = max(self.max_sum, left_sum + right_sum + node.val)
                cache[node] = max_sum
            return cache[node]

        max_path_sum(root)
        # for node, max_val in cache.items():
        #     print(node.val, '-->', max_val)
        return self.max_sum
# leetcode submit region end(Prohibit modification and deletion)


```

#### 679 24点游戏

```python
# You are given an integer array cards of length 4. You have four cards, each co
# ntaining a number in the range [1, 9]. You should arrange the numbers on these c
# ards in a mathematical expression using the operators ['+', '-', '*', '/'] and t
# he parentheses '(' and ')' to get the value 24. 
# 
#  You are restricted with the following rules: 
# 
#  
#  The division operator '/' represents real division, not integer division.
# 
#  
#  For example, 4 / (1 - 2 / 3) = 4 / (1 / 3) = 12. 
#  
#  
#  Every operation done is between two numbers. In particular, we cannot use '-'
#  as a unary operator.
#  
#  For example, if cards = [1, 1, 1, 1], the expression "-1 - 1 - 1 - 1" is not 
# allowed. 
#  
#  
#  You cannot concatenate numbers together
#  
#  For example, if cards = [1, 2, 1, 2], the expression "12 + 12" is not valid. 
# 
#  
#  
#  
# 
#  Return true if you can get such expression that evaluates to 24, and false ot
# herwise. 
# 
#  
#  Example 1: 
# 
#  
# Input: cards = [4,1,8,7]
# Output: true
# Explanation: (8-4) * (7-1) = 24
#  
# 
#  Example 2: 
# 
#  
# Input: cards = [1,2,1,2]
# Output: false
#  
# 
#  
#  Constraints: 
# 
#  
#  cards.length == 4 
#  1 <= cards[i] <= 9 
#  
#  Related Topics 数组 数学 回溯 
#  👍 371 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
class Solution(object):
    def judgePoint24(self, cards):
        """
        :type cards: List[int]
        :rtype: bool
        """

        """

        题目：给4个数字，使用+-*/括号使其得到24
        注意：
            1. 不能组合数字，符号只能对两个数字运算，/是实数除法
            2. 数字的顺序可以变啊
        题解：
            1. 每次挑两个数字组合一下，回溯直到只剩一个数字，检查该数字的结果即可

        """


        def search(nums):
            if len(nums) == 1:
                if abs(nums[0] - 24) < 0.0000001:
                    return True
                else:
                    return False

            for i, x in enumerate(nums):
                for j, y in enumerate(nums):
                    if i != j:
                        next_nums = []
                        for k, val in enumerate(nums):
                            if k != i and k != j:
                                next_nums.append(val)
                        for mark in ['+', '-', '*', '/']:
                            if mark == '+':
                                next_nums.append(x + y)
                            elif mark == '-':
                                next_nums.append(x - y)
                            elif mark == '*':
                                next_nums.append(x * y)
                            elif mark == '/':
                                if y != 0:
                                    next_nums.append(x / float(y))
                                else:
                                    return False
                            if search(next_nums):
                                return True
                            next_nums.pop()
            return False

        return search(cards)






        # TARGET = 24
        # EPSILON = 1e-6
        # ADD, MULTIPLY, SUBTRACT, DIVIDE = 0, 1, 2, 3
        #
        # def solve(nums):
        #     if not nums:
        #         return False
        #     if len(nums) == 1:
        #         return abs(nums[0] - TARGET) < EPSILON
        #     for i, x in enumerate(nums):
        #         for j, y in enumerate(nums):
        #             if i != j:
        #                 newNums = list()
        #                 for k, z in enumerate(nums):
        #                     if k != i and k != j:
        #                         newNums.append(z)
        #                 # newNums:[z1, z2],
        #                 for k in range(4):
        #                     # 乘法加法满足交换定律，可以跳过
        #                     if k < 2 and i > j:
        #                         continue
        #                     if k == ADD:
        #                         newNums.append(x + y)
        #                     elif k == MULTIPLY:
        #                         newNums.append(x * y)
        #                     elif k == SUBTRACT:
        #                         # 减法不满足交换定律
        #                         newNums.append(x - y)
        #                     elif k == DIVIDE:
        #                         # 除法要检查大小
        #                         if abs(y) < EPSILON:
        #                             continue
        #                         # 除法不满足交换定律
        #                         newNums.append(x / y)
        #                     if solve(newNums):
        #                         return True
        #                     # 还原
        #                     newNums.pop()
        #     return False
        #
        # return solve(cards)




# leetcode submit region end(Prohibit modification and deletion)


s = Solution()
print(s.judgePoint24([1,3,4,6]))

```


#### 24 k个一组反转链表

```python

# Given the head of a linked list, reverse the nodes of the list k at a time, an
# d return the modified list. 
# 
#  k is a positive integer and is less than or equal to the length of the linked
#  list. If the number of nodes is not a multiple of k then left-out nodes, in the
#  end, should remain as it is. 
# 
#  You may not alter the values in the list's nodes, only nodes themselves may b
# e changed. 
# 
#  
#  Example 1: 
# 
#  
# Input: head = [1,2,3,4,5], k = 2
# Output: [2,1,4,3,5]
#  
# 
#  Example 2: 
# 
#  
# Input: head = [1,2,3,4,5], k = 3
# Output: [3,2,1,4,5]
#  
# 
#  
#  Constraints: 
# 
#  
#  The number of nodes in the list is n. 
#  1 <= k <= n <= 5000 
#  0 <= Node.val <= 1000 
#  
# 
#  
#  Follow-up: Can you solve the problem in O(1) extra memory space? 
#  Related Topics 递归 链表 
#  👍 1622 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution(object):
    # 翻转一个子链表，并且返回新的头与尾

    # 1 -> 2 -> 3 -> 4

    # pre = None
    # tmp = head.next = 2 -> 3 -> 4
    # pre = 1 -> None
    # head = tmp  2 -> 3 -> 4
    # tmp = 3 -> 4
    # pre = 2 -> 1 -> None
    # head = 3 -> 4


    def reverse(self, head, tail):
        """给头尾结点，反转他
        :param head:
        :param tail:
        :return:
                      pre
                        p
                         l              r
        1 -> 2 -> 3 -> 4
        """


        pre = ListNode(-1)
        pre.next = head
        p = head
        reach_end = False
        while p is not None:
            # 到达tail节点了，那么完成后就要退出了
            if p == tail:
                reach_end = True
            next_node = p.next
            p.next = pre
            pre = p
            p = next_node
            # 完成后退出
            if reach_end:
                break
        return pre, head

    def reverseKGroup(self, head, k):
        """
        :type head
        :type k
        :rtype
# Input: head = [1,2,3,4,5], k = 2
# Output: [2,1,4,3,5]

     l
          r
0 -> 1 -> 2 -> 3
pre = 0
new_head -> 3 -> 2 -> 1
        """
        # 1 2 3 4 5
        new_head = ListNode(-1)
        new_head.next = head
        pre = new_head
        while pre.next is not None:
            cur_head = pre.next
            cur_tail = pre
            for i in range(k):
                cur_tail = cur_tail.next
                if cur_tail is None:
                    return new_head.next
            tmp_tail_next = cur_tail.next
            cur_head, cur_tail = self.reverse(cur_head, cur_tail)
            pre.next = cur_head
            pre = cur_tail
            cur_tail.next = tmp_tail_next
        return new_head.next





        # new_head = ListNode(0)
        # new_head.next = head
        # pre = new_head
        #
        # while head:
        #     tail = pre
        #     print('in head:',head.val, 'tail:', tail.val)
        #     # 查看剩余部分长度是否大于等于 k
        #     for i in range(k):
        #         tail = tail.next
        #         if not tail:
        #             return new_head.next
        #     next_node = tail.next
        #     head, tail = self.reverse(head, tail)
        #     # 把子链表重新接回原链表
        #     pre.next = head
        #     tail.next = next_node
        #     pre = tail
        #     head = tail.next
        #
        #     print('out head:',head.val, 'tail:', tail.val)
        #
        # return new_head.next
        
# leetcode submit region end(Prohibit modification and deletion)


# Input: head = [1,2,3,4,5], k = 2
# Output: [2,1,4,3,5]
head = ListNode(1)
# head.next = ListNode(2)
# head.next.next = ListNode(3)
# head.next.next.next = ListNode(4)
# head.next.next.next.next = ListNode(5)

s = Solution()
new_head = s.reverseKGroup(head, 2)

while new_head:
    print(new_head.val)
    new_head = new_head.next

```


#### 215 第k大的元素


```python
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """

        # 方法二：快速排序思想
        # [3,2,1,5,6,4]
        # 1,2,3,4,5,6

        def findKthLargestFromLR(nums, k, left, right):
            """[left, right]区间里面的第k大"""
            l, r = left, left
            while r < right:
                if nums[r] > nums[right]:
                    nums[l], nums[r] = nums[r], nums[l]
                    l += 1
                r += 1
            nums[l], nums[right] = nums[right], nums[l]
            index = l - left + 1
            if index == k:
                return nums[l]

            elif index > k:
                return findKthLargestFromLR(nums, k, left, l - 1)
            else:
                return findKthLargestFromLR(nums, k - index, l + 1, right)
                
        return findKthLargestFromLR(nums, k, 0, len(nums) - 1)





        # 方法一：最小堆保存k个最大的元素
        # cache = []
        # for num in nums:
        #     if len(cache) < k:
        #         heapq.heappush(cache, num)
        #     else:
        #         if cache[0] < num:
        #             heapq.heappop(cache)
        #             heapq.heappush(cache, num)
        # return cache[0]



```


#### 23 合并k个排序链表


```python

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        
        import heapq
        dummy = ListNode(0)
        p = dummy
        head = []
        for i in range(len(lists)):
            if lists[i] :
                heapq.heappush(head, (lists[i].val, i))
                lists[i] = lists[i].next
        while head:
            val, idx = heapq.heappop(head)
            p.next = ListNode(val)
            p = p.next
            if lists[idx]:
                heapq.heappush(head, (lists[idx].val, idx))
                lists[idx] = lists[idx].next
        return dummy.next

# 作者：powcai
# 链接：https://leetcode.cn/problems/merge-k-sorted-lists/solution/leetcode-23-he-bing-kge-pai-xu-lian-biao-by-powcai/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


    #     n = len(lists)
    #     if n == 0:
    #         return
    #     return self.merge(lists, 0, n-1)

    # def merge(self, lists, left, right):
    #     """分治法"""
    #     if left == right:
    #         return lists[left]
    #     mid = left + (right - left) // 2
    #     l1 = self.merge(lists, left, mid)
    #     l2 = self.merge(lists, mid + 1, right)
    #     return self.mergeTwoLists(l1, l2)
    
    # def mergeTwoLists(self, l1, l2):
    #     """递归"""
    #     if l1 is None:
    #         return l2
    #     if l2 is None:
    #         return l1
    #     if l1.val < l2.val:
    #         l1.next = self.mergeTwoLists(l1.next, l2)
    #         return l1
    #     else:
    #         l2.next = self.mergeTwoLists(l1, l2.next)
    #         return l2

```


#### 44 通配符匹配


```python
class Solution(object):
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """

        m, n = len(s), len(p)
        dp = [[False] * (n+1) for _ in range(m+1)]

        dp[0][0] = True

        for j in range(1, n+1):
            if p[j-1] == '*' and dp[0][j-1]:
                dp[0][j] = True
            else:
                break


        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i-1] == p[j-1] or p[j-1] == '?':
                    if dp[i-1][j-1]:
                        dp[i][j] = True
                else:
                    if p[j-1] == '*':
                        if dp[i][j-1]:
                            dp[i][j] = True
                        if dp[i-1][j]:
                            dp[i][j] = True
        
        # for dpi in dp:
        #     print(dpi)

        return dp[m][n]

```


#### 面试题 17.24. 最大子矩阵


```python
class Solution(object):
    def getMaxMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]

        面试题 17.24. 最大子矩阵
输入：
[
   [-1,0],
   [0,-1]
]
输出：[0,1,0,1]
解释：输入中标粗的元素即为输出所表示的矩阵

        题解：
        1. 利用前缀和将行压缩，k1，k2有n**2个选择
        2. 然后利用dp思想，计算该行的最大数组和
        3. 利用pre统计之前为0的位置
        """

        m, n = len(matrix), len(matrix[0])

        # 1. 行维度的前缀和
        pre_row_sum = [[0] * n for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(n):
                pre_row_sum[i][j] = pre_row_sum[i-1][j] + matrix[i-1][j]

        max_acc = float('-inf')
        x1, y1, x2, y2 = -1, -1, -1, -1
        # m = 2, k1=[0,1]
        # m = 2, k2=[1,2]  
        for k1 in range(m+1):
            for k2 in range(k1+1, m+1):
                cur_row = [0] * (n)
                pre_sum = 0
                pre_j = 0
                for j in range(n):
                    cur_row[j] = pre_row_sum[k2][j] - pre_row_sum[k1][j] # [k1, k2]
                    cur_sum = pre_sum + cur_row[j]
                    if cur_sum > max_acc:
                        max_acc = cur_sum
                        x1, y1, x2, y2 = max(k1, 0), pre_j, k2-1, j
                    if cur_sum < 0:
                        pre_j = j + 1
                        pre_sum = 0
                    else:
                        pre_sum = cur_sum
        return [x1, y1, x2, y2]
                    
                

```