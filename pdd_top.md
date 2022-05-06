
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