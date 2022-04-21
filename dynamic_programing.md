#### 10 正则表达式

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

#### 121 只能买卖一次，买卖股票的最大收益
```python
class Solution(object):
    def maxProfit(self, prices):
        """只能买卖一次，买卖股票的最大收益
        :type prices: List[int]
        :rtype: int

        Example:
            Input: prices = [7,1,5,3,6,4]
            Output: 5
        Solution:
            1. dp[i][0] 第i天没有股票的最大收益
            2. dp[i][1] 第i天有股票的最大收益
            3. 可以去掉dp
        """

        n = len(prices)
        if n <= 1:
            return 0
        # dp[n][k][0-1]
        # p0 手里没有股票的最大收益
        # p1 手里有股票的最大收益
        p0, p1 = 0, -prices[0]
        for i in range(1, n):
            p0, p1 = max(p1 + prices[i], p0), max(-prices[i], p1)
            # print(p0)
        return p0

```

#### 122 买卖多次，买卖股票的最大收益

```python
class Solution(object):
    def maxProfit(self, prices):
        """t+0买卖股票的最大收益
        :type prices: List[int]
        :rtype: int
        Example:
            Input: prices = [7,1,5,3,6,4]
            Output: 7
        Solution:
            一样
        """

        n = len(prices)
        if n <= 1:
            return 0
        
        p0, p1 = 0, -prices[0]
        for i in range(1, n):
            p0, p1 = max(p0, p1+prices[i]), max(p1, p0-prices[i])
        return p0

```

#### 123 交易2次，买卖股票的最大收益

```python
class Solution(object):
    def maxProfit(self, prices):
        """最多交易两次，买卖股票的最大收益
        :type prices: List[int]
        :rtype: int

        Example:
            Input: prices = [3,3,5,0,0,3,1,4]
            Output: 6

        Solution:
            1. dp[i][k][0]: 第i天，交易k次，没有股票的最大收益
            2. dp[i][k][1]: 第i天，交易k次，没有股票的最大收益
        """

        n = len(prices)
        if n <= 1:
            return 0
        dp = [[[0, float('-inf')] for k in range(3)] for _ in range(n)]
        # 1. 初始化
        for j in range(0, 2+1):
            dp[0][j][0] = 0
            # 为什么0天完成j次交易都是-prices[0]
            # 第0天当天，不管你完成多少次交易，有股票就是-prices[0], 无股票就是0
            dp[0][j][1] = -prices[0]
        # 2. 遍历
        for i in range(1, n):
            for d in range(1, 3):
                dp[i][d][0] = max(dp[i-1][d][0], dp[i-1][d][1] + prices[i])
                dp[i][d][1] = max(dp[i-1][d][1], dp[i-1][d-1][0] - prices[i]) # 买股票的时候算发生一次交易
        
        return dp[n-1][2][0]
```

#### 174 到达右下角最少需要的血量

```python
class Solution(object):
    def calculateMinimumHP(self, dungeon):
        """
        :type dungeon: List[List[int]]
        :rtype: int
        Example:
            # Input: dungeon = [[-2,-3,3],
                                [-5,-10,1],
                                [10,30,-5]]
            # Output: 7
            # Explanation: The initial health of the knight must be at least 7 if he follows
            #  the optimal path: RIGHT-> RIGHT -> DOWN -> DOWN.
            5 2 5
                6
                1
            dp result
            7 5  2
            6 11 5
            0 0 6
        Solution:
            1. 逆向dp, dp[i][j]代表从位置i,j走到右下的最小的[额外需要]的血量
            2. dp[i][j] = max(min(dp[i][j+1], dp[i+1][j]) - dungeon[i][j], 1), 血量要大于1，不大于1就挂掉了
        """
        m, n = len(dungeon), len(dungeon[0])
        pre = [0] * n
        for i in range(m-1, -1, -1):
            cur = [0] * n
            for j in range(n-1, -1, -1):
                if i == m - 1 and j == n - 1:
                    cur[j] = max(1 - dungeon[m-1][n-1], 1)
                    continue
                if i == m - 1:
                    min_res = max(cur[j+1] - dungeon[i][j], 1)
                elif j == n -1:
                    min_res = max(pre[j] - dungeon[i][j], 1)
                else:
                    min_res = max(min(pre[j], cur[j+1]) - dungeon[i][j], 1)
                cur[j] = min_res
            pre = cur
            # print(pre)
        return pre[0]

```

#### 416 一个数组，能否分成和相等的两部分
```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:

        """
        题目：一个数组，能否分成和相等的两部分
        Example:
            Input: nums = [1,5,11,5]
            Output: true
        题解：01背包问题
            1. 遍历nums
            2. 遍历0-half，half代表和的一半
            3. dp[i][j]为True代表前i个数字，能否组合成j，如果i-1个数能组合成j-num[i]或者i-1个本来就已经能组合成j了，那么i个也可以组合成j
            4. 总体思想就是需不需要利用上当前的num来凑target，反正需不需要都可以，只要最终能凑成target就行
            5. 初始化凑成0都为True，意思就是不要任何数，都能凑成0
        """
        n = len(nums)
        if n < 2:
            return False
        
        total = sum(nums)
        maxNum = max(nums)
        if total & 1:
            return False
        
        target = total // 2
        if maxNum > target:
            return False
        
        # 前i个数，能不能组合成target
        dp = [[False] * (target + 1) for _ in range(n)]

        # 初始化，组合成0都为True?
        for i in range(n):
            dp[i][0] = True
        # 0-1 -> True
        dp[0][nums[0]] = True
        for i in range(1, n):
            num = nums[i]
            for j in range(1, target + 1):
                if j >= num:
                    # 如果i-1个数就可以组合成j或者i-1个数可以组合成j-num
                    # 那么就为True
                    dp[i][j] = dp[i - 1][j] | dp[i - 1][j - num]
                else:
                    # j不一定大于Num，这个时候看前面i-1个数能不能组合成j
                    dp[i][j] = dp[i - 1][j]
        # 因为target为half，所以不可能为全部数组的和
        return dp[n - 1][target]


```

#### 449 序列化和反序列化搜索二叉树

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

#### 514 最小输出密码的步数-输入字符串环ring和密码key，问需要多少步可以输出密码key，起始位置为0

```python
class Solution(object):
    def findRotateSteps(self, ring, key):
        """输入字符串环ring和密码key，问需要多少步可以输出密码key，起始位置为0
        :type ring: str
        :type key: str
        :rtype: int
        #  Example 1:
        #
        #
        # Input: ring = "godding", key = "gd"
        # Output: 4
        # Explanation:
        # For the first key character 'g', since it is already in place, we just need 1
        # step to spell this character.
        # For the second key character 'd', we need to rotate the ring "godding" anticlo
        # ckwise by two steps to make it become "ddinggo".
        # Also, we need 1 more step for spelling.
        # So the final output is 4.

        Solution:
            dp(l, r): 代表当前位置在l，问需要多少步可以输出密码key[r:]
        """

        m = len(ring)
        n = len(key)
        memo = [[float('inf')] * n for _ in range(m)]
        def search(l, r):
            # 全部输出完毕，直接返回0
            if r == n:
                return 0
            if memo[l][r] == float('inf'):
                for i in range(0, m):
                    # 顺时针走
                    # abca -> ba
                    #  l=1      r
                    if l + i >= m:
                        new_key = l + i - m
                    else:
                        new_key = l + i
                    if ring[new_key] == key[r]:
                        memo[l][r] = min(memo[l][r], search(new_key, r+1) + i + 1)
                        break
                for i in range(0, m):
                    # 逆时针走
                    # abca -> ba
                    #  l=1     r
                    if l - i < 0:
                        new_key = l - i + m
                    else:
                        new_key = l - i
                    if ring[new_key] == key[r]:
                        memo[l][r] = min(memo[l][r], search(new_key, r+1) + i + 1)
                        break
            return memo[l][r]
        #print(memo)
        return search(0, 0)

```


#### 518 凑出零钱为amount的方法总数

```python
class Solution(object):
    def change(self, amount, coins):
        """凑出零钱为amount的方法总数
        :type amount: int
        :type coins: List[int]
        :rtype: int
        Example:
            Input: amount = 5, coins = [1,2,5]
            Output: 4
        Solution:
            1. dp[i][j]代表前i个硬币凑成j有多少总凑法
            2. 可以改成1维的, 因为dp[i][j]至少等于dp[i-1][j]
        """
        # [0,1,2,3,5]
        dp = [0] * (amount + 1)
        dp[0] = 1
        for coin in coins:
            for i in range(1, amount + 1):
                if i >= coin:
                    dp[i] += dp[i-coin]
        return dp[amount]

```

#### 583 最小删除距离

```python
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int

        Example:
            Input: word1 = "sea", word2 = "eat"
            Output: 2

        Solution:
            dp[i][j] = min(dp[i-1][j], dp[i][j-1])
        """
        # word1 = "sea", word2 = "eat"
        # m=n=3
        # pre = [0, 1, 2, 3]
        # i=1,j=1
        # cur = [1,1,1,1]
        # s != e
        # 

        m, n = len(word1), len(word2)
        if m < n:
            m, n , word1, word2 = n, m, word2, word1
        pre = [_ for _ in range(n + 1)]
        for i in range(1, m + 1):
            cur = [i] * (n + 1)
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    cur[j] = pre[j-1]
                else:
                    cur[j] = min(cur[j-1], pre[j]) + 1
            pre = cur
        return pre[-1]

```


#### 1143 最长公共子序列

```python
class Solution(object):
    def longestCommonSubsequence(self, text1, text2):
        """最长公共子序列
        :type text1: str
        :type text2: str
        :rtype: int

        Example:
            Input: text1 = "abcde", text2 = "ace" 
            Output: 3 

        Solution:
            # 如果不相等，最多和原来少一个元素相等，因为dp[i][j]少两个元素，所以
            # 是包含在内的
            dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        """


        m = len(text1)
        n = len(text2)

        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

```