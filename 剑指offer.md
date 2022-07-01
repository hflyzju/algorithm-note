## 一、总结


|  类型 | 难度  | 题目 | 题解 | 
|  ----  | ----  | --- | --- |
 面试题05. 替换空格|简单|请实现一个函数，把字符串 s 中的每个空格替换成"%20"。|c++可变字符串修改，先增加字符串长度，然后倒序修改|
|面试题04. 二维数组中的查找|中等|从左到右，从上到下排序数组搜索|从右上往下搜|
|面试题09. 用两个栈实现队列|简单|用两个栈实现一个队|两个栈|
| 面试题11. 旋转数组的最小数字|简单|二分(最差情况为N复杂度)|mid>high,l=mid+1, mid=high,r=mid, mid<high,h=mid, return nums[l]|
| 面试题14- I. 剪绳子|中等|给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段，求最大的乘积是多少| dp[i] = max(dp[i], dp[i - k]*dp[k], dp[i - k]*k, (i-k)*k)|

|面试题15. 二进制中1的个数|简单|转化为2进制后1的个数|每次减去最右边的1：可利用的性质：把一个整数减去1，再和原整数做与运算，会把该整数最右边的1变为0|


## 二、模板

### 2.1 二分模板

```python
"""
输入：numbers = [3,4,5,1,2]
输出：1
"""
class Solution:
    def minArray(self, numbers: List[int]) -> int:

        #### 1:1,2,3,4
        #### 2:4,3,2,1
        #### 3:2,1,3

        i, j = 0, len(numbers) - 1
        while i < j:
            m = (i + j) // 2
            if numbers[m] > numbers[j]:
                i = m + 1
            elif numbers[m] < numbers[j]:
                j = m
            # 单中间的值和末尾的值相等时，又两种情况:[1,1,1,0,1],[1,0,1,1,1],缩小判定范围后，要找的数字还在i-j里面
            else:
                j -= 1
        return numbers[i]
```

### 2.2 回溯模板

#### 剑指 Offer 12. 矩阵中的路径
```python
"""
题目：给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true

题解：回溯
"""
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        def dfs(i, j, k):
            if not 0 <= i < len(board) or not 0 <= j < len(board[0]) or board[i][j] != word[k]: 
                return False
            if k == len(word) - 1: 
                return True
            tmp, board[i][j] = board[i][j], '/'
            res = dfs(i + 1, j, k + 1) or dfs(i - 1, j, k + 1) or dfs(i, j + 1, k + 1) or dfs(i, j - 1, k + 1)
            board[i][j] = tmp
            return res

        for i in range(len(board)):
            for j in range(len(board[0])):
                if dfs(i, j, 0): return True
        return False

```

### 2.3 图的遍历（带条件）
#### 面试题13. 机器人的运动范围

- dfs
```python
class Solution:
    def movingCount(self, m: int, n: int, k: int) -> int:

        # example1: [[0,0,0,0]]
        # example2:[[0],[0],[0],[0]]
        board = [[0 for _ in range(n)] for __ in range(m)]
        def sum_of_loc(num):
            """数位之和"""
            s = 0
            while num:
                s += num % 10
                num = num // 10
            return s
        self.cnt = 0
        def dfs(i, j):
            if i<0 or j<0 or i>=m or j>=n:
                return
            # 用board[i][j]=0代表未访问
            if board[i][j] == 0 and (sum_of_loc(i) + sum_of_loc(j)) <= k:
                self.cnt += 1
                # 设为访问，因为每到一个节点，都把它的相邻的点访问完了，没有后顾之忧
                board[i][j] = 1
                dfs(i-1, j) 
                dfs(i+1, j)
                dfs(i, j-1)
                dfs(i, j+1)
        dfs(0, 0)
        return self.cnt
```

- bfs

```python
class Solution:
    def movingCount(self, m: int, n: int, k: int) -> int:

        # example1: [[0,0,0,0]]
        # example2:[[0],[0],[0],[0]]
        board = [[0 for _ in range(n)] for __ in range(m)]
        def sum_of_loc(num):
            """数位之和"""
            s = 0
            while num:
                s += num % 10
                num = num // 10
            return s

        self.cnt = 0
        dp = collections.deque()
        dp.append([0, 0])
        self.cnt = 0
        while dp:
            cur_x, cur_y = dp.popleft()
            if cur_x <0 or cur_x >= m or cur_y < 0 or cur_y >= n or board[cur_x][cur_y] == 1 or ((sum_of_loc(cur_x) + sum_of_loc(cur_y)) > k):
                continue
            self.cnt += 1
            board[cur_x][cur_y] = 1
            dp.append([cur_x + 1, cur_y])
            dp.append([cur_x - 1, cur_y])
            dp.append([cur_x, cur_y - 1])
            dp.append([cur_x, cur_y + 1])

        return self.cnt         
```


### 2.4 位运算

#### 面试题15. 二进制中1的个数
- 位运算
```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        cnt = 0
        while n:
            if n & 1 == 1:
                cnt += 1
            n = n >> 1
        return cnt
```

- 数学性质:性质：把一个整数减去1，再和原整数做与运算，会把该整数最右边的1变为0.

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        #位运算：减1相与
        count = 0
        while n:
            count += 1
            n = (n-1) & n
        return count

```

### 2.5 质数

#### 1175. 质数排列
```c++
class Solution {
public:
    int mod = 1e9 + 7;

    int countPrimes(int n) {
        // https://labuladong.github.io/algo/4/30/114/
        vector<bool> isPrime(n+1, true);
        for (int i = 2; i * i <= n; i++) {
            if (isPrime[i]) {
                for (int j = i * i; j <= n; j += i) {
                    isPrime[j] = false;
                }
            }
        }
        int count = 0;
        for (int i = 2; i <= n; i++) {
            if (isPrime[i]) {
                count++;
            }
        }
        return count;
    }

    long factorial(int n) {
        long res = 1;
        for (int i = 1; i <= n; i++) {
            res *= i;
            res %= mod;
        }
        return res;
    }


    int numPrimeArrangements(int n) {
        int primeNum = countPrimes(n);
        int notPrimeNum = n - primeNum;
        // cout << "primeNum:" << primeNum << "notPrimeNum:" << notPrimeNum << endl;
        long res = 1;
        
        return (int) (factorial(primeNum) * factorial(notPrimeNum) % mod);
    }
};
```


## 三、详细题解

### 数组&排序

#### 面试题03. 数组中重复的数字

标签:Hash, 排序
方法一: hash: O(n), O(n)
方法二: 排序: O(nlogn), O(1)

```python
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        nums.sort()
        pred = nums[0]
        for i in range(1, len(nums)):
            if pred == nums[i]:
                return pred
            pred = nums[i]
        return
        
```

#### 面试题04. 二维数组中的查找

题目：在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。


1.逐行使用二分查找
O(n*log(m))
O(1)

2.从左下角或右上角的元素开始查找，如果不等于target，可以直接减少一行或一列的查找
O(n + m)
O(1)

```python
class Solution:
    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:

        if not matrix:
            return False
        m, n = len(matrix), len(matrix[0])
        if m ==0 or n == 0:
            return False
        # 右上角
        i, j = 0, n - 1
        # 居然还要看外面啊
        if matrix[i][j] == target:
            return True
        while matrix[i][j] != target:
            # 如果偏大，那么肯定在左边
            if matrix[i][j] > target:
                j -= 1
            # 如果偏小，肯定在右边
            else:
                i += 1
            if i >= m or j < 0:
                return False
            if matrix[i][j] == target:
                return True

```

#### 面试题05. 替换空格

1.遍历替换

```python
class Solution:
    def replaceSpace(self, s: str) -> str:

        rt = []
        for s_i in list(s):
            if s_i == ' ':
                rt.append('%20')
            else:
                rt.append(s_i)

        return ''.join(rt)
```

2. 可变字符串倒序修改

```c++
class Solution {
public:
    string replaceSpace(string s) {
        int count = 0, len = s.size();
        // 统计空格数量
        for (char c : s) {
            if (c == ' ') count++;
        }
        // 修改 s 长度
        s.resize(len + 2 * count);
        // 倒序遍历修改
        for(int i = len - 1, j = s.size() - 1; i < j; i--, j--) {
            if (s[i] != ' ')
                s[j] = s[i];
            else {
                s[j - 2] = '%';
                s[j - 1] = '2';
                s[j] = '0';
                j -= 2;
            }
        }
        return s;
    }
};

// 作者：jyd
// 链接：https://leetcode.cn/problems/ti-huan-kong-ge-lcof/solution/mian-shi-ti-05-ti-huan-kong-ge-ji-jian-qing-xi-tu-/
// 来源：力扣（LeetCode）
// 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```




#### 面试题06. 从尾到头打印链表

1.遍历然后反向输出

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:

        rt = []
        cur = head
        while cur is not None:
            rt.append(cur.val)
            cur = cur.next

        return rt[::-1]
```

2.居然还可以递归，递归大法比较好啊

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:

        if head:
            return self.reversePrint(head.next) + [head.val]
        else:
            return []
```

3.栈

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:

        # 栈，先进后出
        cache = []
        while head:
            cache.append(head.val)
            head = head.next

        res = []
        while cache:
            res.append(cache.pop())

        return res
```

#### 面试题07. 重建二叉树

1.递归
```python
"""
前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]

前序遍历: root --> left --> right
中序遍历: left --> root --> right
后序遍历： left --> right --> root
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:


        if not preorder or not inorder:
            return None
        node = TreeNode(preorder[0])
        #查找当前的根节点在中序遍历中的位置
        index = inorder.index(preorder[0]) 
        # 划分左右子树, 因为preorder和inorder的长度必须一致, 所以找到了index,可以通用
        left_pre = preorder[1:index+1]
        left_in = inorder[:index]
        right_pre = preorder[index+1:]
        right_in = inorder[index+1:]
        node.left = self.buildTree(left_pre, left_in)
        node.right = self.buildTree(right_pre, right_in)

        return node
```

#### 面试题09. 用两个栈实现队列

1.队列:先进先出, 栈:先进后出
```python

class CQueue:

    def __init__(self):
        # 一个缓存，一个排出来，list可以做到先进后出，两个叠加起来就是先进先出
        self.cache, self.pop = [], []

    def appendTail(self, value: int) -> None:
        self.cache.append(value)

    def deleteHead(self) -> int:

        if self.pop:
            return self.pop.pop()

        if not self.cache:
            return -1

        while self.cache:
            self.pop.append(self.cache.pop())
        return self.pop.pop()



# Your CQueue object will be instantiated and called as such:
# obj = CQueue()
# obj.appendTail(value)
# param_2 = obj.deleteHead()
```

- c++版本

```c++
class CQueue {


private:
    stack<int> instack;
    stack<int> outstack;

public:
    CQueue() {

    }
    
    void appendTail(int value) {
        instack.push(value);
    }
    
    int deleteHead() {
        if(outstack.empty()) {
            while(!instack.empty()) {
                int val = instack.top();
                instack.pop();
                outstack.push(val);
            }
        }

        if(!outstack.empty()) {
            int res = outstack.top();
            outstack.pop();
            return res;
        }
        return -1;
    }
};

/**
 * Your CQueue object will be instantiated and called as such:
 * CQueue* obj = new CQueue();
 * obj->appendTail(value);
 * int param_2 = obj->deleteHead();
 */

```

#### 面试题10- I. 斐波那契数列

```
F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
```

- 动态规划
```python
class Solution:
    def fib(self, n: int) -> int:

        dp = [0 for _ in range(max(2, n+1))]
        dp[0] = 0
        dp[1] = 1
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]

        return dp[n] % 1000000007
     

    
```

- 递归


#### 面试题10- II. 青蛙跳台阶问题
- 动态规划, 就跟上面是一道题

```python
class Solution:
    def numWays(self, n: int) -> int:

        dp = [0 for _ in range(max(3, n + 1))]

        dp[0] = 1
        dp[1] = 1
        dp[2] = 2

        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]

        return dp[n] % 1000000007
```

```c++

class Solution {
public:
    int numWays(int n) {
        if (n == 0) {
            return 1;
        }
        if(n <= 2 ) {
            return n; 
        }

        int a = 1, b = 2;
        int mod = 1000000007;
        for(int i=3; i<n+1; i++) {
            int c = a + b;
            a = b;
            b = c;
            b = b % mod;
        }

        return b;

    }
};
```

#### 面试题11. 旋转数组的最小数字

- 遍历
```python

class Solution:
    def minArray(self, numbers: List[int]) -> int:

        if not numbers:
            return 
        #### 1:1,2,3,4
        #### 2:4,3,2,1
        #### 3:2,1,3

        reverse_index = -1
        for i in range(len(numbers) - 1):
            if numbers[i] > numbers[i + 1]:
                reverse_index = i
        return numbers[reverse_index+1]
          
```

- 二分法
```python
class Solution:
    def minArray(self, numbers: List[int]) -> int:

        #### 1:1,2,3,4
        #### 2:4,3,2,1
        #### 3:2,1,3

        i, j = 0, len(numbers) - 1
        while i < j:
            m = (i + j) // 2
            if numbers[m] > numbers[j]:
                i = m + 1
            elif numbers[m] < numbers[j]:
                j = m
            # 单中间的值和末尾的值相等时，又两种情况:[1,1,1,0,1],[1,0,1,1,1],缩小判定范围后，要找的数字还在i-j里面
            else:
                j -= 1
        return numbers[i]



```


#### 面试题12. 矩阵中的路径

- 递归
```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:

        m = len(board)
        n = len(board[0])
        k = len(word)

        def dfs(i, j, k_i):

            if k_i == k:
                return True
            if i < 0 or i >= m or j < 0 or j >= n:
                return False
            if visited[i][j]:
                return False
            if board[i][j] != word[k_i]:
                return False
            visited[i][j] = 1
            flag = dfs(i - 1, j, k_i + 1) or dfs(i + 1, j, k_i + 1) or dfs(i, j - 1, k_i + 1) or dfs(i, j + 1, k_i + 1)
            # 这里，要归0，因为假设先看左边，左边没找到，后面还是可以继续看左边的.
            visited[i][j] = 0
            return flag

        visited_cache = [[0 for _ in range(n)] for __ in range(m)]
        for i in range(m):
            for j in range(n):
                visited = visited_cache.copy()
                if dfs(i, j, 0):
                    return True
        
        return False
```

- 答案

```python

"""
题目：给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true

题解：回溯
"""
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        def dfs(i, j, k):
            if not 0 <= i < len(board) or not 0 <= j < len(board[0]) or board[i][j] != word[k]: 
                return False
            if k == len(word) - 1: 
                return True
            tmp, board[i][j] = board[i][j], '/'
            res = dfs(i + 1, j, k + 1) or dfs(i - 1, j, k + 1) or dfs(i, j + 1, k + 1) or dfs(i, j - 1, k + 1)
            board[i][j] = tmp
            return res

        for i in range(len(board)):
            for j in range(len(board[0])):
                if dfs(i, j, 0): return True
        return False

```


#### 面试题13. 机器人的运动范围

- dfs
```python
class Solution:
    def movingCount(self, m: int, n: int, k: int) -> int:

        # example1: [[0,0,0,0]]
        # example2:[[0],[0],[0],[0]]
        board = [[0 for _ in range(n)] for __ in range(m)]
        def sum_of_loc(num):
            """数位之和"""
            s = 0
            while num:
                s += num % 10
                num = num // 10
            return s
        self.cnt = 0
        def dfs(i, j):
            if i<0 or j<0 or i>=m or j>=n:
                return
            # 用board[i][j]=0代表未访问
            if board[i][j] == 0 and (sum_of_loc(i) + sum_of_loc(j)) <= k:
                self.cnt += 1
                # 设为访问，因为每到一个节点，都把它的相邻的点访问完了，没有后顾之忧
                board[i][j] = 1
                dfs(i-1, j) 
                dfs(i+1, j)
                dfs(i, j-1)
                dfs(i, j+1)
        dfs(0, 0)
        return self.cnt
```

- bfs

```python
class Solution:
    def movingCount(self, m: int, n: int, k: int) -> int:

        # example1: [[0,0,0,0]]
        # example2:[[0],[0],[0],[0]]
        board = [[0 for _ in range(n)] for __ in range(m)]
        def sum_of_loc(num):
            """数位之和"""
            s = 0
            while num:
                s += num % 10
                num = num // 10
            return s

        self.cnt = 0
        dp = collections.deque()
        dp.append([0, 0])
        self.cnt = 0
        while dp:
            cur_x, cur_y = dp.popleft()
            if cur_x <0 or cur_x >= m or cur_y < 0 or cur_y >= n or board[cur_x][cur_y] == 1 or ((sum_of_loc(cur_x) + sum_of_loc(cur_y)) > k):
                continue
            self.cnt += 1
            board[cur_x][cur_y] = 1
            dp.append([cur_x + 1, cur_y])
            dp.append([cur_x - 1, cur_y])
            dp.append([cur_x, cur_y - 1])
            dp.append([cur_x, cur_y + 1])

        return self.cnt         
```

#### 面试题14- I. 剪绳子
- 动态规划
```python
class Solution:
    def cuttingRope(self, n: int) -> int:

        # dp[i] 代表剪乘m段后长度为i的绳子的最大乘积
        dp = [0 for _ in range(max(3, n + 1))]
        dp[1] = 1
        dp[2] = 1
        for i in range(3, n + 1):
            # n 可以从n-1,n-2,...,2,1中来
            for k in range(1, i):
                dp[i] = max(dp[i], dp[i - k]*dp[k], dp[i - k]*k, (i-k)*k)
        return dp[n]
```
- 找规律，尽可能分出更多的3

```python
class Solution:
    """s = x ** (n/x) = (x ** (1/x)) ** n --> lny = 1/x * ln x"""
    def cuttingRope(self, n: int) -> int:
        """尽可能分出更多的3"""
        if n == 2:
            return 1
        if n == 3:
            return 2
        a, b = n // 3, n % 3
        if b ==  0:
            return 3 ** a
        elif b == 1:
            return 3 ** (a - 1) * 4
        else:
            return 3 ** a * 2
```

#### 面试题15. 二进制中1的个数
- 位运算
```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        cnt = 0
        while n:
            if n & 1 == 1:
                cnt += 1
            n = n >> 1
        return cnt
```

- 数学性质:性质：把一个整数减去1，再和原整数做与运算，会把该整数最右边的1变为0.

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        #位运算：减1相与
        count = 0
        while n:
            count += 1
            n = (n-1) & n
        return count

```

#### 面试题16. 数值的整数次方

- 递归

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:

        if n == 0:
            return 1

        if n == 1:
            return x

        if n < 0:
            return 1 / self.myPow(x, -1 * n)

        if n % 2 == 0:
            return self.myPow(x * x, n // 2)

        else:
            return x*self.myPow(x * x, n // 2)
```

- 二分法

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        """
        n=9
         = 1001
         = 1*2^3 + 0*2^2 + 0*2^1 + 1*2^0
        x^9 = x^(2^3) * 1 * 1 * x^(2^0) = x^(8+1)
        """
        if x == 0: 
            return 0
        res = 1
        if n < 0: 
            x, n = 1 / x, -n
        while n:
            if n & 1: 
                res *= x
            x *= x
            n >>= 1
        return res


```

#### 面试题17. 打印从1到最大的n位数

- 直接法，考大数，这道题无意义
```python
class Solution:
    def printNumbers(self, n: int) -> List[int]:
        return list(range(1, 10**n))
```

#### 面试题18. 删除链表的节点

- 新建一个备胎
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        head_tmp = cur = ListNode(-1)
        cur.next = head
        while cur:
            if cur.next:
                if cur.next.val == val:
                    if cur.next.next:
                        cur.next = cur.next.next
                    else:
                        cur.next = None
            cur = cur.next
        return head_tmp.next
```

#### 面试题21. 调整数组顺序使奇数位于偶数前面

```python
class Solution:
    def exchange(self, nums: List[int]) -> List[int]:

        l, r = 0, 0
        n = len(nums)
        while r < n:
            if nums[r] & 1:
                nums[l], nums[r] = nums[r], nums[l]
                l += 1
                r += 1
            else:
                r += 1
        return nums
        
```

#### 面试题22. 链表中倒数第k个节点
- 遍历记录n
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:

        n = 0
        cur = head
        while cur:
            n += 1
            cur = cur.next

        head = head
        while head:
            if n == k:
                return head
            head = head.next
            n -= 1

        
```

- 快慢指针: 先让第一个指针先跑k步, 第一个指针结束了，那后一个指针的位置就是要求的结果
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:

        l = 0
        r = 0 
        # 记录右边的访问位置
        cur = head
        # head记录左边的访问位置
        while cur is not None:
            # 1,2,3
            #  k = 2 --> l = 1, r = 2
            print(l,r,k)
            if r-l+1 == k:
                cur = cur.next
                if cur is not None:
                    head = head.next
                else:
                    break
                l += 1
                r += 1
            else:
                r += 1
                cur = cur.next
        return head
```

- 递归：比较难理解

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:

    def __init__(self,):

        self.cnt = 0

    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:

        if not head:
            return

        res = self.getKthFromEnd(head.next, k)
        # 也会向上传递
        self.cnt += 1
        if self.cnt == k:
            return head
        # 向上传递
        return res
```


#### 面试题24. 反转链表

- 备胎转正

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:

        cache_node = None
        while head:
            # 一定要新建一个
            cur = ListNode(head.val)
            cur.next = cache_node
            cache_node = cur
            head = head.next
        # 备胎转正
        return cache_node

```

- 备胎转正2: 不需要一直新建

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:

        pre, cur = None, head
        while cur:
            # 记录下一个位置
            tmp = cur.next
            cur.next = pre
            # 记录当前位置
            pre = cur
            # 更新单前位置
            cur = tmp
        return pre
```

- 递归

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        # 5
        new_head = self.reverseList(head.next)
        # 掉头, head.next = 5, head.next.next = 4, 这样就掉头过来了
        head.next.next = head
        # 清洗原来的记录, 防止链表循环
        head.next = None
        return new_head
```


#### 面试题25. 合并两个排序的链表  

- 假头节点

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:

        start = head = ListNode(-1)

        while l1 or l2:
            # print('l1:',l1.val if l1 else 'l1 None')
            # print('l2:',l2.val if l2 else 'l2 None')
            if not l1:
                head.next = l2
                break
            if not l2:
                head.next = l1
                break
            if l1.val >= l2.val:
                head.next = l2
                l2 = l2.next
            else:
                head.next = l1
                l1 = l1.next
            # 更新head, 往后看
            head = head.next

        return start.next
```

- 递归

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:

        if not l1:
            return l2

        if not l2:
            return l1

        if l1.val >= l2.val:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
        else:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
```

#### 面试题26. 树的子结构

- 递归

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:

        def recur(A, B):
            # 如果B为空，返回True
            if not B: 
                return True
            # 如果A为空或者A的值与B的值不想等，返回False
            if not A or A.val != B.val: 
                return False
            # 继续看A的子节点和B的子节点想不想等
            return recur(A.left, B.left) and recur(A.right, B.right)

        return bool(A and B) and (recur(A, B) or self.isSubStructure(A.left, B) or self.isSubStructure(A.right, B))


```

#### 面试题27. 二叉树的镜像

- 递归

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return 
        root.left, root.right = self.mirrorTree(root.right), self.mirrorTree(root.left)
        return root
```

#### 面试题28. 对称的二叉树

- 递归

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:

        if not root:
            return True

        def judgeSym(left, right):
            
            if not left and not right:
                return True

            if not left and right:
                return False

            if left and not right:
                return False

            if left.val != right.val:
                return False

            if judgeSym(left.left, right.right) and judgeSym(left.right, right.left):
                return True

            return False


        return judgeSym(root.left, root.right)
```

#### 面试题29. 顺时针打印矩阵

- 错误解法

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:

        if not matrix:
            return []

        m = len(matrix)
        n = len(matrix[0])

        l,r,u,d = 0, n-1, 0, m-1

        rt = []
        while l < r or u < d:
            # 上
            for k1 in range(l, r):
                rt.append(matrix[u][k1])
                u += 1
            # 右
            for k2 in range(u, d):
                rt.append(matrix[r][k2])
                r += 1
            # 下
            for k3 in range(r - 1, l - 1, -1):
                rt.append(matrix[k3][d])
                d -= 1
            # 左
            for k4 in range(d - 1, u - 1, -1):
                rt.append(matrix[l][k4])
                l += 1

        return rt
```

- 要注意随时检查退出以及边界条件

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix: 
            return []
        l, r, t, b, res = 0, len(matrix[0]) - 1, 0, len(matrix) - 1, []
        while True:
            for i in range(l, r + 1): 
                res.append(matrix[t][i]) # left to right
            t += 1
            if t > b: 
                break
            for i in range(t, b + 1):
                 res.append(matrix[i][r]) # top to bottom
            r -= 1
            if l > r: 
                break
            for i in range(r, l - 1, -1):
                 res.append(matrix[b][i]) # right to left
            b -= 1
            if t > b: 
                break
            for i in range(b, t - 1, -1): 
                res.append(matrix[i][l]) # bottom to top
            l += 1
            if l > r: 
                break
        return res

```


#### 面试题30. 包含min函数的栈


- 错误解法

```python
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.cache, self.min_cache = collections.deque(), collections.deque()

    def push(self, x: int) -> None:
        self.cache.append(x)
        if self.min_cache: 
            if x < self.min_cache[-1]:
                self.min_cache.append(x)
        else:
            self.min_cache.append(x)


    def pop(self) -> None:
        x = self.cache.pop()
        if x == self.min_cache[-1]:
            self.min_cache.pop()

    def top(self) -> int:
        return self.cache[-1]

    def min(self) -> int:
        if not self.min_cache:
            return None
        return self.min_cache[-1]



# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.min()
```

- 正确解法: self.B[-1] >= x: 而不是 self.B[-1] > x: 注意边界条件

```python
class MinStack:
    def __init__(self):
        self.A, self.B = [], []

    def push(self, x: int) -> None:
        self.A.append(x)
        if not self.B or self.B[-1] >= x:
            self.B.append(x)

    def pop(self) -> None:
        if self.A.pop() == self.B[-1]:
            self.B.pop()

    def top(self) -> int:
        return self.A[-1]

    def min(self) -> int:
        return self.B[-1]

```


#### 面试题31. 栈的压入、弹出序列

- 模拟法

```python
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:

        mock = []
        popped_index = 0
        for push in pushed:
            mock.append(push)
            while mock and mock[-1] == popped[popped_index]:
                mock.pop()
                popped_index += 1
        if mock:
            return False
        return True
```


#### 面试题32 - I. 从上到下打印二叉树

- 双端队列, 层次遍历

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        rt = []
        dq = collections.deque()
        dq.append(root)
        while dq:
            cur = dq.popleft()
            rt.append(cur.val)
            if cur.left:
                dq.append(cur.left)
            if cur.right:
                dq.append(cur.right)
        return rt
```

#### 层次遍历

- 双端队列, 层次遍历, 缓存

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:

        if not root:
            return []
        rt = []
        dq = collections.deque()
        dq.append(root)
        while dq:
            tmp_rt = []
            tmp_dq = collections.deque()
            while dq:
                cur = dq.popleft()
                tmp_rt.append(cur.val)
                if cur.left:
                    tmp_dq.append(cur.left)
                if cur.right:
                    tmp_dq.append(cur.right)
            rt.append(tmp_rt)
            while tmp_dq:
                dq.append(tmp_dq.popleft())
        return rt
```


#### 面试题32 - III. 从上到下打印二叉树 III

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:

        if not root:
            return []
        dq = collections.deque()
        rt = []
        dq.append(root)
        flag = True
        while dq:
            tmp_rt = []
            tmp_dp = collections.deque()
            while dq:
                cur = dq.popleft()
                tmp_rt.append(cur.val)
                if cur.left:
                    tmp_dp.append(cur.left)
                if cur.right:
                    tmp_dp.append(cur.right)
            if flag:
                rt.append(tmp_rt)
            else:
                rt.append(tmp_rt[::-1])
            flag = not flag
            while tmp_dp:
                dq.append(tmp_dp.popleft())
        return rt
```


#### 面试题33. 二叉搜索树的后序遍历序列

- 递归: 二叉搜索数后续排列，那么出现第一个大于他的数后，后面所有的数应该都要大于才对，否则为false，然后可以递归取判定左右两边是否都符合。

```python
class Solution:
    def verifyPostorder(self, postorder: List[int]) -> bool:

        def judge(left, right):
            if left >= right:
                return True
            # 找到大于right的数
            first_big_index = right
            flag = False
            for k in range(left, right + 1):
                if not flag:
                    if postorder[k] > postorder[right]:
                        flag = True
                        first_big_index = k
                else:
                    if postorder[k] < postorder[right]:
                        return False
            return judge(left, first_big_index - 1) and judge(first_big_index, right - 1)
        return judge(0, len(postorder) - 1)
```


#### 面试题34. 二叉树中和为某一值的路径

- 递归, 错误写法

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:

        rt = []
        def dfs(node, path_list, path_sum):
            path_list.append(node.val)
            path_sum += node.val
            print(node.val)
            print(path_list)
            print(path_sum)
            if not node or (not node.left and not node.right):
                if path_sum == sum:
                    rt.append(path_list.copy())
                ###!!!!!! 不能加这个
                return


            if node.left:
                dfs(node.left, path_list, path_sum)

            if node.right:
                dfs(node.right, path_list, path_sum)

            path_list.pop()
            path_sum -= node.val

        dfs(root, [], 0)

        return rt
                   
```

- 正确解法

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        rt = []
        def dfs(node, path, path_sum):
            if node is None:
                return
            path.append(node.val)
            path_sum += node.val
            if node.left is None and node.right is None:
                if path_sum == sum:
                    rt.append(path.copy())
            else:
                dfs(node.left, path, path_sum)
                dfs(node.right, path, path_sum)
            path.pop()
            path_sum -= node.val
        dfs(root, [], 0)
        return rt
```

#### 面试题35. 复杂链表的复制

???

#### 面试题37. 序列化二叉树

- 构建一颗树，也是根据层次遍历爹思想来构建
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        # 空
        if not root: 
            return "[]"

        queue = collections.deque()
        queue.append(root)
        res = []
        # 层次遍历
        while queue:
            node = queue.popleft()
            if node:
                res.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else: 
                # 如果当前为空，那么就不要把null塞进去了
                res.append("null")
        return '[' + ','.join(res) + ']'

    def deserialize(self, data):
        # 空
        if data == "[]": 
            return
        # 拆解
        vals, i = data[1:-1].split(','), 1
        # 建立根节点
        root = TreeNode(int(vals[0]))
        queue = collections.deque()
        queue.append(root)
        while queue:
            # 抓出来要填充的节点
            node = queue.popleft()
            # 如果对应的数字不为空，填进去
            if vals[i] != "null":
                node.left = TreeNode(int(vals[i]))
                # 下一个要填的就是你
                queue.append(node.left)
            i += 1
             # 如果对应的数字不为空，填进去
            if vals[i] != "null":
                node.right = TreeNode(int(vals[i]))
                # 下一个要填的就是你
                queue.append(node.right)
            i += 1
        return root


        

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))
```

#### 面试题39. 数组中出现次数超过一半的数字

- 哈希计数法
- 排序，中间的数一定是众数(超过一半)
- 摩尔投票法
```python
"""
两种情况：
- 如果 n1 = x(众数), 抵消时，有一半是众数，抵消后, 还有一半以上是众数.
- 如果 n1 != x, 抵消时，小于一半是众数，剩下的众数变多，找来的还是众数.

"""
class Solution:
    def majorityElement(self, nums: List[int]) -> int:

        votes = 0
        for num in nums:
            if votes == 0: 
                x = num
            if num == x:
                votes += 1 
            else:
                votes -= 1
        return x


```


#### 面试题40. 最小的k个数
- 直接排序
加入数据量超级大，直接排序很浪费.

- 快速排序
1. 如果快速排序的中点与要找的第k个数的长度相同，那快速排序就不用继续往下走了，
否则继续对左边或者右边部分进行排序就行。
2. center<k, 继续对预编剩下的进行排序.
3. center>k, 继续在左边找topk的子数组.
```python
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        
        if not arr:
            return -1

        if k < 0:
            return -1

        def quik_sort(l, r, res_k):
            """从子数组l和r(包括r)之间找到第res_k的数"""
            # print(arr,l,r,res_k)
            if l >= r:
                return
            # 双指针的初始位置
            k1, k2 = l, l
            for k2 in range(l, r):
                if arr[k2] < arr[r]:
                    arr[k1], arr[k2] = arr[k2], arr[k1]
                    k1 += 1
            arr[k1], arr[r] = arr[r], arr[k1]
            left_num = k1 - l + 1
            if left_num == res_k:
                return
            elif left_num > res_k:
                quik_sort(l, k1 - 1, res_k)
            elif left_num < res_k:
                quik_sort(k1 + 1, r, res_k - left_num)
        quik_sort(0, len(arr) - 1, k)
        return arr[:k]

```
- 最大堆(python默认是最小堆)
最大堆堆顶是最大得数，如果来一个数比最大的数还大，就丢弃最大的数，加入新来的数.

```python
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        
        if k == 0:
            return []

        import heapq
        rt = [-x for x in arr[:k]]
        heapq.heapify(rt)
        for k1 in range(k, len(arr)):
            if -rt[0] > arr[k1]:
                heapq.heappop(rt)
                heapq.heappush(rt, -arr[k1])

        return [-x for x in rt]
```

#### 面试题42. 连续子数组的最大和

- 用一个缓存前面的最大值
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:

        max_gain = - (1 << 31)
        prefix_cache = 0
        for num in nums:
            prefix_cache += num
            max_gain = max(max_gain, prefix_cache)
            if prefix_cache < 0:
                prefix_cache = 0

        return max_gain
```

- 动态规划
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:


        max_num = nums[0]
        for i in range(1, len(nums)):
            if nums[i - 1]>0:
                nums[i] += nums[i - 1]
            # 注意这个位置
            max_num = max(max_num, nums[i])
        return max_num

```


#### 面试题43. 1～n整数中1出现的次数

1. =0 --> high*digit
2. =1 --> high*digit + low + 1
3. >1 --> (high+1)*digit
4. high =/ 10, low += cur*digit

```python
class Solution:
    def countDigitOne(self, n: int) -> int:
        # 1,9,10,11,20,21,31,41,91,100,101,102,110,111,112,119,120

        # 比较十位数为0,1,2,3...的情况
        # 十位数 > 1  --> n//100*10 + 9 + 1
        # 2356, 0010-2319 --> 000-239
        # 2359, 0010-2319 --> 000-239
        # 十位数 == 1  --> n//100*10 + n%10 + 1
        # 2319, 0010-2319 --> 000-239
        # 2316, 0010-2316 --> 000-236
        # 十位数 == 0 --> (n//100 - 1) * 10 + 9 + 1
        # 2300, 0010-2219 --> 000-229
        # 2308, 0010-2219 --> 000-229

        # digit位数, res和
        digit, res = 1, 0
        # high高位数, cur本身, low地位数
        high, cur, low = n // 10, n % 10, 0
        # 高位部位0或者地位不为0
        while high != 0 or cur != 0:
            # 当前为0
            if cur == 0: 
                res += high * digit
            # 当前为1
            elif cur == 1: 
                res += high * digit + low + 1
            # 当前大于1
            else: 
                res += (high + 1) * digit
            # 更新地位数字，地位加上本身就变成了地位
            low += cur * digit
            # 下个当前数字就是高位数字 % 10
            cur = high % 10
            # 下个高位数字就是高位数字 // 10
            high //= 10
            # 记录位数
            digit *= 10
        return res
```



#### 面试题44. 数字序列中某一位的数字

- 错误解法

```python
class Solution:
    def findNthDigit(self, n: int) -> int:

        if n == 0:
            return 0
        # 1-9 --> 10, 10-99 --> 90, 100-999 --> 900
        digit = 0
        while n > 0:
            cache_num = 9 * (10 ** digit)
            if n > cache_num:
                n = n - cache_num
                digit += 1
            else:
                start_num = 10 ** digit
                num_base = n // (digit+1)
                num_res = n % (digit+1)
                # 11 - 9 = 2, 2 // 2 = 1, 2 % 2 = 0, 10, 11.
                # 100， 
                str_num = str(start_num + num_base - 1)
                return int(str_num[num_res-1])

```

- 正确解法

```python
class Solution:
    """0-9 十个数,10个数位, 10-99 一百个数 90*2=180数位 , 100-999 一千个数，900*3=2700个数位"""
    def findNthDigit(self, n: int) -> int:
        # 初始数位，开始位置，当前digit所能容纳的数位
        digit, start, count = 1, 1, 9
        # 如果n大于count，那么说明数太大，在后面
        while n > count: # 1.
            # 减去当前容量
            n -= count
            # 更新初始位置
            start *= 10
            # 更新每个阶段单个数字拥有的数字位数
            digit += 1
            # 更新当前digit的容量
            count = 9 * start * digit
        # 如果count>n了，代表当前count装得下n了，当前位置每个数字的数位位digit，那么就可以定位到哪个数
        # 例如: n=11 --> n=n-9=2 --> start = 10,  2，就是10这个位置，所以(n-1)//digit, index = (n-1)%digit
        num = start + (n - 1) // digit # 2.
        # index = (n-1)%digit
        return int(str(num)[(n - 1) % digit]) # 3.
```


#### 面试题45. 把数组排成最小的数
- 快速排序
- 传递性证明？字符串 xy < yx , yz < zy ，需证明 xz < zx 一定成立。

```python
class Solution:
    def minNumber(self, nums: List[int]) -> str:


        def fast_sort(l, r):
            """对包括r在内的l-r区间内排序"""

            if l >= r:
                return
            k1 = k2 = l
            for k2 in range(l, r):
                if int(str(nums[k2])+str(nums[r])) < int(str(nums[r])+str(nums[k2])):
                    nums[k1], nums[k2] = nums[k2], nums[k1]
                    k1 += 1
            nums[k1], nums[r] = nums[r], nums[k1]

            fast_sort(l, k1 - 1)
            fast_sort(k1 + 1, r)

        fast_sort(0, len(nums) - 1)

        return ''.join([str(_) for _ in nums])
               
```

#### 面试题46. 把数字翻译成字符串


- 思路: 如果与前一位能够match, 则dp[i] = dp[i-1]+dp[i-2], 即可以由前面一个跳一步来达到, 也可以由前两位跳一步达到, 所以是想加.
- 错误解法
```python
class Solution:
    def translateNum(self, num: int) -> int:

        trans_dict = dict(zip(list(range(0,26)),list('abcdefghijklmnopqrstuvwxyz')))
        num = str(num)
        dp = [0 for _ in range(max(1,len(num)))]
        dp[0] = 1

        for k in range(1, len(num)):
            if int(num[k-1]+num[k]) > 25 and int(num[k-1]+num[k]) < 10:
                dp[k] = dp[k - 1]
            else:
                if k > 1:
                    dp[k] = dp[k - 1] + dp[k - 2]
                else:
                    dp[k] = dp[k - 1] + 1
        
        return dp[len(num) - 1]
```

- 正确解法
```python
class Solution:
    def translateNum(self, num: int) -> int:

        trans_dict = dict(zip(list(range(0,26)),list('abcdefghijklmnopqrstuvwxyz')))
        num = str(num)
        dp = [0 for _ in range(max(1,len(num)))]
        dp[0] = 1

        for k in range(1, len(num)):
            print('num[k-1]+num[k]:',num[k-1]+num[k])
            # 注意这个关系是or， or就行了
            if int(num[k-1]+num[k]) > 25 or int(num[k-1]+num[k]) < 10:
                dp[k] = dp[k - 1]
            else:
                if k > 1:
                    dp[k] = dp[k - 1] + dp[k - 2]
                else:
                    dp[k] = dp[k - 1] + 1
        print(dp)
        return dp[len(num) - 1]
```


#### 面试题47. 礼物的最大价值

- 超时解法, 遍历, 可能路径会超级超级多

```python
class Solution:
    def maxValue(self, grid: List[List[int]]) -> int:

        self.max_gain = 0
        n = len(grid)
        m = len(grid[0])
        def search(i, j, gain):
            # 注意这里好像不是正方形
            if i >= n or j >= m:
                return
            gain += grid[i][j]
            if i == n - 1 and j == m - 1:
                self.max_gain = max(gain, self.max_gain)
            else:
                search(i + 1, j, gain)
                search(i, j + 1, gain)
        search(0, 0, 0)
        return self.max_gain
```

- 动态规划
1. 核心思想是利用上方和左边的值更新当前值

```python

class Solution(object):
    def maxValue(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m, n = len(grid), len(grid[0])
        dp = [[0 for _ in range(n)] for __ in range(m)]

        dp[0][0] = grid[0][0]

        for j in range(1, n):
            dp[0][j] = dp[0][j-1] + grid[0][j]

        for i in range(1, m):
            dp[i][0] = dp[i-1][0] + grid[i][0]

        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] += (max(dp[i-1][j], dp[i][j-1])) + grid[i][j]

        print(dp)
        return dp[m-1][n-1]
```

#### 面试题48. 最长不含重复字符的子字符串

- 最长子串和解法
```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:

        cache_substring = ''
        max_gain = 0
        for i in range(len(s)):
            cache_substring += s[i]
            while len(set(list(cache_substring))) // len(cache_substring) == 0:
                cache_substring = cache_substring[1:]
            max_gain = max(max_gain, len(cache_substring))

        return max_gain
```

- 双指针: 可以返回详细的子串情况

```python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """

        s_list = list(s)
        l = r = 0
        # 记录
        exist_set = set()
        max_length = 0
        while r < len(s_list):
            cur = s_list[r]
            if cur not in exist_set:
                r += 1
                exist_set.add(cur)
                max_length = max(max_length, r - l)
            else:
                # 移动l，直到
                while cur in exist_set:
                    l_cur = s_list[l]
                    exist_set.remove(l_cur)
                    l += 1
        return max_length
```

- 动态规划
??? 

#### 面试题49. 丑数

- 错误写法
```python
class Solution(object):
    def nthUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        # dp[i]代表第i个丑数
        dp = [0 for _ in range(n+1)]
        dp[1], a, b, c = 1, 1, 1, 1
        for k in range(2, n+1):
            can1 = dp[a] * 2
            can2 = dp[b] * 3
            can3 = dp[c] * 5
            min_can = min(can1, can2, can3)
            if min_can == can1:
                a += 1
            elif min_can == can2:
                b += 1
            else:
                c += 1
            dp[k] = min_can

        print(dp)
        return dp[n]
```

- 正确解法

```python
class Solution(object):
    def nthUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        # dp[i]代表第i个丑数
        dp = [0 for _ in range(n+1)]
        dp[1], a, b, c = 1, 1, 1, 1
        for k in range(2, n+1):
            # 每个后面的数，都是由前面的数乘以一个因子的到的，因此，可以用a,b,c记录每个index已经利用了的因子
            can1 = dp[a] * 2
            can2 = dp[b] * 3
            can3 = dp[c] * 5
            min_can = min(can1, can2, can3)
            # 这里要注意，2*3 == 2*2*2, 这样, 会有可能存在重复的情况的, 所以每次更新的时候,
            # 因为每次取的都是最小值, 所以可以通过+1将可能重复的情况取掉
            if min_can == can1:
                a += 1
            if min_can == can2:
                b += 1
            if min_can == can3:
                c += 1
            dp[k] = min_can
        # print(dp)
        return dp[n]
```

#### 面试题50. 第一个只出现一次的字符

```python
class Solution(object):
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: str
        """
        cnt_dict = dict()
        for s_i in s:
            if s_i not in cnt_dict:
                cnt_dict[s_i] = 0
            cnt_dict[s_i] += 1

        for s_i in s:
            if cnt_dict[s_i] == 1:
                return s_i

        return ' '
```


#### 面试题52. 两个链表的第一个公共节点

- 错误写法
```python
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        #  输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, s
        # kipB = 3
        # 输出：Reference of the node with value = 8

        # 4,1,8,4,5,5,0,1,8,4,5
        # 5,0,1,8,4,5,4,1,8,4,5

        #  输入：intersectVal = 2, listA = [0,9,1,2,4], listB = [3,2,4], skipA = 3, skipB =
        #  1
        # 输出：Reference of the node with value = 2
        # 输入解释：相交节点的值为 2 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [0,9,1,2,4]，链表 B 为 [3,2,4
        # ]。在 A 中，相交节点前有 3 个节点；在 B 中，相交节点前有 1 个节点。

        # 0,9,1,2,4,3,2,4
        # 3,2,4,0,9,1,2,4


        # 1，2，3，4，5
        # 2，2，4，5，6

        # 1. 首先要解决两个链表长度不一致的情况，首位相加就解决了.
        # 2. 第二个要找到公共点，那么就循环取比较就好了, 注意这里比较相等的时候， 是直接比较链表是否相等.

        curA = headA
        curB = headB
        while curA and curB:
            if curA == curB:
                return curB
            if curA.next:
                curA = curA.next
            else:
                curA = headB
            if curB.next:
                curB = curB.next
            else:
                curB = headA
        return
```

- 正确解法

```python
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        #  输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, s
        # kipB = 3
        # 输出：Reference of the node with value = 8

        # 4,1,8,4,5,5,0,1,8,4,5
        # 5,0,1,8,4,5,4,1,8,4,5

        #  输入：intersectVal = 2, listA = [0,9,1,2,4], listB = [3,2,4], skipA = 3, skipB =
        #  1
        # 输出：Reference of the node with value = 2
        # 输入解释：相交节点的值为 2 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [0,9,1,2,4]，链表 B 为 [3,2,4
        # ]。在 A 中，相交节点前有 3 个节点；在 B 中，相交节点前有 1 个节点。

        # 0,9,1,2,4,3,2,4
        # 3,2,4,0,9,1,2,4


        # 1，2，3，4，5
        # 2，2，4，5，6

        # 1. 首先要解决两个链表长度不一致的情况，首位相加就解决了.
        # 2. 第二个要找到公共点，那么就循环取比较就好了, 注意这里比较相等的时候， 是直接比较链表是否相等.
        # 3. 如何退出来呢？如果相等就退出来，当
        curA = headA
        curB = headB
        while curA != curB:
            if curA:
                curA = curA.next
            else:
                curA = headB
            if curB:
                curB = curB.next
            else:
                curB = headA
        return curA
```

#### 面试题53. 在排序数组中查找数字

- 二分法
```python
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        def search_sub_list(l, r):
            if l > r:
                return -1
            mid = (l + r) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                return search_sub_list(l, mid - 1)
            else:
                return search_sub_list(mid + 1, r)
        target_index = search_sub_list(0, len(nums) - 1)
        if target_index == -1:
            return 0
        left_index = target_index - 1
        right_index = target_index + 1
        while left_index >= 0 and nums[left_index] == target:
            left_index -= 1
        while right_index < len(nums) and nums[right_index] == target:
            right_index += 1
        return right_index - left_index - 1
```


#### 面试题53. 0 ～ n-1中缺失的数字

- 错误解法
```python
class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        
        #  输入: [0,1,2,3,4,5,6,7,9]
        # 输出: 8 
        """
        def search(l, r):
            """在子数组中查找缺失数字"""
            if l > r:
                return -1
            if l == r:
                return nums[l]
            mid = (l + r) // 2
            if nums[mid] == mid:
                return search(mid + 1, r)
            elif nums[mid] != mid:
                return search(l, mid - 1)
        return search(0, len(nums) - 1)
```


- 正确解法
```python
class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        def search(l, r):
            """在子数组中查找缺失数字"""
            if l >= r:
                return l
            mid = (l + r) // 2
            if nums[mid] == mid:
                return search(mid + 1, r)
            elif nums[mid] != mid:
                return search(l, mid - 1)
        return search(0, len(nums) - 1)
```




#### Number of Paths


```python
class Solution(object):
  
    def __init__(self, ):
        # 0,0 --> n-1, n-1
        self.cnt = 0
    def search(self, cur_i, cur_j, n):
        if cur_i >= n or cur_i <0 or cur_j >= n or cur_j <0:
            return
        if cur_j > cur_i:
            return
        if cur_i == n-1 and cur_j == n-1:
            self.cnt += 1
        # go right
        self.search(cur_i + 1, cur_j, n)
        # go up
        self.search(cur_i, cur_j + 1, n)
        return self.cnt

    def num_of_paths_to_dest(self, n):
        return self.search(0, 0, n)

s = Solution()
print(s.num_of_paths_to_dest(4))
```



#### 5429. 数组中的 k 个最强值

- 超时解法

```python
class Solution:
    def getStrongest(self, arr: List[int], k: int) -> List[int]:
        
        
        arr.sort()
        self.m = arr[((len(arr) - 1) // 2)]
        
        
        def big_comp(k1, k2):
            """
            |arr[i] - m| > |arr[j] - m|
            |arr[i] - m| == |arr[j] - m|，且 arr[i] > arr[j]
            """         
            if abs(arr[k1] - self.m) > abs(arr[k2] - self.m):
                return True
            elif abs(arr[k1] - self.m) == abs(arr[k2] - self.m) and arr[k1] > arr[k2]:
                return True
            
            return False
            
        
        def quik_sort(l, r, res_k):
            """
            包括r
            """
            if l >= r:
                return
            k1, k2 = l, l
            for k2 in range(l, r):
                if big_comp(k2, r):
                    arr[k1], arr[k2] = arr[k2], arr[k1]
                    k1 += 1       
            arr[k1], arr[r] = arr[r], arr[k1]
            big_num = (k1 - l + 1)
            if big_num == res_k:
                return
            elif big_num > res_k:
                quik_sort(l, k1 - 1, res_k)
            else:
                quik_sort(k1 + 1, r, res_k - big_num)
                
        quik_sort(0, len(arr)-1, k)
        
        return arr[:k]
            
            
```


#### 面试题54. 二叉搜索树的第k大节点

- 递归遍历
```python
class Solution(object):
    def kthLargest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        self.cnt = 0
        self.res = 0
        def search(node, k):
            if node.right:
                search(node.right, k)
            self.cnt += 1
            if self.cnt == k:
                self.res = node.val
                return
            if node.left:
                search(node.left, k)
        search(root, k)
        return self.res
```

#### 面试题55. 二叉树的深度

- dfs
```python
# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        self.max_depth = 0
        def dfs(node, depth):
            if node:
                self.max_depth = max(self.max_depth, depth)
                dfs(node.left, depth=depth+1)
                dfs(node.right, depth=depth+1)
        dfs(root,1)
        return self.max_depth
```

- 另一种思路

```python
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
```

#### 面试题55. 平衡二叉树

- 从上到下，自己解法, 时间比较亏，算树的深度肯定重复了, 时间为nlog2n
```python
class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True

        def depth(node):
            if not node:
                return 0
            return max(depth(node.left), depth(node.right)) + 1

        def left_right_balance(left, right):
            if not left and not right:
                return True
            bool1 = left_right_balance(left.left, left.right) if left else True
            bool2 = left_right_balance(right.left, right.right) if right else True
            bool3 = abs(depth(left)-depth(right)) <= 1
            return bool1 & bool2 & bool3

        return left_right_balance(root.left, root.right)
```
- 从上到下2
```python
class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True

        def depth(node):
            if not node:
                return 0
            return max(depth(node.left), depth(node.right)) + 1
        
        return abs(depth(node.left) - depth(node.right)) <=1 and \
               self.isBalanced(node.left) and self.isBalanced(node.right)
```

- 从下到上
```python
class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def recur(node):
            if not node:
                return 0
            left_depth = recur(node.left)
            if left_depth == -1:
                return -1
            right_depth = recur(node.right)
            if right_depth == -1:
                return -1

            return max(left_depth, right_depth) + 1 if abs(left_depth - right_depth) <= 1 else -1
        return recur(root) != -1
```

#### 面试题56-i 数组中数字出现的次数
位运算
```python
class Solution(object):
    def singleNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """

        xor = 0
        # 找到剩下的两个数异或和
        for num in nums:
            xor ^= num

        # 找到某一位不同的作为mask
        mask = 1
        while True:
            if mask & xor:
                break
            mask = mask << 1

        num1, num2 = 0, 0
        for num in nums:
            if num & mask == 0:
                num1 ^= num
            else:
                num2 ^= num
        return [num1, num2]
```

#### 面试题56-ii 数组中数字出现的次数II

```python
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        rt = 0
        for i in range(32):
            cnt = 0
            bit = 1 << i
            for num in nums:
                if num & bit:
                    cnt += 1
            if cnt % 3 != 0:
                rt |= bit
        return rt
```


#### 面试题57 和为s的两个数字

- 双指针
```python
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """

        l, r = 0, len(nums) - 1

        while l < r:
            s = nums[l] + nums[r]
            if s  == target:
                return [nums[l], nums[r]]
            elif s > target:
                r -= 1
            else:
                l += 1
        return -1
```

#### 面试题57-II 和为s的连续正数序列

- 错误写法
```python
class Solution(object):
    def findContinuousSequence(self, target):
        """
        :type target: int
        :rtype: List[List[int]]
        """
        """
        5
        1,2,3
        2,3
        3,4
        
        
        9
        1,2,3,4
        2,3,4
        3,4,5
        5,6
        
        """

        rt = []
        for i in range(1, target // 2):
            tmp, s, next = [i], i, i
            while s < target:
                next += 1
                tmp.append(next)
                s += next
                if s == target:
                    rt.append(tmp)
                    break
        return rt
```

- 正确代码

```python
class Solution(object):
    def findContinuousSequence(self, target):
        """
        :type target: int
        :rtype: List[List[int]]
        """
        """
        5
        1,2,3
        2,3
        3,4
        
        
        9
        1,2,3,4
        2,3,4
        3,4,5
        5,6
        
        """

        rt = []
        for i in range(1, target // 2 + 1):
            tmp, s, next = [i], i, i
            while s < target:
                next += 1
                tmp.append(next)
                s += next
                if s == target:
                    rt.append(tmp)
                    break
        return rt
```


#### 面试题58-I 翻转单词顺序

```python
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """

        s = s.strip()

        l, r = len(s) - 1, len(s) - 1

        rt = []

        while l >= 0:

            while l >= 0 and s[l] != ' ':
                l -= 1

            rt.append(s[l+1:r+1])

            while s[l] == ' ':
                l -= 1

            r = l

        return ' '.join(rt)
```

#### 面试题58-I 左旋转字符串


#### 面试题59-I 滑动窗口的最大值

```python
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        import collections
        if not nums or k == 0: return []
        deque = collections.deque()
        for i in range(k): # 未形成窗口
            # 最后一个数比新来的数小
            while deque and deque[-1] < nums[i]:
                # 把最后的数pop出来
                deque.pop()
            # 然后把新的数搞进去
            deque.append(nums[i])
        res = [deque[0]]
        for i in range(k, len(nums)): # 形成窗口后
            # 判定最大的数是不是需要去掉的那个元素，如果是就先去掉它
            if deque[0] == nums[i - k]:
                deque.popleft()
            # 把前面的小罗罗去掉
            while deque and deque[-1] < nums[i]:
                deque.pop()
            deque.append(nums[i])
            res.append(deque[0])
        return res
```

#### 面试题59-II 队列的最大值

```python
class MaxQueue(object):

    def __init__(self):

        import collections

        self.cache = collections.deque()
        self.max_cache = collections.deque()


    def max_value(self):
        """
        :rtype: int
        """
        if not self.max_cache:
            return -1
        else:
            return self.max_cache[0]


    def push_back(self, value):
        """
        :type value: int
        :rtype: None
        """
        self.cache.append(value)
        if not self.max_cache:
            self.max_cache.append(value)
        else:
            while self.max_cache and self.max_cache[-1] < value:
                self.max_cache.pop()
            self.max_cache.append(value)


    def pop_front(self):
        """
        :rtype: int
        """
        if self.cache:
            if self.cache[0] == self.max_cache[0]:
                self.max_cache.popleft()
            return self.cache.popleft()
        else:
            return -1
```
