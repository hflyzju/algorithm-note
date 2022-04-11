#### 128 O(n)时间从数组中找出最长的连续数组

```python

class UniFind(object):

    def __init__(self, n):
        self.parent = [-1] * n
        for i in range(n):
            self.parent[i] = i
        self.n = n

    def get_root(self, x):
        """压缩x的root"""
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def add(self, x, y):
        root_x = self.get_root(x)
        root_y = self.get_root(y)
        if root_x != root_y:
            self.n -= 1
            self.parent[root_x] = root_y

    def get_union_number(self):
        return self.n

class Solution(object):
    def longestConsecutive(self, nums):
        """输入一个数组，实现O(n)的算法统计连续数组的个数
        :type nums: List[int]
        :rtype: int
        Example:
            # Input: nums = [100,4,200,1,3,2]
            # Output: 4
        Solution:
            1. 并查集，统计最长的集合，注意数字不能重复
        """
        if len(nums) == 0:
            return 0

        # 1. 统计每个脚本出现的次数
        num_to_cnt = defaultdict(int)
        for num in nums:
            num_to_cnt[num] += 1

        # 2. 建立映射
        i = 0
        union_num_to_index = dict()
        for num in num_to_cnt.keys():
            union_num_to_index[num] = i
            i += 1

        # 3. 将数组的邻居添加到Union find中
        union_find = UniFind(len(num_to_cnt))
        for num in nums:
            if num - 1 in num_to_cnt:
                union_find.add(union_num_to_index[num], union_num_to_index[num - 1])
            if num + 1 in num_to_cnt:
                union_find.add(union_num_to_index[num], union_num_to_index[num + 1])

        max_continue_number_cnt = 0
        index_to_cnt = defaultdict(int)

        # 4. 遍历num_to_cnt相当于对nums去重
        for num in num_to_cnt:
            parent = union_find.get_root(union_num_to_index[num])
            index_to_cnt[parent] += 1
            if index_to_cnt[parent] > max_continue_number_cnt:
                max_continue_number_cnt = index_to_cnt[parent]

        return max_continue_number_cnt
```


#### 130 将所有非边界相连的被X包围的O转化成X

```python
class UnionFind(object):

    def __init__(self, n):
        self.parent = [-1] * n
        for i in range(n):
            self.parent[i] = i
        self.n = n

    def get_root(self, x):
        """压缩x的root"""
        # 跳两步持续压缩，直到走到根节点
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            # 走到父节点继续向上压缩
            x = self.parent[x]
        return x

    def add(self, x, y):
        root_x = self.get_root(x)
        root_y = self.get_root(y)
        if root_x != root_y:
            self.n -= 1
            self.parent[root_x] = root_y

    def get_union_number(self):
        return self.n

class Solution(object):
    def solve(self, board):
        """将所有非边界相连的被X包围的O转化成X
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        方法：
            1. 先利用并查集算法将与边界相连的O
            2. 将剩下的O直接转化成X即可
        """
        if not board:
            return
        m, n = len(board), len(board[0])
        union_find = UnionFind(m * n + 1)
        bound_index = m * n
        for i in range(m):
            for j in [0, n-1]:
                if board[i][j] != 'X':
                    x = i * n + j
                    union_find.add(x, bound_index)
        for j in range(n):
            for i in [0, m-1]:
                if board[i][j] != 'X':
                    x = i * n + j
                    union_find.add(x, bound_index)

        for i in range(1, m - 1):
            for j in range(1, n - 1):
                if board[i][j] != 'X':
                    for dx, dy in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                        nx, ny = i + dx, j + dy
                        if 0 <= nx < m and 0 <= ny < n and board[nx][ny] != 'X':
                            x = i * n + j
                            y = nx * n + ny
                            union_find.add(x, y)
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                x = i * n + j
                if board[i][j] != 'X' and union_find.get_root(x) != union_find.get_root(bound_index):
                    board[i][j] = "X"

```

#### 765 最小的swap次数使所有夫妻都做到一起

```python
class UniFind(object):

    def __init__(self, n):
        self.parent = [-1] * n
        for i in range(n):
            self.parent[i] = i
        self.n = n

    def get_root(self, x):
        """压缩x的root"""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def add(self, x, y):
        root_x = self.get_root(x)
        root_y = self.get_root(y)
        if root_x != root_y:
            self.n -= 1
            self.parent[root_y] = root_x

    def get_union_number(self):
        return self.n

class Solution(object):
    def minSwapsCouples(self, row):
        """求最小需要多少次swap可以将所有夫妻都放到一起
        :type row: List[int]
        :rtype: int
        example：
            # Input: row = [0,2,1,3]
            # Output: 1
            # Explanation: We only need to swap the second (row[1]) and third (row[2]) perso
        题解：
            1. 为每对夫妻设立一个编号，这里应该只有[0/2=0,1/2=0, 2/2=1, 3/2=1]=[0,1]两对编号
            2. 本来的并查集为2
            3. 这里0/2=0 -> 2/2=1, 0和1连起来了，并查集组数只有1了，这里需要swap1次来平衡
            4. 所以最终结果：n - union_cnt
        """

        n = len(row) // 2
        union_find = UniFind(n)
        for i in range(0, len(row), 2):
            union_find.add(row[i] // 2, row[i+1] // 2)
        return n - union_find.get_union_number()
```


#### 990 判断表达式是否存在冲突，有冲突返回false，没有返回true

```python
class UniFind(object):

    def __init__(self, n):
        self.parent = [-1] * n
        for i in range(n):
            self.parent[i] = i
        self.n = n

    def get_root(self, x):
        """压缩x的root"""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def is_connect(self, x, y):
        root_x = self.get_root(x)
        root_y = self.get_root(y)
        return root_x == root_y

    def add(self, x, y):
        root_x = self.get_root(x)
        root_y = self.get_root(y)
        if root_x != root_y:
            self.n -= 1
            self.parent[root_y] = root_x

    def get_union_number(self):
        return self.n


class Solution(object):
    def equationsPossible(self, equations):
        """
        :type equations: List[str]
        :rtype: bool
        """
        union_find = UniFind(len(equations) * 2)
        letter_to_index = dict()
        index = 0
        for equation in equations:
            if '==' in equation:
                x, y = equation.split('==')
                if x not in letter_to_index:
                    letter_to_index[x] = index
                    index += 1
                if y not in letter_to_index:
                    letter_to_index[y] = index
                    index += 1
                x, y = letter_to_index[x], letter_to_index[y]
                union_find.add(x, y)

        for equation in equations:
            if '!=' in equation:
                x, y = equation.split('!=')
                if x not in letter_to_index:
                    letter_to_index[x] = index
                    index += 1
                if y not in letter_to_index:
                    letter_to_index[y] = index
                    index += 1
                x, y = letter_to_index[x], letter_to_index[y]
                if union_find.is_connect(x, y):
                    return False

        return True


```