
### 2. 2211. 统计道路上的碰撞次数
方法：前缀和

```python

class Solution(object):
    def countCollisions(self, a):
        """
        :type directions: str
        :rtype: int
        """
        n = len(a)
        left_has_s_or_r = [False] * n
        right_has_s_or_l = [False] * n
        pre = False
        for i in range(n):
            if a[i] == 'R' or a[i] == 'S':
                pre = True
            if pre:
                left_has_s_or_r[i] = True

        pre = False
        for i in range(n-1, -1, -1):
            if a[i] == 'L' or a[i] == 'S':
                pre = True
            if pre:
                right_has_s_or_l[i] = True

        cur = 0
        cnt = 0
        while cur < n:
            if a[cur] == 'L':
                # check if R or S in the left part
                if left_has_s_or_r[cur]:
                    cnt += 1
                # left = cur - 1
                # while left >= 0:
                #     if a[left] == "R" or a[left] == "S":
                #         cnt += 1
                #         break
                #     left -= 1
            elif a[cur] == 'R':
                # check if L or S in the right part
                if right_has_s_or_l[cur]:
                    cnt += 1
                # right = cur + 1
                # while right < n:
                #     if a[right] == "L" or a[right] == "S":
                #         cnt += 1
                #         break
                #     right += 1
            cur += 1
        return cnt
```

### 3. 6029. 射箭比赛中的最大得分

方法：回溯

```python

class Solution:
    def maximumBobPoints(self, numArrows: int, aliceArrows: List[int]) -> List[int]:

        ans = 0
        plan = [0] * 10
        def search(i, arrows, current, current_plan):
            """回溯算法来搜索

            Args:
                i(int): 起点位置?
                arrows(int): 剩下的没射的箭的数量
                current(int):当前的分数和
                current_plan(list):当前的射的箭的计划，最终保持的其实是从1开始的后续位置的射箭计划
            """
            nonlocal ans, plan
            if i == len(aliceArrows):
                # 如果位置走到终点了，并且当前的分数和大雨ans，那么ans就取当前的值，并记录最大值
                if current > ans:
                    ans = current
                    plan = current_plan[:]
                return
            # 当前节点符合要求，如果剩下的箭的数量大雨alice在当前节点i的数量+1，那么就尝试在这里超过他
            if aliceArrows[i] + 1 <= arrows:
                # 计划就是的箭的数量
                current_plan.append(aliceArrows[i] + 1)
                # 找下个位置，i+1，剩下的箭的数量更新，总分数更新，计划更新
                search(i + 1, arrows - aliceArrows[i] - 1, current + i, current_plan)
                # reset
                current_plan.pop()
            # 不考虑当前节点，也试着搜索一次
            current_plan.append(0)
            search(i + 1, arrows, current, current_plan)
            # 同样需要还原
            current_plan.pop()
        # 第0个位置是0分，先不考虑?
        search(1, numArrows, 0, [])
        # 最终输出plan，
        # 箭头的个数可能会多出来，这样已经取得最大的分数了，把其余箭都射到第1个位置，返回结果就行
        return [numArrows - sum(plan)] + plan
```


### 4. 2213. 由单个字符重复的最长子字符串

```python
class Node:
    def __init__(self):
        """初始化
        Args:
            left(Node): 区间的左边子区间
            right(Node): 区间的右边子区间
            lmost(int): 左边的最大长度
            lchar(string): 区间最左边的字符
            rmost(int): 右边的最大长度
            lchar(string)：区间最右边的的字符
            most(int): 全局的最大长度
            i(int): 区间的起点
            j(int): 区间的终点
        """
        self.left = None
        self.right = None
        self.lmost = None
        self.lchar = None
        self.rmost = None
        self.rchar = None
        self.most = None
        self.i = None
        self.j = None

    def update_stats(self):
        """合并左右两边的状态
        1. left信息更新，包括lchar, lmost更新，尝试利用right拓展left
        2. right信息更新，包括rchar, rmost更新，尝试利用left拓展right
        3. 全局信息更新，most更新，主要合并该区间的left和right
        example：
        # 问：为啥要更新lmost，并且左边全是连续的才合并右边?
        # 答：为了后续更长的重复子长度的合并
        # 更新该区间的lchar和lmost，[1,1,1,1].[1,1,2,2,3,4,5] -> [1,1,1,1,1,1], [1,1,2,2,3,4,5
        """
        # 左边为空，代表没有孩子，直接返回
        if self.left is None:
            return
        # 尝试利用right拓展left
        if self.left.lmost == self.left.j - self.left.i and self.left.rchar == self.right.lchar:
            self.lchar = self.left.lchar
            self.lmost = self.left.lmost + self.right.lmost
        else:
            self.lchar = self.left.lchar
            self.lmost = self.left.lmost
        # 尝试利用left拓展right
        if self.right.rmost == self.right.j - self.right.i and self.right.lchar == self.left.rchar:
            self.rchar = self.right.rchar
            self.rmost = self.right.rmost + self.left.rmost
        else:
            self.rchar = self.right.rchar
            self.rmost = self.right.rmost
        # 最终的长度为左边或者右边的最大长度
        self.most = max(self.left.most, self.right.most)
        # 左右两边的字符串相等，代表可以合并字符串，合并取最大值
        # [left.lchar, left.rchar] [right.lchar, right.rchar]
        if self.left.rchar == self.right.lchar:
            most2 = self.left.rmost + self.right.lmost
            self.most = max(self.most, most2)

    # 类方法，不需要实例化
    @classmethod
    def create(cls, s, i, j):
        node = cls()
        if i + 1 == j: # [i, j)
            # 最长的长度最开始都是1
            node.lmost = 1
            node.rmost = 1
            # 左右两边的字符串也是知道的
            node.lchar = s[i]
            node.rchar = s[i]
            # 最大长度为1
            node.most = 1
            # 记录该区间的起始位置
            node.i = i
            node.j = j
        else:
            # 如果长度不为1，用2分的方法构建数据结构
            m = (i + j) // 2
            # 构建左边节点[i, m)
            node.left = cls.create(s, i, m)
            # 构建右边节点[m, j)
            node.right = cls.create(s, m, j)
            # 记录该区间的起始位置
            node.i = i
            node.j = j
            # 每次更新一下状态
            node.update_stats()
        return node

    def update(self, pos, char):
        # 没有左节点了，代表没有孩子，直接更新当前的char
        if self.left is None:
            self.lchar = self.rchar = char
        # 如果在左边，更新左边
        elif pos < self.left.j:
            self.left.update(pos, char)
        # 否则更新右边
        else:
            self.right.update(pos, char)
        # 更新状态
        self.update_stats()


class Solution:
    def longestRepeating(self, s: str, queryCharacters: str, queryIndices: List[int]) -> List[int]:
        tree = Node.create(s, 0, len(s))
        ans = []
        for i, c in zip(queryIndices, queryCharacters):
            tree.update(i, c)
            # 每次取出最大值就行了
            ans.append(tree.most)
        return ans

```