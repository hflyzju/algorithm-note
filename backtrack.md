

#### 756 金字塔转换矩阵

```python

class Solution(object):
    def pyramidTransition(self, bottom, allowed):
        """
        :type bottom: str
        :type allowed: List[str]
        :rtype: bool
        题目：bottom为底部元素，BCG代表组合BC可以生成G，问最终能否生成金字塔（只剩下1个元素）。
        题解：
            1. 回溯生成（利用递归实现），然后判断。


输入：bottom = "BCD", allowed = ["BCG", "CDE", "GEA", "FFF"]
输出：true
解释：允许的三角形模式显示在右边。
从最底层(第3层)开始，我们可以在第2层构建“CE”，然后在第1层构建“E”。
金字塔中有三种三角形图案，分别是“BCC”、“CDE”和“CEA”。都是允许的。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/pyramid-transition-matrix
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
        """


        T = collections.defaultdict(set)
        for u, v, w in allowed:
            T[u, v].add(w)

        def gen_next_bottom(cur_bottom):
            # print("cur_bottom:", cur_bottom)
            n = len(cur_bottom)
            if n <= 1:
                return []

            if n == 2:
                if len( T[cur_bottom[0], cur_bottom[1]]) > 0:
                    return T[cur_bottom[0], cur_bottom[1]]
                else:
                    return []

            new_bottom = []
            for cand in T[cur_bottom[0], cur_bottom[1]]:
                # print("cur_bottom:", cur_bottom, "cand:", cand)
                for next_cand in gen_next_bottom(cur_bottom[1:]):
                    new_cand = cand + next_cand
                    if len(new_cand) == len(cur_bottom) - 1:
                        new_bottom.append(new_cand)

            return new_bottom


        n = len(bottom)
        cur_bottom = [bottom]
        for i in range(n - 1):
            # print('cur_bottom:', cur_bottom)
            next_bottoms_set = set()
            for cur in cur_bottom:
                next_bottoms = gen_next_bottom(cur)
                for next_bottom in next_bottoms:
                    next_bottoms_set.add(next_bottom)
            if len(next_bottoms_set) == 0:
                return False
            cur_bottom = next_bottoms_set
        # print('final:', cur_bottom)
        return True

            


```

#### 37. 解数独

```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        m, n = len(board), len(board[0])
        row_visited = [[False] * m for _ in range(m)]
        column_visited = [[False] * m for _ in range(m)]
        box_visited = [[False] * m for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if board[i][j] != '.':
                    key = int(board[i][j]) - 1
                    row_visited[i][key] = True
                    column_visited[j][key] = True
                    box_visited[i//3*3 + j//3][key] = True

        def valid(i, j, num):
            key = int(num) - 1
            if row_visited[i][key] == True:
                return False
            if column_visited[j][key] == True:
                return False
            if box_visited[i//3*3 + j//3][key] == True:
                return False
            return True

        def backtrack(board, i, j):
            if j == n:
                return backtrack(board, i + 1, 0)
            if i == m:
                return True
            if board[i][j] == '.':
                for num in range(1, 10):
                    if valid(i, j, num):
                        board[i][j] = str(num)
                        row_visited[i][num-1] = True
                        column_visited[j][num-1] = True
                        box_visited[i//3*3 + j//3][num-1] = True
                        if backtrack(board, i, j + 1):
                            return True
                        row_visited[i][num-1] = False
                        column_visited[j][num-1] = False
                        box_visited[i//3*3 + j//3][num-1] = False
                        board[i][j] = '.'
            else:
                return backtrack(board, i, j + 1)
            return False
            
        return backtrack(board, 0, 0)


```