
#### 752 开锁需要的最短步长

```python
class Solution(object):
    def openLock(self, deadends, target):
        """开锁需要的最短步长
        :type deadends: List[str]
        :type target: str
        :rtype: inta
        Example:
            # Input: deadends = ["8888"], target = "0009"
            # Output: 1
        Solution:
            1. bfs, 记得初始化和更新visited
        """

        def get_upper(cur_status_list, i):
            """往下拨动+1"""
            upper = cur_status_list[:]
            if upper[i] == '9':
                upper[i] = '0'
            else:
                upper[i] = str(int(upper[i]) + 1)
            return upper

        def get_down(cur_status_list, i):
            """往下拨动-1"""
            down = cur_status_list[:]
            if down[i] == '0':
                down[i] = '9'
            else:
                down[i] = str(int(down[i]) - 1)
            return down
        from collections import deque
        cache = deque()
        # cur_status_list, cur_status_string, step
        cache.append([['0', '0', '0', '0'], '0000', 0])
        visited = set()
        visited.add("0000")

        while cache:
            cur_status_list, cur_status_string, step = cache.popleft()
            if cur_status_string == target:
                return step
            if cur_status_string in deadends:
                continue
            for i in range(4):
                upper_cur_status_list = get_upper(cur_status_list, i)
                upper_cur_status_string = ''.join(upper_cur_status_list)
                if upper_cur_status_string not in visited:
                    visited.add(upper_cur_status_string)
                    cache.append([upper_cur_status_list, upper_cur_status_string, step + 1])

                down_cur_status_list = get_down(cur_status_list, i)
                down_cur_status_string = ''.join(down_cur_status_list)
                if down_cur_status_string not in visited:
                    visited.add(down_cur_status_string)
                    cache.append([down_cur_status_list, down_cur_status_string, step + 1])
        return -1

```


#### 773 移动到123450需要的最小的步数

```python
class Solution(object):

    def slidingPuzzle(self, board):
        """移动0，问移动多少步可以变成123450
        :type board: List[List[int]]
        :rtype: int
        Example:
            # Input: board = [[1,2,3],[4,0,5]]
            # Output: 1
        Solution:
            1. 每次找到0，与其周围的邻居换，记录每个位置的邻居（转化成为1d更好操作）
            2. bfs即可
        """
        from collections import deque
        # adjacents[i]代表转换成一维矩阵后能转换到哪个邻居
        adjacents = [
            [1, 3],
            [0, 2, 4],
            [1, 5],
            [0, 4],
            [5, 3, 1],
            [4, 2]
        ]
        visited = set()
        # 转换成1维度的string
        board_1d = []
        for i in range(len(board)):
            for j in range(len(board[0])):
                board_1d.append(str(board[i][j]))
        board_1d_string = ''.join(board_1d)
        visited.add(board_1d_string)
        zero_index = board_1d_string.find('0')
        cache = deque()
        cache.append([board_1d, board_1d_string, 0, zero_index])
        while cache:
            cur_board_1d, cur_board_string, step, cur_zero_index = cache.popleft()
            if cur_board_string == "123450":
                return step
            for adjacent_index in adjacents[cur_zero_index]:
                # print('cur_zero_index, adjacent_index:', cur_zero_index, adjacent_index)
                cur_board_1d[cur_zero_index], cur_board_1d[adjacent_index] = cur_board_1d[adjacent_index], cur_board_1d[cur_zero_index]
                adjacent_board_1d = ''.join(cur_board_1d)
                if adjacent_board_1d not in visited:
                    visited.add(adjacent_board_1d)
                    # 注意这里需要传入一个新的cur_board_1d， 不然会发生错误
                    cache.append([cur_board_1d[:], adjacent_board_1d, step + 1, adjacent_index])
                cur_board_1d[cur_zero_index], cur_board_1d[adjacent_index] = cur_board_1d[adjacent_index], cur_board_1d[cur_zero_index]

        return -1

```