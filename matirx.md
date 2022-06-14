矩阵变换相关题解


#### 498. 对角线遍历

```python
class Solution:
    def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:

        """498. 对角线遍历，交替遍历对角线
给你一个大小为 m x n 的矩阵 mat ，请以对角线遍历的顺序，用一个数组返回这个矩阵中的所有元素。
输入：mat = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,4,7,5,3,6,8,9]

解法：
1. 总共有m+n-1条对角线，为每个对角线找到对应的起点
2. 当对角线层数layer>m或者layer>n时，需要注意一下起点的位置
        """
        m, n = len(mat), len(mat[0])
        res = []
        cnt = 0
        for layer in range(m + n - 1):
            if layer & 1 == 0:
                if layer < m:
                    start_x, start_y = layer, 0
                else:
                    # m=3, layer=4, y=2
                    start_x, start_y = m-1, layer - m + 1
                while start_x >= 0 and start_y < n and cnt < m * n:
                    res.append(mat[start_x][start_y])
                    start_x -= 1
                    start_y += 1
                    cnt += 1
            else:
                if layer < n:
                    start_x, start_y = 0, layer
                else:
                    # layer=3, n=3, x=1
                    start_x, start_y = layer - n + 1, n - 1
                while start_x < m and start_y >= 0 and cnt < m * n:
                    res.append(mat[start_x][start_y])
                    start_x += 1
                    start_y -= 1
                    cnt += 1
        return res

```