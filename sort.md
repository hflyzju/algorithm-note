
#### 1051. 高度检查器

```python

class Solution:
    def heightChecker(self, heights: List[int]) -> int:

        """
        学生需要按照 非递减 的高度顺序排成一行。
        返回满足 heights[i] != expected[i] 的 下标数量 。
        
        输入：heights = [1,1,4,2,1,3]
        输出：3 
        解释：
        高度：[1,1,4,2,1,3]
        预期：[1,1,1,2,3,4]
        下标 2 、4 、5 处的学生高度不匹配。
        
        
        题解：计数排序
        time: O(max(n, max_val))
        space: O(max_val)
        """

        cnt = [0] * max(heights)
        for h in heights:
            cnt[h-1] += 1

        ans = 0
        k = 0
        for i in range(len(cnt)):
            while cnt[i] > 0:
                if i+1 != heights[k]:
                    ans += 1
                k += 1
                cnt[i] -= 1
        return ans
```