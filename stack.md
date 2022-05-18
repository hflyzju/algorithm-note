

#### 224. 基本计算器 I
#### 227. 基本计算器 II

```python

from typing import List
from collections import deque

class Solution:
    def calculate(self, s):
        """实现计数器
        Example:
            input: 1 + 1 * (2 + 3) / 3
            output: 2

            input: 0 + 1 * (2 - 7) / 3
            output: -1

        """
        cache = deque()
        for si in s:
            cache.append(si)

        def calculate_bracket(cache):
            """计算当前括号内的值"""
            stack = []
            # 每次添加num的时候，看的是上一个遇到的sign的脸色
            pre_sign = '+'
            num = 0
            # 这里用deque可以一直从左消耗deque里面的元素，并且递归的时候也可以持续消耗
            while cache:
                cur = cache.popleft()
                if cur.isdigit():
                    num = num * 10 + int(cur)
                if cur == '(':
                    # 递归消耗cache
                    num = calculate_bracket(cache) # 递归拿到当前括号内的值
                # 如果不为数字且不为空，或者 到达终点了
                if (not cur.isdigit() and cur !=' ') or len(cache) == 0:
                    if pre_sign == '+':
                        stack.append(num)
                    elif pre_sign == '-':
                        stack.append(-num)
                    elif pre_sign == '*':
                        stack[-1] *= num
                    elif pre_sign == '/':
                        stack[-1] = int(stack[-1] / float(num))
                    num = 0
                    pre_sign = cur
                if cur ==')':
                    break
            return sum(stack)
        return calculate_bracket(cache)

s = Solution()

print(s.calculate("1 + 1 * (2 + 3) / 3")==2)
print(s.calculate("0 + 1 * (2 - 7) / 3")==-1)


```



#### 83 最大的矩形面积-单调栈

```python
class Solution(object):
    def largestRectangleArea(self, heights):
        """计算可能的最大的矩形的面积
        :type heights: List[int]
        :rtype: int
        Example:
            # Input: heights = [2,1,5,6,2,3]
            # Output: 10
            # heights = [2,1,5,6,2,3]
            # [2,5,6,110,111,111,3]
        Solution:
            0. 最核心是找到每个高度最长的左边界和右边界
            1. 单调栈，递增的时候就加栈里面，否者证明前面的高度需要计算了，因为后面没有比前面高的了
            2. 如何计算前面的高度，[2,5,6,110,111,111]，这里111先一直往回收缩，如果遇到比他小的了，那么111的宽度就出来了，这里是2
            3. [2,5,6,110] 计算完111后，还需要计算110，这种case，直到都比三小 -> [2, 3]
            4.[2, ..., 3] 这里代表2还没有被计算过，因为2后面的数都比2大，所以2可以一直往后计算
            5. 最后栈里面是一个递增数组，还需要最终算一下，因为存的是index，所以算的时候可以拿到w和h
        """
        n = len(heights)
        res = 0
        # 记录的是index
        stack = []
        for i in range(n):
            # 当当前stack不为空 并且新来的i小于最后一个index的高度，前面的高度111到头了，需要高为111时候的面积了
            # [2,5,6,110,111,111]  heights[i]=3
            while stack and heights[i] < heights[stack[-1]]:
                # 把最后的index拿出来，这个cur_height的宽度不可能再高了，需要在这个时候被算掉
                cur_height = heights[stack.pop()] # [2,5,6,110,111] cur_height=111
                # 如果还有和当前高度cur_height相等的，继续怼出来
                while stack and cur_height == heights[stack[-1]]:
                    stack.pop() # [2,5,6,110] cur_height=111
                if len(stack) > 0:
                    # [2,5,6,110, [111, 111], i]
                    # 这里相当于计算的是[111, 111]这个单前最高高度的宽度
                    # 后面再记录[100, '111', '111']的宽度
                    # 直到所有比i大的高度都计算完了
                    # 就可以考虑把下一个装进来了
                    # 这里就拿到cur_height的宽度了
                    cur_width = i - stack[-1] - 1
                else:
                    cur_width = i
                # 后面继续，如果有比heights[i]高的，这个时候都要被计算掉
                res = max(res, cur_height * cur_width)
            stack.append(i)

        # 跑完一轮里面可能还有东西
        # [111, 112, 113, 114]
        # 可能是一直递增的，这里还需要处理下
        while stack:
            cur_height = heights[stack.pop()]
            while len(stack) > 0 and cur_height == heights[stack[-1]]:
                stack.pop()
            if len(stack) > 0:
                cur_width = n - stack[-1] - 1
            else:
                cur_width = n
            res = max(res, cur_height * cur_width)
        return res

``