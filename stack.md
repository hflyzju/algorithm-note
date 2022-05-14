

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