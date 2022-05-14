


### 字符串

#### HJ17

```python

import sys

def check_valid(line):
    if len(line) <= 1:
        return False
    if line[0] not in ["A", "D", "W", "S"]:
        return False
    for i in range(1, len(line)):
        if not ('0'<=line[i]<='9'):
            return False
    return True

try:
    x, y = 0, 0
    while True:
        line = sys.stdin.readline().strip()
        if line == '':
            break
        lines = line.split(';')
        for line in lines:
            if check_valid(line):
                val = int(line[1:])
                if line[0] == 'A':
                    x -= val
                elif line[0] == "D":
                    x += val
                elif line[0] == "S":
                    y -= val
                elif line[0] == "W":
                    y += val
    print('%d,%d'%(x,y))
            
except:
    pass
```


#### HJ29

```python
import sys



def encoder(s):
    if s == 'z':
        return "A"
    if s == 'Z':
        return 'a'
    if s == '9':
        return '0'
    if '0'<=s<'9':
        return str(int(s)+1)
    if 'a'<=s<'z':
        return chr(ord(s) + 1 + ord('A') - ord('a'))
    else:
        return chr(ord(s) + 1 - ord('A') + ord('a'))

def decoder(s):
    if s == 'a':
        return "Z"
    if s == "A":
        return 'z'
    if '0'<s<='9':
        return str(int(s)-1)
    if s == '0':
        return '9'
    if 'a'<s<='z':
        return chr(ord(s) - 1).upper()
    else:
        return chr(ord(s) - 1).lower()

    
if 1:
    cnt = 0
    while True:
        line = sys.stdin.readline().strip()
#         if not line:
#             continue
            
#         print('line:', line, 'cnt:', cnt)
        if cnt == 0:
            result = []
            for si in line:
                result.append(encoder(si))
                
#             for k in result:
#                 print(k,end='')
#             print('\r')
            print(''.join(result))
#             print('\r')
            
        elif cnt == 1:
#             print('adaf')
            result = []
#             print('result:', result)
            for si in line:
#                 print('si:', si, 'decoder(si):', decoder(si))
                result.append(decoder(si))
#             print('result:', result)
            print(''.join(result))
#             print('\r')
#             print('result:', result)
#             for k in result:
#                 print(k,end='')
#             print('\r')
        else:
            break
#         print('line:',line)
        cnt += 1
        if line == '':
            break

# except:
#     pass

```


### 数组

#### 解码编码
```python

def check(s):
    if len(s) <= 8:
        return 0
    a, b, c, d = 0, 0, 0, 0
    for item in s:
        if ord('a') <= ord(item) <= ord('z'):
            a = 1
        elif ord('A') <= ord(item) <= ord('Z'):
            b = 1
        elif ord('0') <= ord(item) <= ord('9'):
            c = 1
        else:
            d = 1
    if a + b + c + d < 3:
        return 0
    for i in range(len(s)-3):
        # 如果存在重复的会有多个
#         print('i:',i, 's[i:i+3]:',s[i:i+3] ,'s.split(s[i:i+3]):', s.split(s[i:i+3]))
        if len(s.split(s[i:i+3])) >= 3:
            return 0
    return 1

while 1:
    try:
        print('OK' if check(input()) else 'NG')
    except:
        break

```

### 数学

#### HJ107 立方根

```python
# 牛顿迭代法求解立方根的思路：
# 令f(x) = x^3 - a，求解f(x) = x^3 - a = 0。
# 利用泰勒公式展开，即f(x)在x0处的函数值为：
# f(x) = f(x0) +f'(x0)(x-x0) = (x0^3-a) + (3x0^2)(x-x0) = 0，
# 解之得：x = x0 - (x0^3 - a) / (3x0^2)。
#     即 x = x - ((x*x*x - n) / (3*x*x));

# 拓展：求平方根用一个套路：
# 令f(x) = x^2 - a，求解f(x) = x^2 - a = 0。
# 利用泰勒公式展开，即f(x)在x0处的函数值为：
# f(x) = f(x0) +f'(x0)(x-x0) = (x0^2-a) + 2x0(x-x0) = 0，
# 解之得：x = x0 - (x0^2 - a) / 2x0
#     即 x = x - (x*x-a)/2x 可进一步化简为:=(x+a/x) / 2。

# 总结：
# 平方根与立方根的求解迭代公式：
# 新x = 旧x - f(x)/f'(x)
# 新x = 旧x - (x平方或者立方与输入数a的差)/f(x)求导数
'''
# 法一：牛顿迭代法
while True:
    try:
        a = float(input().strip())  # 获取输入的实数a
        e = 0.0001  # 设定一个精度值
        t = a  # 初始化立方根t的值为输入的值a
        while abs(t*t*t - a) > e:  # 差值没有达到精度，便一直更新立方根
    # x(i+1) = x(i) - f(xi)/f'(xi)
    # 更新后的x = 原x - (原x的立方-a)/f(原x)导数
            t = t - (t*t*t - a) * 1.0 / (3 * t*t)
        print("%.1f" % t)  # 当精度达到要求时，此时的立方根t便为输入实数的立方根解
    except:
        break
'''

# 法二：二分法

while True:
    try:
        a = float(input().strip())
        epsilon = 0.0001
        low = min(-1.0, a) #左边界 (-1, -3) -> (-3), (-1, -0.1) -> (-1)
        high = max(1.0, a) #右边界 (1, 3) -> (3), (1, 0.1) -> (1)
        ans = (low + high)/2
        while abs(ans**3 - a) >= epsilon:
            if ans**3 < a:
                low = ans
            else:
                high = ans
            ans = (low + high)/2.0
        print('%.1f' % ans)
    except:
        break

	




```

#### HJ90 有效ip

```python

while True:
    try:
        ip = input()
        c = ip.split('.')
        c = [_ for _ in c if _]
#         if len(c) != 4:
#             print("NO")
        all = [0, 0, 0, 0]  # 各部分合法性初始为假
        if len(c) != 4:  # 长度只能为4
            print("NO")
        else:
            for i in range(4):
                # ip地址四部分值介于0~255，且每部分值不需要占位（例如：0010.01.002.011不合法）
                if 0 <= int(c[i]) < 256 and len(c[i]) == len(str(int(c[i]))):
                    all[i] = 1
            if all == [1, 1, 1, 1]:  # 检验后四部分都为真，则IP合法性为真
                print("YES")
            else:
                print("NO")
    except:
        break


```


#### HJ52 编辑距离

```python
def editDistance(str1, str2):
    '''
    计算字符串str1和str2的编辑距离
    '''
    edit = [[i+j for j in range(len(str2)+1)] for i in range(len(str1)+1)]
    for i in range(1,len(str1)+1):
        for j in range(1,len(str2)+1):
            if str1[i-1] == str2[j-1]:
                d = 0
            else:
                d = 1
            edit[i][j] = min(edit[i-1][j]+1,edit[i][j-1]+1,edit[i-1][j-1]+d)
    return edit[len(str1)][len(str2)]

while True:
    try:
        print(editDistance(input(), input()))
    except:
        break


```


#### HJ50 四则运算

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
                if cur == '(' or cur == '{' or cur == '[':
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
                if cur ==')' or cur == '}' or cur == ']':
                    break
            return sum(stack)
        return calculate_bracket(cache)

s = Solution()

# print(s.calculate("1 + 1 * (2 + 3) / 3")==2)
# print(s.calculate("0 + 1 * (2 - 7) / 3")==-1)


while True:
    try:
        print(s.calculate(input()))
    except:
        break

```


#### HJ43 迷宫问题

```python

def dfs(i,j):
    dx = [0,0,-1,1]
    dy = [-1,1,0,0]
    if i == m-1 and j == n-1:
        for pos in route:
            print('('+str(pos[0])+','+str(pos[1])+')')
        return
    
    for k in range(4):
        x = i+dx[k]
        y = j+dy[k]
        if x>=0 and x<m and y>=0 and y<n and map1[x][y]==0:
            map1[x][y]=1
            route.append((x,y))
            dfs(x,y)
            map1[x][y]=0
            route.pop()
#     else:
#         return
            

while True:
    try:
        m,n = list(map(int,input().split()))
        map1=[]
        for i in range(m):
            s=list(map(int,input().split()))
            map1.append(s)
		#初始值是（0，0）将其标记为已经访问
        route = [(0,0)]
        map1[0][0]=1
        dfs(0, 0)
        
    except:
        break


```


#### HJ45 名字的漂亮度


```python
while True:
    try:
        n = int(input())
        for i in range(n):
            each_name = input()
            beauty = 0
            
            # 字典放名字中每种字母对应出现到次数
            dict1 = {}
            for c in each_name:
                dict1[c] = each_name.count(c)
                
            # 每种字母的出现次数从大到小排列
            times_list = sorted(dict1.values(), reverse=True)
            
            # 次数从大到小以此乘以26,25,24...
            for j in range(len(times_list)):
                beauty += (26 - j) * times_list[j]
            print(beauty)
        
    except:
        break

```


#### HJ103 Redraiment的走法



```python

"""
输入：
6
2 5 1 5 4 5 
复制
输出：
3
复制
说明：
6个点的高度各为 2 5 1 5 4 5
如从第1格开始走,最多为3步, 2 4 5 ，下标分别是 1 5 6
从第2格开始走,最多只有1步,5
而从第3格开始走最多有3步,1 4 5， 下标分别是 3 5 6
从第5格开始走最多有2步,4 5， 下标分别是 5 6
所以这个结果是3。     
"""

while True:
    try:
        n, nums = int(input()), list(map(int, input().split()))
        dp = [1] * n
        for i in range(n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        print(max(dp))
    except:
        break

```