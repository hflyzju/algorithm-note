
#### 2305. 公平分发饼干

```python
class Solution(object):
    def distinctNames(self, ideas):
        """
        :type ideas: List[str]
        :rtype: int

题目：随机选两个，交换首字母后不在ideas中，则结果+1

输入：ideas = ["coffee","donuts","time","toffee"]
输出：6
解释：下面列出一些有效的选择方案：
- ("coffee", "donuts")：对应的公司名字是 "doffee conuts" 。
- ("donuts", "coffee")：对应的公司名字是 "conuts doffee" 。
- ("donuts", "time")：对应的公司名字是 "tonuts dime" 。
- ("donuts", "toffee")：对应的公司名字是 "tonuts doffee" 。
- ("time", "donuts")：对应的公司名字是 "dime tonuts" 。
- ("toffee", "donuts")：对应的公司名字是 "doffee tonuts" 。
因此，总共有 6 个不同的公司名字。

下面列出一些无效的选择方案：
- ("coffee", "time")：在原数组中存在交换后形成的名字 "toffee" 。
- ("time", "toffee")：在原数组中存在交换后形成的两个名字。
- ("coffee", "toffee")：在原数组中存在交换后形成的两个名字。

题解：根据首字母分组，然后统一计算每组结果，i组可以转到j的个数*j组可以转到i的个数
        """

        cache = defaultdict(set)
        for name in ideas:
            cache[name[0]].add(name[1:])
            
        ans = 0
        alphas = list(cache.keys())
        for i in range(len(alphas)):
            a = alphas[i]
            for j in range(i+1, len(alphas)):
                b = alphas[j]
                ans += len(cache[a]-cache[b])*len(cache[b]-cache[a])*2
        return ans

```