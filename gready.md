
#### 942. 增减字符串匹配

```python

class Solution(object):
    def diStringMatch(self, s):
        """
        :type s: str
        :rtype: List[int]

由范围 [0,n] 内所有整数组成的 n + 1 个整数的排列序列可以表示为长度为 n 的字符串 s ，其中:

如果 perm[i] < perm[i + 1] ，那么 s[i] == 'I' 
如果 perm[i] > perm[i + 1] ，那么 s[i] == 'D' 
给定一个字符串 s ，重构排列 perm 并返回它。如果有多个有效排列perm，则返回其中 任何一个 。

输入：s = "IDID"
输出：[0,4,1,3,2]

解法1：贪心，第一个为I，取最小的元素，第一个为D，取最大的元素
解法2：遍历生成，会超时

        """
        lo = 0
        hi = n = len(s)
        perm = [0] * (n + 1)
        for i, ch in enumerate(s):
            if ch == 'I':
                perm[i] = lo
                lo += 1
            else:
                perm[i] = hi
                hi -= 1
        perm[n] = lo  # 最后剩下一个数，此时 lo == hi
        return perm

# 作者：LeetCode-Solution
# 链接：https://leetcode.cn/problems/di-string-match/solution/zeng-jian-zi-fu-chuan-pi-pei-by-leetcode-jzm2/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

        # n = len(s)

        # def search(path, mark, l):
        #     # print('path:', path, 'mark:', mark, 'l:', l)
        #     # print("len(path) == n + 1:", len(path) == n + 1, len(path), n+1)
        #     if len(path) == n + 1:
        #         return path[:]

        #     for i in range(n+1):
        #         if mark & 1 << (i+1) == 0:

        #             if l != -1:
        #                 if s[l] == "I":
        #                     if path[-1] > i:
        #                         continue
        #                 if s[l] == "D":
        #                     if path[-1] < i:
        #                         continue
        #             path.append(i)
        #             res = search(path, mark | 1 << (i+1), l+1)
        #             path.pop()
        #             if res:
        #                 return res

        #     return []

        # res =  search([], 0, -1)
        # return res
```