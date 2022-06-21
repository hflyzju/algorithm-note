
#### 3. 无重复字符的最长子串

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        """3. 无重复字符的最长子串

        题目：给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。
        题解：记录每个letter的last_index，如果遇到重复的字符，并且last_index>l，那么我们就需要更新l到last_index+1

输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。


        """
        last_index = dict()
        n = len(s)
        if n <= 0:
            return 0
        l, r = 0, 0
        max_sub_length = 0
        while r < n:
            if s[r] in last_index:
                if last_index[s[r]] + 1 > l:
                    l = last_index[s[r]] + 1
            max_sub_length = max(max_sub_length, r - l + 1)
            last_index[s[r]] = r
            r += 1
        return max_sub_length

```


#### 151. 颠倒字符串中的单词

```python
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str

        151. 颠倒字符串中的单词

输入：s = "the sky is blue"
输出："blue is sky the"

        起始和结束位置都可能存在空格

        解法：
        1. 先去除首位空格
        2. 双指针遍历每个word，并push到result里面
        3. 利用首位指针
        """

        n = len(s)
        left, right = 0, n - 1

        while left <= right and s[left] == " ":
            left += 1

        while left <= right and s[right] == " ":
            right -= 1

        result = []
        word = []
        while left <= right:
            if s[left] != ' ':
                word.append(s[left])
            else:
                if word:
                    result.append(''.join(word))
                    word = []
            left += 1

        if word:
            result.append(''.join(word))
        
        l, r = 0, len(result) - 1
        while l < r:
            result[l], result[r] = result[r], result[l]
            l += 1
            r -= 1
        
        return ' '.join(result)
        

```

#### 5. 最长回文串

```python

class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        if n < 2:
            return s
        max_len = 1
        begin = 0
        # dp[i][j] 表示 s[i..j] 是否是回文串
        dp = [[False] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = True
        # 递推开始
        # 先枚举子串长度
        for L in range(2, n + 1):
            # 枚举左边界，左边界的上限设置可以宽松一些
            for i in range(n):
                # 由 L 和 i 可以确定右边界，即 j - i + 1 = L 得
                j = i + L - 1
                # 如果右边界越界，就可以退出当前循环
                if j >= n:
                    break
                if s[i] != s[j]:
                    dp[i][j] = False 
                else:
                    if L <= 2: # 长度为1或者2，并且s[i]==s[j], 那么就是回文
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i + 1][j - 1]
                # 只要 dp[i][L] == true 成立，就表示子串 s[i..L] 是回文，此时记录回文长度和起始位置
                if dp[i][j] and L > max_len:
                    max_len = L
                    begin = i
        return s[begin:begin + max_len]

# 作者：LeetCode-Solution
# 链接：https://leetcode.cn/problems/longest-palindromic-substring/solution/zui-chang-hui-wen-zi-chuan-by-leetcode-solution/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```

#### 面试题 01.05. 一次编辑

```python
class Solution:
    def oneEditAway(self, first: str, second: str) -> bool:
        """字符串有三种编辑操作:插入一个字符、删除一个字符或者替换一个字符。 给定两个字符串，编写一个函数判定它们是否只需要一次(或者零次)编辑。
输入: 
first = "pale"
second = "ple"
输出: True

        题解：比较每一个字符
        1. 长度相等,m==n，只能用替换来操作，这里l+=1， r+=1
        2. 长度不相等, m>n, 只能用插入，这里l+=1, r不变
        """

        m, n = len(first), len(second)
        if m < n:
            return self.oneEditAway(second, first)
        if m - n > 1:
            return False
        l, r = 0, 0
        cnt = 0
        while l < m and r < n and cnt <= 1:
            if first[l] == second[r]:
                l += 1
                r += 1
            else:
                if m == n:
                    l += 1
                    r += 1
                    cnt += 1
                else:
                    l += 1
                    cnt += 1
        # print('cnt:', cnt)
        return cnt <= 1


```


#### 468. 验证IP地址

```python
class Solution:

    """
输入：queryIP = "172.16.254.1"
输出："IPv4"
解释：有效的 IPv4 地址，返回 "IPv4"

输入：queryIP = "2001:0db8:85a3:0:0:8A2E:0370:7334"
输出："IPv6"
解释：有效的 IPv6 地址，返回 "IPv6"

输入：queryIP = "256.256.256.256"
输出："Neither"
解释：既不是 IPv4 地址，又不是 IPv6 地址

题解：
1. 注意前置零 01.02.03.04 => False
2. 注意连续空格 0..1.2，0::2:2:1:1:2
    """

    def is_valid_number(self, ip, l, r):
        #print(ip[l:r+1])
        if not ip[l:r+1].isdigit():
            return False
        if r - l >= 1 and ip[l] == '0':
            return False
        if not 0 <= int(ip[l:r+1]) <= 255:
            return False
        return True

    def validateIPV4(self, ip):
        l, r = 0, 0
        n = len(ip)
        dot_cnt = 0
        number_cnt = 0
        while r < n:
            if ip[r] == '.':
                dot_cnt += 1
                if not self.is_valid_number(ip, l, r-1):
                    return False
                l = r + 1
                number_cnt += 1
            if r == n-1:
                if not self.is_valid_number(ip, l, r):
                    return False
                number_cnt += 1
            r += 1
            if dot_cnt > 3:
                return False
        if dot_cnt != 3 or number_cnt != 4:
            return False
        return True


    def is_valid_number_v6(self, ip, l, r):
        #print(ip[l:r+1])
        if l > r:
            return False
        if r - l + 1 > 4:
            return False
        for k in range(l, r+1):
            if not ('0' <= ip[k] <= '9' or 'a' <= ip[k] <= 'f' or 'A' <= ip[k] <= 'F'):
                return False
        return True


    def validateIPV6(self, ip):
        l, r = 0, 0
        n = len(ip)
        dot_cnt = 0
        number_cnt = 0
        while r < n:
            if ip[r] == ':':
                dot_cnt += 1
                if not self.is_valid_number_v6(ip, l, r-1):
                    return False
                l = r + 1
                number_cnt += 1
            if r == n-1:
                if not self.is_valid_number_v6(ip, l, r):
                    return False
                number_cnt += 1
            r += 1
            if dot_cnt > 7:
                return False
        if dot_cnt != 7 or number_cnt != 8:
            return False
        return True

    def validIPAddress(self, queryIP: str) -> str:
        if self.validateIPV4(queryIP):
            return "IPv4"
        elif self.validateIPV6(queryIP):
            return "IPv6"
        else:
            return "Neither"

```