
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