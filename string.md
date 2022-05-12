
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