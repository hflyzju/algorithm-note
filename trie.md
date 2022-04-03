#### 472. Concatenated Words 检查words里面的字是否可以被多个字concat起来

```python

class Trie:
    def __init__(self):
        self.children = [None] * 26
        self.isEnd = False

    def insert(self, word: str):
        """插入一个字到trie树中"""
        node = self
        for ch in word:
            ch = ord(ch) - ord('a')
            # 检查有没有，没有的话就新建一个
            if not node.children[ch]:
                node.children[ch] = Trie()
            # 更新当前节点
            node = node.children[ch]
        # 标记结尾位置
        node.isEnd = True

    def contains(self, word: str):
        """检查word是否存在于trie树中"""
        node = self
        for ch in word:
            ch = ord(ch) - ord('a')
            if not node.children[ch]:
                return False
            node = node.children[ch]
        return node.isEnd

    def can_concat(self, word: str) -> bool:
        """检查一个word是否可以被前面的多个字concat起来
        """
        if self.contains(word): # without duplicates, so only in recursion loop can be true
            return True
        n = len(word)
        for sep in range(1, n):
            if self.contains(word[:sep]) and self.can_concat(word[sep:]):
                return True
        return False

class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        # sorted the word by the length first,
        # so we won't check the longer word first.
        # and we can put the short word into the trie
        # first 
        words.sort(key=len)
        ans = []
        root = Trie()
        for word in words:
            if word == "":
                continue
            if root.can_concat(word):
                ans.append(word)
            else:
                root.insert(word)
        return ans

# 作者：LeetCode-Solution
# 链接：https://leetcode-cn.com/problems/concatenated-words/solution/lian-jie-ci-by-leetcode-solution-mj4d/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```


#### 5302 加密解密字符串


```python

class Trie:
    def __init__(self):
        self.children = [None] * 26
        self.isEnd = False

    def insert(self, word: str):
        """插入一个字到trie树中"""
        node = self
        for ch in word:
            ch = ord(ch) - ord('a')
            # 检查有没有，没有的话就新建一个
            if not node.children[ch]:
                node.children[ch] = Trie()
            # 更新当前节点
            node = node.children[ch]
        # 标记结尾位置
        node.isEnd = True

    def contains(self, word: str):
        """检查word是否存在于trie树中"""
        node = self
        for ch in word:
            ch = ord(ch) - ord('a')
            if not node.children[ch]:
                return False
            node = node.children[ch]
        return node.isEnd
    

class Encrypter:

    def __init__(self, keys: List[str], values: List[str], c: List[str]):
        
        k2v = dict()
        v2keys = dict()
        for k, v in zip(keys, values):
            k2v[k] = v
            if v not in v2keys:
                v2keys[v] = []
            v2keys[v].append(k)
        
        self.k2v = k2v
        self.v2keys = v2keys
        t = Trie()
        for word in c:
            t.insert(list(word))
        
        self.t = t
        # print('done!')

    def encrypt(self, word1: str) -> str:
        res = []
        for w in word1:
            if w in self.k2v:
                res.append(self.k2v[w])
            else:
                res.append(w)
        return ''.join(res)
    
    def decrypt(self, word2: str) -> int:
        """decrypt word2有多少种解码方法，需要保证解码的字存在给定的wordlist中"""
        wl = []
        n = len(word2)
        for i in range(0, n, 2):
            wl.append(word2[i:i+2])
        # print('wl:', wl)
        
        
        self.cnt = 0
        m = len(wl)
        
        def search(i, node):
            """检查节点i的结果是否满足，满足的话就继续往下找"""
            if i >= m:
                if node.isEnd:
                    self.cnt += 1
                return
            cur = wl[i]
            if cur in self.v2keys:
                for k in self.v2keys[cur]:
                    ch = ord(k) - ord('a')
                    if node.children[ch] is not None:
                        search(i+1, node.children[ch])
                
        search(0, self.t)
        return self.cnt


# Your Encrypter object will be instantiated and called as such:
# obj = Encrypter(keys, values, dictionary)
# param_1 = obj.encrypt(word1)
# param_2 = obj.decrypt(word2)

```