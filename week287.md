#### 1

```python

class Solution(object):
    def convertTime(self, a, b):
        """
        :type current: str
        :type correct: str
        :rtype: int
        """
        cnt = 0
        a_total = int(a[:2])
        b_total = int(b[:2])
        diff = (b_total - a_total) * 60
        
        c_total = int(a[3:])
        d_total = int(b[3:])
        
        diff += (d_total - c_total)
        
        while diff:
            for t in [60, 15, 5, 1]:
                while diff / t > 0:
                    cnt += (diff / t)
                    diff = diff % t
        
        return cnt
```
#### 2
```python

class Solution(object):
    def findWinners(self, a):
        """
        :type matches: List[List[int]]
        :rtype: List[List[int]]
        """
        
        win_cnt = defaultdict(int)
        loss_cnt = defaultdict(int)
        
        for m in a:
            win_cnt[m[0]] += 1
            loss_cnt[m[1]] += 1
            
        wl, ll = [], []
        for w in win_cnt:
            if win_cnt[w] > 0 and loss_cnt[w] <= 0:
                wl.append(w)
                
        for l in loss_cnt:
            if loss_cnt[l] == 1:
                ll.append(l)
                
        wl.sort()
        ll.sort()
        
        return wl, ll
```

#### 3 二分

```python
class Solution(object):
    def maximumCandies(self, nums, k):
        """
        :type candies: List[int]
        :type k: int
        :rtype: int
        """
        
        """
        k < len(num)
        
        return num.sort()[k]


        k > len(num)
        
        s = sum(k)
        mean = s // k
        min_val = mean(num)
        
        1,5,5, 4
        
        
        """
        nums.sort()
        n = len(nums)
        def check_k_val(val, k):
            cnt = 0
            for i in range(n-1, -1, -1):
                if nums[i] >= val:
                    cnt += (nums[i] // val)
                    if cnt >= k:
                        return True
                else:
                    break
            return False
        
        s = sum(nums)
        if k > s:
            return 0
        
        
        mean_val = s // k
        l, r = 1, mean_val
        max_val = 0
        while l <= r: # l=4, r=4
            # print('l:', l, 'r:',r)
            m = l + (r - l) // 2 # m = 4
            if check_k_val(m, k): # 
                # print('m:', m, 'yes!')
                max_val = m
                l = m + 1
            else:
                # print('m:', m, 'no!')
                r = m - 1
            

                
        return max_val

```

#### 4 trie

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