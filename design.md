

#### 381 设计插入删除随机O(1)的时间的数据结构


```python
from collections import defaultdict
import random

class RandomizedCollection(object):

    def __init__(self):

        self.cache = []
        self.val_to_indexlist = defaultdict(list)


    def insert(self, val):
        """
        :type val: int
        :rtype: bool
        """
        self.cache.append(val)
        self.val_to_indexlist[val].append(len(self.cache) - 1)
        # print('self.cache:', self.cache)
        return len(self.val_to_indexlist[val]) == 1



    def remove(self, val):
        """
        :type val: int
        :rtype: bool
        与最后的数字交换，然后pop掉最后的元素
        """
        if len(self.val_to_indexlist[val]) > 0:
            val_index = self.val_to_indexlist[val][-1]
            last_index = len(self.cache) - 1
            last_val = self.cache[last_index]
            self.val_to_indexlist[last_val].remove(last_index)
            self.val_to_indexlist[last_val].append(val_index) 
            self.cache[val_index], self.cache[last_index] = self.cache[last_index], self.cache[val_index]
            self.cache.pop()
            self.val_to_indexlist[val].pop()
            return True
        return False


    def getRandom(self):
        """
        :rtype: int
        """
        # print('self.cache:', self.cache)
        index = random.randint(0, len(self.cache)-1)
        # print('index:', index, 'len:', len(self.cache))
        return self.cache[index]

```



#### 535. TinyURL 的加密与解密

```
输入：url = "https://leetcode.com/problems/design-tinyurl"
输出："https://leetcode.com/problems/design-tinyurl"

解释：
Solution obj = new Solution();
string tiny = obj.encode(url); // 返回加密后得到的 TinyURL 。
string ans = obj.decode(tiny); // 返回解密后得到的原本的 URL 。


```

```python
class Codec:
    def __init__(self):
        self.database = {}
        self.id = 0

    def encode(self, longUrl: str) -> str:
        self.id += 1
        self.database[self.id] = longUrl
        return "http://tinyurl.com/" + str(self.id)

    def decode(self, shortUrl: str) -> str:
        i = shortUrl.rfind('/')
        id = int(shortUrl[i + 1:])
        return self.database[id]

# 作者：LeetCode-Solution
# 链接：https://leetcode.cn/problems/encode-and-decode-tinyurl/solution/tinyurl-de-jia-mi-yu-jie-mi-by-leetcode-ty5yp/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

        

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(url))

```