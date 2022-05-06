

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