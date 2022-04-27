#### 146 LRU实现

```python
class DLinkedNode:
    """双向链表"""
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:

    def __init__(self, capacity: int):
        # key到双向节点的cache
        self.cache = dict()
        # 使用伪头部和伪尾部节点    
        # 初始化构建头节点和尾部节点
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.capacity = capacity
        self.size = 0

    def get(self, key: int) -> int:
        """查询节点并更新到最前面"""
        if key not in self.cache:
            return -1
        # 如果 key 存在，先通过哈希表定位，再移到头部
        node = self.cache[key]
        self.moveToHead(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        """添加节点，并更新到最前面"""
        if key not in self.cache:
            # 如果 key 不存在，创建一个新的节点
            node = DLinkedNode(key, value)
            # 添加进哈希表
            self.cache[key] = node
            # 添加至双向链表的头部
            self.addToHead(node)
            self.size += 1
            if self.size > self.capacity:
                # 如果超出容量，删除双向链表的尾部节点
                removed = self.removeTail()
                # 删除哈希表中对应的项
                self.cache.pop(removed.key)
                self.size -= 1
        else:
            # 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
            node = self.cache[key]
            node.value = value
            self.moveToHead(node)
    
    def addToHead(self, node):
        """将节点添加到最前面，更新当前节点和头结点的前后位置"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node 
    
    def removeNode(self, node):
        """删除单个节点"""
        node.prev.next = node.next
        node.next.prev = node.prev

    def moveToHead(self, node):
        """将节点移动到最前面
        1. 删除节点
        2. 在头节点添加节点
        """
        self.removeNode(node)
        self.addToHead(node)

    def removeTail(self):
        """删除尾部的节点"""
        node = self.tail.prev
        self.removeNode(node)
        return node

```