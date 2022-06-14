https://codetop.cc/home


### 树

#### 124. 二叉树中的最大路径和


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def maxPathSum(self, root):
        """124. 二叉树中的最大路径和
        :type root: TreeNode
        :rtype: int


输入：root = [1,2,3]
输出：6
解释：最优路径是 2 -> 1 -> 3 ，路径和为 2 + 1 + 3 = 6

        思路：
        1. search函数，计算以该节点为起点(包含该接点)，的最大和。
        2. 同时每次求最大路径和
        """

        self.max_path_sum = float('-inf')

        def search(node):
            """以该节点为起点(包含该接点)，的最大和"""
            if node is None:
                return node

            left = search(node.left)
            right = search(node.right)

            left = max(0, left)
            right = max(0, right)

            cur_max_path_sum = left + right + node.val
            self.max_path_sum = max(self.max_path_sum, cur_max_path_sum)

            return max(left, right) + node.val


        search(root)

        return self.max_path_sum

```

#### 124. 二叉树中的最大路径和


```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def maxPathSum(self, root):
        """124. 二叉树中的最大路径和
        :type root: TreeNode
        :rtype: int


输入：root = [1,2,3]
输出：6
解释：最优路径是 2 -> 1 -> 3 ，路径和为 2 + 1 + 3 = 6

        思路：
        1. search函数，计算以该节点为起点(包含该接点)，的最大和。
        2. 同时每次求最大路径和
        """

        self.max_path_sum = float('-inf')

        def search(node):
            """以该节点为起点(包含该接点)，的最大和"""
            if node is None:
                return node

            left = search(node.left)
            right = search(node.right)

            left = max(0, left)
            right = max(0, right)

            cur_max_path_sum = left + right + node.val
            self.max_path_sum = max(self.max_path_sum, cur_max_path_sum)

            return max(left, right) + node.val


        search(root)

        return self.max_path_sum

```
#### 236. 二叉树的最近公共祖先

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        
        Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
        Output: 3
        Explanation: The LCA of nodes 5 and 1 is 3.

        题解：https://mp.weixin.qq.com/s/njl6nuid0aalZdH5tuDpqQ
        """
        

        def search(node):

            if node is None:
                return None

            if node.val == p.val or node.val == q.val:
                return node
            
            left = search(node.left)
            right = search(node.right)

            if left is not None and right is not None:
                return node
            
            return left if left is not None else right

        return search(root)
```

#### 450. 删除二叉搜索树中的节点

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):

    def get_min_val(self, node):
        while node.left:
            node = node.left
        return node.val


    def deleteNode(self, root, key):
        """450. 删除二叉搜索树中的节点
        :type root: TreeNode
        :type key: int
        :rtype: TreeNode
输入：root = [5,3,6,2,4,null,7], key = 3
输出：[5,4,6,2,null,null,7]
解释：给定需要删除的节点值是 3，所以我们首先找到 3 这个节点，然后删除它。
一个正确的答案是 [5,4,6,2,null,null,7], 如下图所示。
另一个正确答案是 [5,2,6,null,4,null,7]。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/delete-node-in-a-bst
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

        题解：
        https://www.bilibili.com/video/BV1JQ4y1Z7Sj?spm_id_from=333.337.search-card.all.click
        1. 如果没有左右节点（叶子节点），直接删除
        2. 如果左节点不存在，则替换为右节点，否则替换为左节点
        3. 如果左右节点都存在，用又节点下的最小值替换当前节点，然后直接删除
        """
        if root is None:
            return None

        if root.val == key:
            if root.left is None and root.right is None:
                return None
            elif root.left is None:
                return root.right
            elif root.right is None:
                return root.left
            else:
                min_val = self.get_min_val(root.right)
                root.val = min_val
                root.right = self.deleteNode(root.right, min_val)
        elif root.val < key:
            root.right = self.deleteNode(root.right, key)
        else:
            root.left = self.deleteNode(root.left, key)

        return root

```

#### 103. 二叉树的锯齿形层序遍历

```python

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def zigzagLevelOrder(self, root):
        """z字型层次遍历
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []

        zflag = False
        cur_cache = deque()
        cur_cache.append(root)
        rt = []
        while cur_cache:
            cur_res = []
            next_cache = deque()
            while cur_cache:
                cur = cur_cache.popleft()
                cur_res.append(cur.val)
                if cur.left:
                    next_cache.append(cur.left)
                if cur.right:
                    next_cache.append(cur.right)
            cur_cache = next_cache
            if not zflag:
                rt.append(cur_res)
            else:
                rt.append(cur_res[::-1])
            zflag = not zflag
        return rt
```

#### 968. 监控二叉树 - 方法一，树形dp

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def minCameraCover(self, root):
        """
        :type root: TreeNode
        :rtype: int

题目：
给定一个二叉树，我们在树的节点上安装摄像头。
节点上的每个摄影头都可以监视其父对象、自身及其直接子对象。
计算监控树的所有节点所需的最小摄像头数量。

输入：[0,0,null,0,0]
输出：1
解释：如图所示，一台摄像头足以监控所有节点。

题解：每个节点的状态有，装相机，不装相机，被覆盖，不被覆盖。
1. 当前节点装相机->满足->最小需要多少相机。
2. 当前不装相机，满足的情况下->最小需要多少相机。
3. 当前不装相机，不满足（但是孩子需要满足）情况下->最小需要多少相机

1.左右满足或者不满足都可以
cur_set = min(l_set, l_not_set_but_cover, l_not_set_not_cover) + min(r_set, r_not_set_but_cover, r_not_set_not_cover) + 1
2. 左右一个装了就行
cur_not_set_but_cover = min(l_set + min(r_set, r_not_set_but_cover), r_set + min(l_not_set_but_cover, l_set))
3. 左右都满足就行，当前不满足
cur_not_set_not_cover = l_not_set_but_cover + r_not_set_but_cover
        """
        def dfs(root):
            """
            状态0：当前节点安装相机的时候，需要的最少相机数
            状态1：当前节点不安装相机，但是能被覆盖到的时候，需要的最少相机数
            状态2：当前节点不安装相机，也不能被覆盖到的时候，需要的最少相机数
            """
            if not root:
                return [1, 0, 0]
            l_set, l_not_set_but_cover, l_not_set_not_cover = dfs(root.left)
            r_set, r_not_set_but_cover, r_not_set_not_cover = dfs(root.right)
            # 左右装，不装满足，不装不满足都可以
            cur_set = min(l_set, l_not_set_but_cover, l_not_set_not_cover) + min(r_set, r_not_set_but_cover, r_not_set_not_cover) + 1
            # 左侧装，右侧装或者不装满足 或者右侧装，左侧装或者左侧不装满足
            cur_not_set_but_cover = min(l_set + min(r_set, r_not_set_but_cover), r_set + min(l_not_set_but_cover, l_set))
            # 当前不装不满足，可能由父节点来装，但是孩子需要覆盖到
            cur_not_set_not_cover = l_not_set_but_cover + r_not_set_but_cover
            return [cur_set, cur_not_set_but_cover, cur_not_set_not_cover]

        cur_set, cur_not_set_but_cover, cur_r_not_set_not_cover = dfs(root)
        return min(cur_set, cur_not_set_but_cover)


```


#### 968. 监控二叉树 - 方法二，贪心

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def minCameraCover(self, root):
        """
        :type root: TreeNode
        :rtype: int

题目：
给定一个二叉树，我们在树的节点上安装摄像头。
节点上的每个摄影头都可以监视其父对象、自身及其直接子对象。
计算监控树的所有节点所需的最小摄像头数量。

输入：[0,0,null,0,0]
输出：1
解释：如图所示，一台摄像头足以监控所有节点。

方法2：贪心

0，未放置未覆盖
1，放置
2，未放置覆盖

        """
        self.cnt = 0
        def dfs(root):
            """
            0，未放置未覆盖
            1，放置
            2，未放置覆盖
            """
            if not root:
                return 2
            left_state = dfs(root.left)
            right_state = dfs(root.right)
            # 有一个孩子没有覆盖到，那么当前必须放置了
            if left_state == 0 or right_state == 0:
                self.cnt += 1
                return 1
            # 如果孩子都有覆盖了，那么当前可以不放置了
            if left_state == 2 and right_state == 2:
                return 0
            # 有一个孩子放置了，当前可以不放置了
            if left_state == 1 or right_state == 1:
                return 2
            return -1
            
        # 检查一下根节点有没有被覆盖到
        root_state = dfs(root)
        if root_state == 0:
            self.cnt += 1
        return self.cnt


```

### 排序
#### 215 第k大的数

```python

class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """

        # 方法二：快速排序思想
        # [3,2,1,5,6,4]
        # 1,2,3,4,5,6

        def findKthLargestFromLR(nums, k, left, right):
            """[left, right]区间里面的第k大"""
            l, r = left, left
            while r < right:
                if nums[r] > nums[right]:
                    nums[l], nums[r] = nums[r], nums[l]
                    l += 1
                r += 1
            nums[l], nums[right] = nums[right], nums[l]
            index = l - left + 1
            if index == k:
                return nums[l]

            elif index > k:
                return findKthLargestFromLR(nums, k, left, l - 1)
            else:
                return findKthLargestFromLR(nums, k - index, l + 1, right)
                
        return findKthLargestFromLR(nums, k, 0, len(nums) - 1)





        # 方法一：最小堆保存k个最大的元素
        # cache = []
        # for num in nums:
        #     if len(cache) < k:
        #         heapq.heappush(cache, num)
        #     else:
        #         if cache[0] < num:
        #             heapq.heappop(cache)
        #             heapq.heappush(cache, num)
        # return cache[0]
```



### 链表

#### 206. 反转链表

```python

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """


        pre = None

        cur = head

        while cur:

            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp

        return pre

```

### 数组

#### 53. 最大子数组和

```python
class Solution(object):
    def maxSubArray(self, nums):
        """53. 最大子数组和
        :type nums: List[int]
        :rtype: int

输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
        """
        pre = 0
        max_sum = float('-inf')
        for num in nums:
            cur = pre + num
            max_sum = max(max_sum, cur)
            pre = max(0, cur)
        return max_sum

```

#### 56. 合并区间

```python
class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]


        [[1,5],[1,3],[1,2][2,6],[8,10],[15,18]]

        """
        intervals.sort(key=lambda x: [x[0], -x[1]])
        # print(intervals)
        result = []
        for i in range(len(intervals)):
            if not result:
                result.append(intervals[i])
            else:
                pre_x, pre_y = result[-1]
                cur_x, cur_y = intervals[i]
                if cur_x > pre_y:
                    result.append(intervals[i])
                else:
                    result[-1][1] = max(pre_y, cur_y)
        return result

```


#### 15 三数之和

```python
class Solution(object):
    def threeSum(self, nums):
        """找出所有三数之和为0的数字, 注意要去重
        # https://www.1point3acres.com/bbs/thread-728439-1-1.html
        :type nums: List[int]
        :rtype: List[List[int]]
        Example:
            #  Example 1:
            #  Input: nums = [-1,0,1,2,-1,-4]
            #  Output: [[-1,-1,2],[-1,0,1]]
        Solution:
            O(n**2)
        """

        n = len(nums)
        if n < 3:
            return []
        nums.sort() # [0,0,0,0,0]
        result = []
        for i in range(n):
            if i >= 1 and nums[i] == nums[i-1]:
                continue
            r = n - 1
            target = -nums[i]
            l = i + 1
            while l < r:
                # 跳过相同数字
                if l > i + 1 and nums[l] == nums[l - 1]:
                    l += 1
                else:
                    # 缩小范围
                    while l < r and nums[l] + nums[r] > target:
                        r -= 1
                    if l == r:
                        break
                    if nums[l] + nums[r] == target:
                        result.append([nums[i], nums[l], nums[r]])
                    l += 1
        return result

```

#### 42 接雨水

```python
class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        n = len(height)
        if n <= 2:
            return 0
        left_max_height = [0] * n
        right_max_height = [0] * n
        left_max = height[0]
        for i in range(1, n):
            left_max_height[i] = left_max
            left_max = max(height[i], left_max)
        right_max = height[n-1]
        for i in range(n-2, -1, -1):
            right_max_height[i] = right_max
            right_max = max(height[i], right_max)

        s = 0
        for i in range(n):
            s += max(min(left_max_height[i], right_max_height[i]) - height[i], 0)
        return s

```

### 字符串

#### 3. 无重复字符的最长子串

```python

class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """3. 无重复字符的最长子串
        :type s: str
        :rtype: int
给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。

输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
        """

        n = len(s)
        if n <= 1:
            return n
        char_cnt = defaultdict(int)
        pre_index = dict()
        l, r = 0, 0
        max_len = 0
        while r < n:
            char_cnt[s[r]] += 1
            if s[r] in pre_index and pre_index[s[r]] + 1 > l:
                l = pre_index[s[r]] + 1
                char_cnt[s[r]] -= 1
            pre_index[s[r]] = r
            max_len = max(max_len, r - l + 1)
            r += 1
            
        return max_len
            
```

#### 8 字符串转整数

```python
class Solution(object):
    def myAtoi(self, s):
        """
        :type s: str
        :rtype: int
        """

        MAX_VAL = ((1 << 31) - 1)
        MIN_VAL = -2 ** 31
        # print('MAX_VAL:', MAX_VAL)
        # print('MIN_VAL:', MIN_VAL)
        n = len(s)
        if n == 0:
            return 0
        start = 0
        while start < n and s[start] == ' ':
            start += 1

        # 1. 处理前面的符号
        sign = 1
        if start < n and s[start] == '-':
            sign = -1
            start += 1

        # badcase：-+12 
        if start < n and sign != -1 and s[start] == '+':
            start += 1

        # 2. 收集后面的数字
        res = 0
        while start < n:
            if s[start].isdigit():
                res = res * 10 + int(s[start]) * sign
                # print('res:', res)
                if res >= MAX_VAL:
                    return MAX_VAL
                if res <= MIN_VAL:
                    return MIN_VAL
            else:
                break
            start += 1

        return res

```

### 拓扑排序

#### 207. 课程表

```python

class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool

输入：numCourses = 2, prerequisites = [[1,0]]
输出：true
解释：总共有 2 门课程。学习课程 1 之前，你需要完成课程 0 。这是可能的。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/course-schedule
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
        """

        graph = defaultdict(set)
        indegree = [0] * numCourses
        for prerequisite in prerequisites:
            graph[prerequisite[1]].add(prerequisite[0])
            indegree[prerequisite[0]] += 1
        

        cache = deque()
        for i in range(numCourses):
            if indegree[i] == 0:
                cache.append(i)

        cnt = 0
        while cache:
            # print('cache:', cache)
            cur = cache.popleft()
            cnt += 1
            for child in graph[cur]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    cache.append(child)

        return cnt == numCourses
```

### 设计

#### 146. LRU 缓存

```python
class DLinkedNode:
    """双向链表"""
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    """
请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。
实现 LRUCache 类：
LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。
函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/lru-cache
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

    """

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
        if key not in self.cache:
            return -1
        # 如果 key 存在，先通过哈希表定位，再移到头部
        node = self.cache[key]
        self.moveToHead(node)
        return node.value

    def put(self, key: int, value: int) -> None:
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
        """将节点添加到最前面"""
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


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

```

### 二分

#### 4. 寻找两个正序数组的中位数

```python

class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """求两个排序数组的中位数

输入：nums1 = [1,3], nums2 = [2]
输出：2.00000
解释：合并数组 = [1,2,3] ，中位数 2

        解法：二分法
        """
        m = len(nums1)
        n = len(nums2)
        # 第一个数组的长度小于第二个数组的长度，这样可以确保k1和k2的值是有效的
        if m > n:
            m, n, nums1, nums2 = n, m, nums2, nums1
        l, r = 0, m
        while l <= r:
            k1 = l + (r - l) // 2  # 代表数组1左边的个数
            k2 = (m + n + 1) // 2 - k1 # 代表数组2左边的个数
            # k1 [0, m], k2 [0, n]
            # 只需要看k1是否是有效范围内就行了?
            # k2一定是有效范围内
            # [0, k1-1], [k1, m)
            # [0, k2-1], [k2, n)
            if k1 < m and nums2[k2-1] > nums1[k1]: # 右边还有空间，需要往右则可以往右移动，nums1[k1]需要存在,。
                l = k1 + 1
            elif k1 - 1 >= 0 and nums1[k1-1] > nums2[k2]: # 左边还有空间，需要往左则往左移动，nums[k1-1]存在
                r = k1 - 1
            else:
                if k1 == 0: # 第一个左边为0个
                    left_max = nums2[k2-1]
                elif k2 == 0: # 第二个左边为0个
                    left_max = nums1[k1-1]
                else: 
                    left_max = max(nums1[k1-1], nums2[k2-1]) # 都有
                if (m+n) % 2 == 1:
                    return left_max

                if k1 == m: # 第一个右边为0
                    right_min = nums2[k2]
                elif k2 == n:# 第二个右边为0
                    right_min = nums1[k1]
                else: 
                    right_min = min(nums2[k2], nums1[k1]) # 都有

                return (left_max + right_min) / 2.0


# class Solution(object):
#     def findMedianSortedArrays(self, nums1, nums2):
#         """
#         :type nums1: List[int]
#         :type nums2: List[int]
#         :rtype: float
#         """

#         m, n = len(nums1), len(nums2)

#         if m > n:
#             return self.findMedianSortedArrays(nums2, nums1)
#         k1, k2 = 0, m
#         while k1 < k2:
#             mid = (k1 + k2) >> 1
#             # [k1, mid) vs [mid, k2)
#             # [k3, mid2) [mid2, k4)
#             mid2 = (m + n + 1) // 2 - mid
#             if mid <= k2 and nums1[mid-1] > nums2[mid2]:
#                 k2 = mid - 1
#             elif mid < k2 and nums1[mid] < nums2[mid2-1]:
#                 k1 = mid + 1
#             else:
#                 if mid-1>= 0:
#                     left_max = max(nums1[mid-1], nums2[mid2-1])
#                 else:
#                     left_max = nums2[mid2-1]

#                 if (m + 2) & 1 != 0:
#                     return left_max

#                 if mid < m:
#                     right_min = min(nums1[mid], nums2[mid2])
#                 else:
#                     right_min = nums2[mid2]

#                 return (left_max + right_min) / 2.0

#         return -1
```

### 矩阵变换

#### 48. 旋转图像

```python

class Solution(object):
    def rotate(self, matrix):
        """旋转图像
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.

        Example:
            # Input: matrix = [[1,2,3],
                               [4,5,6],
                               [7,8,9]]
            # Output:         [[7,4,1],
                               [8,5,2],
                               [9,6,3]]

        题解：
            1. 对于左上1/4的区域内的点，与其他3个1/4区域的对应的点，共4个点做一次旋转即可。
            2. n为偶数，旋转左上四分之一，n为奇数，旋转x:[0, n/2], y:[0, n/2+1]的区域就行
        """
        m, n = len(matrix), len(matrix[0])
        for i in range(m / 2):
            for j in range((n + 1) / 2):
                tmp = matrix[i][j]
                # i=0, j = 1 from i = 2, j = 0
                matrix[i][j] = matrix[n-j-1][i]
                # i = 2, j = 0 from i = 3, j = 2
                matrix[n-j-1][i] = matrix[n-i-1][n-j-1]
                # i = 3, j = 2 from i = 1, j = 3
                matrix[n-i-1][n-j-1] = matrix[j][n-i-1]
                matrix[j][n-i-1] = tmp

        # n = len(matrix)
        # # 一次换四个点
        # # 总共只要换1/4的区域就行了,这1/4的区域按下面情况选出来就行
        # for i in range(n//2):
        #     for j in range(i, n - i - 1):
        #         matrix[i][j],matrix[j][n-i-1],matrix[n-i-1][n-j-1],matrix[n-j-1][i] = \
        #         matrix[n-j-1][i], matrix[i][j],matrix[j][n-i-1],matrix[n-i-1][n-j-1]
        # #print(matrix)
```


### 线段树


#### 3轮 307. 区域和检索 - 数组可修改

```python
class NumArray:
    def __init__(self, nums: List[int]):
        self.nums = nums
        # 初始化为0
        self.tree = [0] * (len(nums) + 1)
        for i, num in enumerate(nums):
            # 树状数组的index从1开始
            self.add(i+1, num)

    def lowbit(self, index):
        """求补码+1
        lowbit(index) = index & -index
        -index = ~index + 1, 相当于补码加1
        """
        return index & -index

    def add(self, index: int, val: int):
        """往树状数组里面更新值，index更新后，将其所有父节点相关知识都更新"""
        while index < len(self.tree):
            self.tree[index] += val
            # 树状数组index的父节点为 index + lowbit(index)
            # lowbit(index) = index & -index
            # -index = ~index + 1, 相当于补码加1
            index += self.lowbit(index)


    def prefixSum(self, index) -> int:
        """[1,index]的前缀和
        index该位置包括lowbit(index)个数字的和
        上一个需要加上的和就是index - lowbit(x)
        一直加到0就行
        """
        s = 0
        while index:
            s += self.tree[index]
            index -= self.lowbit(index)
        return s

    def update(self, index: int, val: int) -> None:
        """更新线段树和备份的nums值"""
        # 相当于tree的基础上，当前节点和其父节点加上val-self.nums[index]
        self.add(index + 1, val - self.nums[index])
        # 这里相当于有个nums的备份，也需要更新
        self.nums[index] = val

    def sumRange(self, left: int, right: int) -> int:
        """拿到区间和，right+1才是tree的区间,[left, right]->[left, right+1]"""
        return self.prefixSum(right + 1) - self.prefixSum(left)

# 作者：LeetCode-Solution
# 链接：https://leetcode.cn/problems/range-sum-query-mutable/solution/qu-yu-he-jian-suo-shu-zu-ke-xiu-gai-by-l-76xj/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# obj.update(index,val)
# param_2 = obj.sumRange(left,right)

```

#### 3轮方法2 线段树 307. 区域和检索 - 数组可修改

```python
from typing import List

class Node:

    def __init__(self, left, right, val):
        """构建[left, right]的树"""
        self.left = left
        self.right = right
        self.val = val
        self.sum = 0
        self.left_node = None
        self.right_node = None


class NumArray:

    def __init__(self, nums: List[int]):
        self.root = self.build_tree(nums, 0, len(nums ) -1)


    def build_tree(self, nums, left, right):
        if left > right:
            return None
        # 当前节点
        cur_node = Node(left, right, nums[left])
        if left == right:
            cur_node.sum = nums[left]
            return cur_node
        # 左右节点
        mid = left + (right - left) // 2
        cur_node.left_node = self.build_tree(nums, left, mid)
        cur_node.right_node = self.build_tree(nums, mid + 1, right)
        cur_node.sum = cur_node.left_node.sum + cur_node.right_node.sum
        return cur_node


    def update_node(self, node, index, val):
        """更新左右孩子中当前节点的值，并根据左右孩子的sum更新当前sum"""
        if node.left == node.right:
            node.val = val
            # 只有一个节点也需要更新sum
            node.sum = val
            return
        mid = node.left + (node.right - node.left) // 2
        if index > mid:
            self.update_node(node.right_node, index, val)
        else:
            self.update_node(node.left_node, index, val)
        # 根据左右孩子更新当前节点的sum
        node.sum = node.left_node.sum + node.right_node.sum

    def update(self, index: int, val: int) -> None:
        """更新index下节点的值"""
        self.update_node(self.root, index, val)


    def query(self, node, left, right):
        """查询某节点[node.left, node.right]下的[left, right]之间的区域和"""
        # node包括在范围内，直接向上输出
        # print('node.left, node.right, left, right:', node.left, node.right, left, right)
        if left > right or left > node.right or right < node.left:
            return 0
        if left <= node.left and node.right <= right:
            return node.sum
        left_acc = self.query(node.left_node, left, right)
        right_acc = self.query(node.right_node, left, right)
        return left_acc + right_acc




    def sumRange(self, left: int, right: int) -> int:
        """求区域和"""
        return self.query(self.root, left, right)




# Your NumArray object will be instantiated and called as such:

# ["NumArray","sumRange","sumRange","sumRange","update","update","update","sumRange","update","sumRange","update"]
# [[[0,9,5,7,3]],[4,4],[2,4],[3,3],[4,5],[1,7],[0,8],[1,2],[1,9],[4,4],[3,4]]

# nums = [1,2,3,4,5]
# nums = [0,9,5,7,3]
# obj = NumArray(nums)
# print(obj.sumRange(left=4,right=4))
# print(obj.sumRange(left=1,right=2))
# print(obj.sumRange(left=1,right=4))
# obj.update(index=1,val=3)
# print(obj.sumRange(left=1,right=2))





```



### SDE2020面试

#### 1轮 151. 颠倒字符串中的单词

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

#### 2轮 顺时针生成矩阵

```python

class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        """给你一个正整数 n ，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的 n x n 正方形矩阵 matrix 。

输入：n = 3
输出：[[1,2,3],[8,9,4],[7,6,5]]

        题解：
        分层模拟，注意用完一层就更新
        """



        l, r, t, b = 0, n - 1, 0, n - 1
        result = [[0 for _ in range(n)] for _ in range(n)]
        num, tar = 1, n * n
        while num <= tar:
            for i in range(l, r + 1): # left to right
                result[t][i] = num
                num += 1
            # 每次用完就更新
            t += 1
            for i in range(t, b + 1): # top to bottom
                result[i][r] = num
                num += 1
            # 每次用完就更新
            r -= 1
            for i in range(r, l - 1, -1): # right to left
                result[b][i] = num
                num += 1
            # 每次用完就更新
            b -= 1
            for i in range(b, t - 1, -1): # bottom to top
                result[i][l] = num
                num += 1
            # 每次用完就更新
            l += 1
        return result

# 作者：jyd
# 链接：https://leetcode.cn/problems/spiral-matrix-ii/solution/spiral-matrix-ii-mo-ni-fa-she-ding-bian-jie-qing-x/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

#### 3轮 307. 区域和检索 - 数组可修改

```python
class NumArray:
    def __init__(self, nums: List[int]):
        self.nums = nums
        # 初始化为0
        self.tree = [0] * (len(nums) + 1)
        for i, num in enumerate(nums):
            # 树状数组的index从1开始
            self.add(i+1, num)

    def lowbit(self, index):
        """求补码+1
        lowbit(index) = index & -index
        -index = ~index + 1, 相当于补码加1
        """
        return index & -index

    def add(self, index: int, val: int):
        """往树状数组里面更新值，index更新后，将其所有父节点相关知识都更新"""
        while index < len(self.tree):
            self.tree[index] += val
            # 树状数组index的父节点为 index + lowbit(index)
            # lowbit(index) = index & -index
            # -index = ~index + 1, 相当于补码加1
            index += self.lowbit(index)


    def prefixSum(self, index) -> int:
        """[1,index]的前缀和
        index该位置包括lowbit(index)个数字的和
        上一个需要加上的和就是index - lowbit(x)
        一直加到0就行
        """
        s = 0
        while index:
            s += self.tree[index]
            index -= self.lowbit(index)
        return s

    def update(self, index: int, val: int) -> None:
        """更新线段树和备份的nums值"""
        # 相当于tree的基础上，当前节点和其父节点加上val-self.nums[index]
        self.add(index + 1, val - self.nums[index])
        # 这里相当于有个nums的备份，也需要更新
        self.nums[index] = val

    def sumRange(self, left: int, right: int) -> int:
        """拿到区间和，right+1才是tree的区间,[left, right]->[left, right+1]"""
        return self.prefixSum(right + 1) - self.prefixSum(left)

# 作者：LeetCode-Solution
# 链接：https://leetcode.cn/problems/range-sum-query-mutable/solution/qu-yu-he-jian-suo-shu-zu-ke-xiu-gai-by-l-76xj/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# obj.update(index,val)
# param_2 = obj.sumRange(left,right)

```

#### 3轮方法2 线段树 307. 区域和检索 - 数组可修改

```python
from typing import List

class Node:

    def __init__(self, left, right, val):
        """构建[left, right]的树"""
        self.left = left
        self.right = right
        self.val = val
        self.sum = 0
        self.left_node = None
        self.right_node = None


class NumArray:

    def __init__(self, nums: List[int]):
        self.root = self.build_tree(nums, 0, len(nums ) -1)


    def build_tree(self, nums, left, right):
        if left > right:
            return None
        # 当前节点
        cur_node = Node(left, right, nums[left])
        if left == right:
            cur_node.sum = nums[left]
            return cur_node
        # 左右节点
        mid = left + (right - left) // 2
        cur_node.left_node = self.build_tree(nums, left, mid)
        cur_node.right_node = self.build_tree(nums, mid + 1, right)
        cur_node.sum = cur_node.left_node.sum + cur_node.right_node.sum
        return cur_node


    def update_node(self, node, index, val):
        """更新左右孩子中当前节点的值，并根据左右孩子的sum更新当前sum"""
        if node.left == node.right:
            node.val = val
            # 只有一个节点也需要更新sum
            node.sum = val
            return
        mid = node.left + (node.right - node.left) // 2
        if index > mid:
            self.update_node(node.right_node, index, val)
        else:
            self.update_node(node.left_node, index, val)
        # 根据左右孩子更新当前节点的sum
        node.sum = node.left_node.sum + node.right_node.sum

    def update(self, index: int, val: int) -> None:
        """更新index下节点的值"""
        self.update_node(self.root, index, val)


    def query(self, node, left, right):
        """查询某节点[node.left, node.right]下的[left, right]之间的区域和"""
        # node包括在范围内，直接向上输出
        # print('node.left, node.right, left, right:', node.left, node.right, left, right)
        if left > right or left > node.right or right < node.left:
            return 0
        if left <= node.left and node.right <= right:
            return node.sum
        left_acc = self.query(node.left_node, left, right)
        right_acc = self.query(node.right_node, left, right)
        return left_acc + right_acc




    def sumRange(self, left: int, right: int) -> int:
        """求区域和"""
        return self.query(self.root, left, right)




# Your NumArray object will be instantiated and called as such:

# ["NumArray","sumRange","sumRange","sumRange","update","update","update","sumRange","update","sumRange","update"]
# [[[0,9,5,7,3]],[4,4],[2,4],[3,3],[4,5],[1,7],[0,8],[1,2],[1,9],[4,4],[3,4]]

# nums = [1,2,3,4,5]
# nums = [0,9,5,7,3]
# obj = NumArray(nums)
# print(obj.sumRange(left=4,right=4))
# print(obj.sumRange(left=1,right=2))
# print(obj.sumRange(left=1,right=4))
# obj.update(index=1,val=3)
# print(obj.sumRange(left=1,right=2))





```



### 八面微软offer




#### 968. 监控二叉树 - 方法一，树形dp

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def minCameraCover(self, root):
        """
        :type root: TreeNode
        :rtype: int

题目：
给定一个二叉树，我们在树的节点上安装摄像头。
节点上的每个摄影头都可以监视其父对象、自身及其直接子对象。
计算监控树的所有节点所需的最小摄像头数量。

输入：[0,0,null,0,0]
输出：1
解释：如图所示，一台摄像头足以监控所有节点。

题解：每个节点的状态有，装相机，不装相机，被覆盖，不被覆盖。
1. 当前节点装相机->满足->最小需要多少相机。
2. 当前不装相机，满足的情况下->最小需要多少相机。
3. 当前不装相机，不满足（但是孩子需要满足）情况下->最小需要多少相机

1.左右满足或者不满足都可以
cur_set = min(l_set, l_not_set_but_cover, l_not_set_not_cover) + min(r_set, r_not_set_but_cover, r_not_set_not_cover) + 1
2. 左右一个装了就行
cur_not_set_but_cover = min(l_set + min(r_set, r_not_set_but_cover), r_set + min(l_not_set_but_cover, l_set))
3. 左右都满足就行，当前不满足
cur_not_set_not_cover = l_not_set_but_cover + r_not_set_but_cover
        """
        def dfs(root):
            """
            状态0：当前节点安装相机的时候，需要的最少相机数
            状态1：当前节点不安装相机，但是能被覆盖到的时候，需要的最少相机数
            状态2：当前节点不安装相机，也不能被覆盖到的时候，需要的最少相机数
            """
            if not root:
                return [1, 0, 0]
            l_set, l_not_set_but_cover, l_not_set_not_cover = dfs(root.left)
            r_set, r_not_set_but_cover, r_not_set_not_cover = dfs(root.right)
            # 左右装，不装满足，不装不满足都可以
            cur_set = min(l_set, l_not_set_but_cover, l_not_set_not_cover) + min(r_set, r_not_set_but_cover, r_not_set_not_cover) + 1
            # 左侧装，右侧装或者不装满足 或者右侧装，左侧装或者左侧不装满足
            cur_not_set_but_cover = min(l_set + min(r_set, r_not_set_but_cover), r_set + min(l_not_set_but_cover, l_set))
            # 当前不装不满足，可能由父节点来装，但是孩子需要覆盖到
            cur_not_set_not_cover = l_not_set_but_cover + r_not_set_but_cover
            return [cur_set, cur_not_set_but_cover, cur_not_set_not_cover]

        cur_set, cur_not_set_but_cover, cur_r_not_set_not_cover = dfs(root)
        return min(cur_set, cur_not_set_but_cover)


```


#### 968. 监控二叉树 - 方法二，贪心

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def minCameraCover(self, root):
        """
        :type root: TreeNode
        :rtype: int

题目：
给定一个二叉树，我们在树的节点上安装摄像头。
节点上的每个摄影头都可以监视其父对象、自身及其直接子对象。
计算监控树的所有节点所需的最小摄像头数量。

输入：[0,0,null,0,0]
输出：1
解释：如图所示，一台摄像头足以监控所有节点。

方法2：贪心

0，未放置未覆盖
1，放置
2，未放置覆盖

        """
        self.cnt = 0
        def dfs(root):
            """
            0，未放置未覆盖
            1，放置
            2，未放置覆盖
            """
            if not root:
                return 2
            left_state = dfs(root.left)
            right_state = dfs(root.right)
            # 有一个孩子没有覆盖到，那么当前必须放置了
            if left_state == 0 or right_state == 0:
                self.cnt += 1
                return 1
            # 如果孩子都有覆盖了，那么当前可以不放置了
            if left_state == 2 and right_state == 2:
                return 0
            # 有一个孩子放置了，当前可以不放置了
            if left_state == 1 or right_state == 1:
                return 2
            return -1
            
        # 检查一下根节点有没有被覆盖到
        root_state = dfs(root)
        if root_state == 0:
            self.cnt += 1
        return self.cnt


```