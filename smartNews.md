


#### [LeetCode] 269、火星词典
```python

from collections import defaultdict
import heapq

def AlienDictionary2(word_list):
    """
现有一种使用字母的全新语言，这门语言的字母顺序与英语顺序不同。您有一个单词列表（从词典中获得的），该单词列表内的单词已经按这门新语言的字母顺序进行了排序。需要根据这个输入的列表，还原出此语言中已知的字母顺序。

示例：


输入:
[
“wrt”,
“wrf”,
“er”,
“ett”,
“rftt”
]

输出: “wertf”

    思路：拓扑排序，注意可能不能正确排序，输出空字符串

    """

    graph = defaultdict(set)
    graph2 = defaultdict(set)
    indegree = defaultdict(int)
    char_set = set()
    for w in word_list:
        for i in range(len(w)-1):
            if w[i] != w[i+1]:
                graph[w[i]].add(w[i+1])
                graph2[w[i+1]].add(w[i])
            char_set.add(w[i])
            char_set.add(w[i+1])
    # for child, parents in graph2.items():
    #     indegree[child] = len(parents)

    print('char_set:', char_set)
    cache = []
    for char in char_set:
        indegree[char] = len(graph2[char])
        if indegree[char] == 0:
            cache.append(char)
    print('indegree:', indegree)
    result = []
    while cache:
        cur = cache.pop()
        result.append(cur)
        for child in graph[cur]:
            indegree[child] -= 1
            if indegree[child] == 0:
                cache.append(child)

    if len(result) != len(char_set):
        return ""
    return ''.join(result)



if __name__ == '__main__':

    a = [ "wrt", "wrf", "er", "ett", "rftt" ]
    print(AlienDictionary2(a))



```


#### 236. Lowest Common Ancestor of a Binary Tree - 迭代方法


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
        """
        
        cache = []
        cache.append([root, [root]])
        left_path = None
        while cache:
            cur, cur_path = cache.pop()
            # print("cur:", cur.val)
            # print("cur_path:", cur_path)
            if cur.val == p.val:
                left_path = cur_path
                break
            if cur.right:
                cache.append([cur.right, cur_path + [cur.right]])
            if cur.left:
                cache.append([cur.left, cur_path + [cur.left]])
        right_path = None
        cache = [[root, [root]]]
        while cache:
            cur, cur_path = cache.pop()
            if cur.val == q.val:
                right_path = cur_path
                break
            if cur.right:
                cache.append([cur.right, cur_path + [cur.right]])
            if cur.left:
                cache.append([cur.left, cur_path + [cur.left]])
                
        if left_path is not None and right_path is not None:
            l = 0
            result = None
            while l < len(left_path) and l < len(right_path) and left_path[l].val == right_path[l].val:
                result = left_path[l]
                l += 1
            return result
        return None
```


#### 236. Lowest Common Ancestor of a Binary Tree - 递归方法

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
        
        if root is None:
            return None
        # 代表该节点下找到了p或者q
        if root.val == p.val or root.val == q.val:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        # 如果要从左右两个节点来找数字，那么返回root
        if left is not None and right is not None:
            return root
        return left if left is not None else right

```


#### 41. 缺失的第一个正数


```python
class Solution(object):
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int

41. 缺失的第一个正数
给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。
请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。
输入：nums = [1,2,0]
输出：3
        [1, 2, 0]
        题解1：
        1. 所有的值肯定会在 1-n之间。
        2. 先把不在1-n的nums[i]变为1
        3. 把index=abs(nums[i])-1的位置变为负数，重新遍历数组，不为负数的代表没有这个数字，那么就可以直接输出了


        题解2：
        把每个数字nums[i]移动到目标位置
        1. 遍历每个数字，对于1-n的数字，把他放到nums[i]-1的位置。
        2. 再次遍历数字找到缺失的那个位置
        
        [1,2,4]
        [1,2,4]
        [4,2,1]
        [9,2,1,6,7,8,3,2]
        [1,2,9,8,7,6,3,2]
        # 
        [-4,2,3,4,5,9,2,3,-1,-8]

        """

        # 方法2
        n = len(nums)
        for i in range(n):
            # print('i:', i)
            while 1 <= nums[i] <= n:
                # print('nums:', nums)
                key = nums[i] - 1
                if nums[key] != nums[i]:
                    nums[i], nums[key] = nums[key], nums[i]
                else:
                    break
                # print('nums 2:', nums)
        for i in range(n):
            if nums[i] != i + 1:
                return i + 1
        return len(nums) + 1
        # 方法1
        # n = len(nums)
        # if 1 not in nums:
        #     return 1

        # # 1. 先把不在1-n的处理掉
        # for i in range(n):
        #     if nums[i] <= 0 or nums[i] > n:
        #         nums[i] = 1
        # # 2. 在遍历这个数组，对存在的数据进行标记
        # # [2,1]
        # # n=2
        # # key=2-1=1
        # # nums[1] = 0 bas(nums)
        # for i in range(n):
        #     # 0 = 1 - 1
        #     # 1 = 2 - 1
        #     # print('nums:', nums)
        #     key = abs(nums[i]) - 1
        #     # print('key:', key)
        #     nums[key] = -abs(nums[key])
        #     # print("nums[key]:", nums[key])
        #     # print('nums:', nums)
        # # print('nums:', nums)

        # # 找到没有标记的数字
        # for i in range(n):
        #     if nums[i] > 0:
        #         return i + 1

        # return len(nums) + 1
```


#### 480. 滑动窗口中位数


```python

class Solution(object):
    def medianSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[float]

        1,2,3,4,

        5,6,7

        5,6,7,8

        """

        min_heap = [] # 5,6,7,4
        max_heap = [] # -3, -2, -1

        n = len(nums)
        res = []
        for i in range(n):
            if not min_heap:
                heapq.heappush(min_heap, nums[i])
            else:
                if nums[i] >= min_heap[0]:
                    heapq.heappush(min_heap, nums[i])
                else:
                    heapq.heappush(max_heap, -nums[i])
                
                # remove nums[i-k]
                if i - k >= 0:
                    if nums[i-k] >= min_heap[0]:
                        new_min_heap = []
                        remove_flag = False
                        for j in range(len(min_heap)):
                            if not remove_flag and min_heap[j] == nums[i-k]:
                                remove_flag = True
                                continue
                            heapq.heappush(new_min_heap, min_heap[j])
                        min_heap = new_min_heap
                    else:
                        new_max_heap = []
                        remove_flag = False
                        for j in range(len(max_heap)):
                            if not remove_flag and max_heap[j] == -nums[i-k]:
                                remove_flag = True
                                continue
                            heapq.heappush(new_max_heap, max_heap[j])
                        max_heap = new_max_heap

                while len(min_heap) - len(max_heap) > 1:
                    min_val = heapq.heappop(min_heap)
                    heapq.heappush(max_heap, -min_val)

                while len(max_heap) > len(min_heap):
                    max_val = heapq.heappop(max_heap)
                    heapq.heappush(min_heap, -max_val)
            # print('i:', i)
            if i >= k - 1:
                # print('i --> :', i)
                # print('max_heap:', max_heap)
                # print('min_heap:', min_heap)
                if len(min_heap) == len(max_heap):
                    # print("(min_heap[0] - max_heap[0]):", (min_heap[0] - max_heap[0]))
                    res.append((min_heap[0] - max_heap[0]) / 2.0)
                else:
                    res.append(min_heap[0])
                # print('res:', res)

        return res


```