


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