
#### 206. Reverse Linked List
```
Given the head of a singly linked list, reverse the list, and return the reversed list.

 

Example 1:


Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]
Example 2:


Input: head = [1,2]
Output: [2,1]
Example 3:

Input: head = []
Output: []
```


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
        while head:
            tmp = head.next
            head.next = pre
            pre = head
            head = tmp
        return pre
```

#### 92. Reverse Linked List II
```
Given the head of a singly linked list and two integers left and right where left <= right, reverse the nodes of the list from position left to position right, and return the reversed list.

 

Example 1:


Input: head = [1,2,3,4,5], left = 2, right = 4
Output: [1,4,3,2,5]
Example 2:

Input: head = [5], left = 1, right = 1
Output: [5]
```
```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reverseBetween(self, head, left, right):
        """
        :type head: ListNode
        :type left: int
        :type right: int
        :rtype: ListNode
        """
        
        if left == right:
            return head
        
        head2 = head
        pre = None
        cnt = 1
        tail = head
        while cnt < left:
            pre = head
            head = head.next
            cnt += 1
            tail = head
        # print(head.val)
            
        pre2 = None
        while cnt <= right:
            tmp = head.next
            head.next = pre2
            pre2 = head
            head = tmp
            cnt += 1
        # print(pre2)

        
        tail.next = head
        
        if pre:
            pre.next = pre2
            return head2
        else:
            return pre2
        
            

```

#### 86. Partition List

```
Given the head of a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.
You should preserve the original relative order of the nodes in each of the two partitions.
Example 1:
Input: head = [1,4,3,2,5,2], x = 3
Output: [1,2,2,4,3,5]
Example 2:
Input: head = [2,1], x = 2
Output: [1,2]
```

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        
        less = ListNode(-1)
        h1 = less
        big = ListNode(-1)
        h2 = big
        while head:
            if head.val < x:
                less.next = head
                less = less.next
            else:
                big.next = head
                big = big.next
            head = head.next
                
        less.next = h2.next
        big.next = None
        return h1.next

```


#### 19. Remove Nth Node From End of List


```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        
       s
                  f
        1 2 3 4 5
        """
        
        
        fast = head
        while fast and n:
            fast = fast.next
            n -= 1
            
        slow = head
        
        pre = None
        while slow and fast:
            pre = slow
            slow = slow.next
            fast = fast.next
            
        
        if pre:
            pre.next = slow.next
        else:
            head = head.next
            
        return head

```
#### 21. Merge Two Sorted Lists

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def mergeTwoLists(self, list1, list2):
        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        
        
        h1 = ListNode(-1)
        h2 = h1
        
        while list1 and list2:
            if list1.val > list2.val:
                h1.next = list2
                list2 = list2.next
            else:
                h1.next = list1
                list1 = list1.next
            h1 = h1.next
                
        if list1:
            h1.next = list1
        
        if list2:
            h1.next = list2
            
        return h2.next
        

```

#### 23. Merge k Sorted Lists

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        
        
        minheap = []
        for h in lists:
            if h:
                heapq.heappush(minheap, [h.val, h])
                
        h1 = h2 = ListNode(-1)
        
        while minheap:
            min_val, min_head = heapq.heappop(minheap)
            h1.next = min_head
            h1 = h1.next
            
            if min_head.next:
                heapq.heappush(minheap, [min_head.next.val, min_head.next])
                
        return h2.next

```


#### 24. Swap Nodes in Pairs


```
Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)

Example 1:
Input: head = [1,2,3,4]
Output: [2,1,4,3]
Example 2:
Input: head = []
Output: []
Example 3:
Input: head = [1]
Output: [1]
```


```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        
        -1 -> 1 -> 2 -> 3 -> 4
        
        -1 -> 2 -> 1
        """
        
        h2 = ListNode(-1)
        h2.next = head
        
        last = h2
        
        while head:
            if head.next:
                tmp = head.next.next
                last.next = head.next
                last.next.next = head
                last = head
                last.next = None
                head = tmp
            else:
                last.next = head
                break
                
        return h2.next
            
```


#### 25. Reverse Nodes in k-Group


```
Given the head of a linked list, reverse the nodes of the list k at a time, and return the modified list.

k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k then left-out nodes, in the end, should remain as it is.

You may not alter the values in the list's nodes, only nodes themselves may be changed.

 

Example 1:


Input: head = [1,2,3,4,5], k = 2
Output: [2,1,4,3,5]
Example 2:


Input: head = [1,2,3,4,5], k = 3
Output: [3,2,1,4,5]
```

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    
    
    def reverseFromLR(self, left, right):
        tail = left
        pre = None
        while left and left != right:
            tmp = left.next
            left.next = pre
            pre = left
            left = tmp
        right.next = pre
        return right, tail
            
    def reverseKGroup(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if k == 1:
            return head
        pre = ListNode(-1)
        pre.next = head
        h1 = pre
        cnt = 1
        while head:
            head = head.next
            cnt += 1
            if head and cnt % k == 0:
                tmp = head.next
                new_h, new_t = self.reverseFromLR(pre.next, head)
                pre.next = new_h
                pre = new_t
                pre.next = tmp
                head = tmp
                cnt += 1
        return h1.next
                
            
        

```

#### 61. Rotate List

```
Given the head of a linked list, rotate the list to the right by k places.

 

Example 1:


Input: head = [1,2,3,4,5], k = 2
Output: [4,5,1,2,3]
Example 2:


Input: head = [0,1,2], k = 4
Output: [2,0,1]

```

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        
        if not head:
            return head
        
        h = head
        cnt = 0
        while h:
            h = h.next
            cnt += 1
        
        if k % cnt == 0:
            return head
        
        if k > cnt:
            return self.rotateRight(head, k % cnt)
        
        fast = head
        while k:
            fast = fast.next
            k -= 1
        
        slow_pre = None
        fast_pre = None
        slow = head
        
        while slow and fast:
            slow_pre = slow
            fast_pre = fast
            fast = fast.next
            slow = slow.next
            
        slow_pre.next = None
        fast_pre.next = head
        
        return slow
        
        

```


#### 82. Remove Duplicates from Sorted List II


```
Given the head of a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list. Return the linked list sorted as well.
Example 1:
Input: head = [1,2,3,3,4,4,5]
Output: [1,2,5]
Example 2:
Input: head = [1,1,1,2,3]
Output: [2,3]
```


```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        
        if not head:
            return head
        pre = ListNode(-600)
        pre.next = head
        h1 = pre
        pre1 = head
        head = head.next
        
        dup = False
        while head:
            if head.val == pre1.val:
                dup = True
                head = head.next
            else:
                if dup:
                    pre.next = head
                    pre1 = head
                    head = head.next
                    dup = False
                else:
                    pre1.next = head
                    pre = pre1
                    pre1 = head
                    dup = False
                    head = head.next
                    
        if dup:
            pre.next = head
        return h1.next
```


#### 83. Remove Duplicates from Sorted List


```
Given the head of a sorted linked list, delete all duplicates such that each element appears only once. Return the linked list sorted as well.

 

Example 1:


Input: head = [1,1,2]
Output: [1,2]
Example 2:


Input: head = [1,1,2,3,3]
Output: [1,2,3]
```

```python

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return head
        pre = head
        h1 = head
        head = head.next
        dup = False
        while head:  
            if head.val != pre.val:
                pre.next = head
                pre = head
                dup = False
            else:
                dup = True
            head = head.next
        if dup:
            pre.next = head
        
        return h1
        
```



#### 109. Convert Sorted List to Binary Search Tree

```
Given the head of a singly linked list where elements are sorted in ascending order, convert it to a height balanced BST.
For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.
Example 1:
Input: head = [-10,-3,0,5,9]
Output: [0,-3,9,-10,null,5]
Explanation: One possible answer is [0,-3,9,-10,null,5], which represents the shown height balanced BST.
Example 2:
Input: head = []
Output: []

```


```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def sortedListToBST(self, head):
        """
        :type head: Optional[ListNode]
        :rtype: Optional[TreeNode]
        
        """
        
        if not head:
            return head
        
        if head.next is None:
            return TreeNode(head.val)
        
        h1 = head
        fast = head
        slow = head
        
        pre = ListNode(-10**5 - 1)
        while fast and fast.next and slow:
            fast = fast.next.next
            pre = slow
            slow = slow.next

        cur = TreeNode(slow.val)
        pre.next = None
        if h1:
            cur.left = self.sortedListToBST(h1)
        if slow.next:
            cur.right = self.sortedListToBST(slow.next)
        
        return cur
        

```

#### 114. Flatten Binary Tree to Linked List


```
Given the root of a binary tree, flatten the tree into a "linked list":

The "linked list" should use the same TreeNode class where the right child pointer points to the next node in the list and the left child pointer is always null.
The "linked list" should be in the same order as a pre-order traversal of the binary tree.
Example 1:
Input: root = [1,2,5,3,4,null,6]
Output: [1,null,2,null,3,null,4,null,5,null,6]
Example 2:

Input: root = []
Output: []
Example 3:

Input: root = [0]
Output: [0]
```

```python

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    
    
    def flatten_tree(self, node):
        if not node:
            return None, None
        if not node.left and not node.right:
            return node, node
        ls, le = self.flatten_tree(node.left)
        rs, re = self.flatten_tree(node.right)
        node.left = None
        if ls:
            node.right = ls
            le.right = rs
        else:
            node.right = rs
        if rs:
            return node, re
        else:
            return node, le
        
    
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: None Do not return anything, modify root in-place instead.
        
        [1,2,null,3]
        
          1
         2 null
        3
        """
        if not root:
            return root
        s, e = self.flatten_tree(root)
        return s
        
```


#### 116. Populating Next Right Pointers in Each Node

```
You are given a perfect binary tree where all leaves are on the same level, and every parent has two children. The binary tree has the following definition:

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

Initially, all next pointers are set to NULL.

 

Example 1:


Input: root = [1,2,3,4,5,6,7]
Output: [1,#,2,3,#,4,5,6,7,#]
Explanation: Given the above perfect binary tree (Figure A), your function should populate each next pointer to point to its next right node, just like in Figure B. The serialized output is in level order as connected by the next pointers, with '#' signifying the end of each level.
Example 2:

Input: root = []
Output: []

```

```python
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
        
        -1
      0.    1
     2 3.   4 5
    6 7 8 9 10 11 12 13
 
    
"""

class Solution(object):
    
    def connectLR(self, left, right):
        if not left and not right:
            return
        left.next = right
        # if left.right and right.left:
        #     left.right.next = right.left
        if left.left:
            self.connectLR(left.left, left.right)
        if right.left:
            self.connectLR(right.left, right.right)
        if left.right:
            self.connectLR(left.right, right.left)
            
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        if not root:
            return root
        self.connectLR(root.left, root.right)
        return root
        

```


#### 138. Copy List with Random Pointer

```
A linked list of length n is given such that each node contains an additional random pointer, which could point to any node in the list, or null.

Construct a deep copy of the list. The deep copy should consist of exactly n brand new nodes, where each new node has its value set to the value of its corresponding original node. Both the next and random pointer of the new nodes should point to new nodes in the copied list such that the pointers in the original list and copied list represent the same list state. None of the pointers in the new list should point to nodes in the original list.

For example, if there are two nodes X and Y in the original list, where X.random --> Y, then for the corresponding two nodes x and y in the copied list, x.random --> y.

Return the head of the copied linked list.

The linked list is represented in the input/output as a list of n nodes. Each node is represented as a pair of [val, random_index] where:

val: an integer representing Node.val
random_index: the index of the node (range from 0 to n-1) that the random pointer points to, or null if it does not point to any node.
Your code will only be given the head of the original linked list.

Example 1:
Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]
Example 2:
Input: head = [[1,1],[2,1]]
Output: [[1,1],[2,1]]
Example 3:
Input: head = [[3,null],[3,0],[3,null]]
Output: [[3,null],[3,0],[3,null]]

```


```python
"""
# Definition for a Node.
class Node:
    def __init__(self, x, next=None, random=None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        l
          r
        7 7 13 13 11 11 10 10 1 1
        """
        if not head:
            return None
        
        # 1. add node 
        h1 = head
        while head:
            nx = head.next
            head.next = Node(head.val)
            head.next.next = nx
            head = nx
            
        # 2. add random for new added node
        h2 = h1
        h3 = h1.next
        while h2 and h3:
            if h2.random:
                h3.random = h2.random.next
            h2 = h2.next.next
            if h2:
                h3 = h3.next.next  
        # 3. split
        h4 = h1  
        h5 = h1.next
        h6 = h5
        # h4
        #   h5
        # 7 7 13 13 11 11 10 10 1 1
        while True:
            if h4.next:
                h4.next = h4.next.next
                h4 = h4.next
            if h5.next:
                h5.next = h5.next.next
                h5 = h5.next
            if h4 is None or h5 is None:
                break
        return h6
        
        

```




#### 117. Populating Next Right Pointers in Each Node II


```
Given a binary tree

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

Initially, all next pointers are set to NULL.
Example 1:
Input: root = [1,2,3,4,5,null,7]
Output: [1,#,2,3,#,4,5,7,#]
Explanation: Given the above binary tree (Figure A), your function should populate each next pointer to point to its next right node, just like in Figure B. The serialized output is in level order as connected by the next pointers, with '#' signifying the end of each level.
Example 2:
Input: root = []
Output: []

```

```python
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
        
        -1
      0.    1
     2 3.   4 5
    6 7 8 9 10 11 12 13
 
    
"""

class Solution(object):
    
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        if not root:
            return root
        
        d = deque()
        d.append(root)
        pre = None
        while d:
            n = len(d)
            for i in range(n):
                cur = d.popleft()
                if pre:
                    pre.next = cur
                pre =cur
                if cur.left:
                    d.append(cur.left)
                if cur.right:
                    d.append(cur.right)
            pre = None
        return root
        

```


#### 141. Linked List Cycle

```
Given head, the head of a linked list, determine if the linked list has a cycle in it.
There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.
Return true if there is a cycle in the linked list. Otherwise, return false.
Example 1:
Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).
Example 2:
Input: head = [1,2], pos = 0
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 0th node.
Example 3:
Input: head = [1], pos = -1
Output: false
Explanation: There is no cycle in the linked list.
```


```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if not head or not head.next:
            return False
        
        fast = head.next
        slow = head
        while fast and fast.next:
            if fast == slow:
                return True
            fast = fast.next.next
            slow = slow.next
            
        return False

```


#### 142. Linked List Cycle II


```
Given the head of a linked list, return the node where the cycle begins. If there is no cycle, return null.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to (0-indexed). It is -1 if there is no cycle. Note that pos is not passed as a parameter.

Do not modify the linked list.

 

Example 1:


Input: head = [3,2,0,-4], pos = 1
Output: tail connects to node index 1
Explanation: There is a cycle in the linked list, where tail connects to the second node.
Example 2:


Input: head = [1,2], pos = 0
Output: tail connects to node index 0
Explanation: There is a cycle in the linked list, where tail connects to the first node.
Example 3:


Input: head = [1], pos = -1
Output: no cycle
Explanation: There is no cycle in the linked list.
```


```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return None
        
        fast = head
        slow = head
        
        has_cycle = False
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                has_cycle = True
                break
            
        if not has_cycle:
            return None
        
        # print('slow.val:', slow.val)
        
        h = head
        while h != slow:
            h = h.next
            slow = slow.next
        
        return slow

```


#### 143. Reorder List

```
You are given the head of a singly linked-list. The list can be represented as:

L0 → L1 → … → Ln - 1 → Ln
Reorder the list to be on the following form:

L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
You may not modify the values in the list's nodes. Only nodes themselves may be changed.

 

Example 1:


Input: head = [1,2,3,4]
Output: [1,4,2,3]
Example 2:


Input: head = [1,2,3,4,5]
Output: [1,5,2,4,3]
```

```python

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: None Do not return anything, modify head in-place instead.
        
                f
            s
        1 2 3 4 5
        
        123
        54
        
        1 5 2 4 3
        """
        # 1. find middle
        if not head or not head.next:
            return
        
        fast = slow = head
        pre = None
        while fast and fast.next:
            fast = fast.next.next
            pre = slow
            slow = slow.next
        
        # print('slow:', slow)
        
        if pre:
            pre.next = None
        
        # 2. reverse slow pointer
        pre = None
        while slow:
            nx = slow.next
            slow.next = pre
            pre = slow
            slow = nx
            
        # 1 2
        # 5 4 3
        # 1->5->2
        # 3. merge result
        h1 = head
        while h1 and pre:
            nx1 = h1.next
            nx2 = pre.next
            
            h1.next = pre
            if nx1:
                pre.next = nx1
            h1 = nx1
            pre = nx2

            
        
        
        
```


#### 146. LRU Cache

```
Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

Implement the LRUCache class:

LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
int get(int key) Return the value of the key if the key exists, otherwise return -1.
void put(int key, int value) Update the value of the key if the key exists. Otherwise, add the key-value pair to the cache. If the number of keys exceeds the capacity from this operation, evict the least recently used key.
The functions get and put must each run in O(1) average time complexity.

 

Example 1:

Input
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
Output
[null, null, null, 1, null, -1, null, -1, 3, 4]

Explanation
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // cache is {1=1}
lRUCache.put(2, 2); // cache is {1=1, 2=2}
lRUCache.get(1);    // return 1
lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
lRUCache.get(2);    // returns -1 (not found)
lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
lRUCache.get(1);    // return -1 (not found)
lRUCache.get(3);    // return 3
lRUCache.get(4);    // return 4
```


```python
class BiNode(object):
    
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.pre = None
        self.next = None
        

class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        
        
        
        
        1 2 3
        
        
        h -> 1 -> 2 -> 3 -> t
        
        """
        
        self.cap = capacity
        
        self.key2node = dict()
        
        self.h = BiNode(-1, -1)
        self.t = BiNode(-1, -1)
        
        self.h.next = self.t
        self.t.pre = self.h
        
    def remove_node(self, node):
        pre = node.pre
        nx = node.next
        pre.next = nx
        nx.pre = pre
        del self.key2node[node.key]
        
    def remove_least_node(self,):
        node = self.h.next
        self.remove_node(node)
        
        
    def add_node_to_tail(self, node):
        pre = self.t.pre
        pre.next = node
        node.pre = pre
        node.next = self.t
        self.t.pre = node
        

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        
        if key not in self.key2node:
            return -1
        node = self.key2node[key]
        self.remove_node(node)
        self.add_node_to_tail(node)
        self.key2node[key] = node
        return node.val
        

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        if key not in self.key2node:
            node = BiNode(key, value)
            self.key2node[key] = node
            if len(self.key2node) > self.cap:
                self.remove_least_node()
            self.add_node_to_tail(node)
        else:
            node = self.key2node[key]
            self.remove_node(node)
            node.val = value
            self.add_node_to_tail(node)
            self.key2node[key] = node
        
        


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

```



#### 147. Insertion Sort List

```
Given the head of a singly linked list, sort the list using insertion sort, and return the sorted list's head.

The steps of the insertion sort algorithm:

Insertion sort iterates, consuming one input element each repetition and growing a sorted output list.
At each iteration, insertion sort removes one element from the input data, finds the location it belongs within the sorted list and inserts it there.
It repeats until no input elements remain.
The following is a graphical example of the insertion sort algorithm. The partially sorted list (black) initially contains only the first element in the list. One element (red) is removed from the input data and inserted in-place into the sorted list with each iteration.

```

Example 1:


Input: head = [4,2,1,3]
Output: [1,2,3,4]
Example 2:


Input: head = [-1,5,3,4,0]
Output: [-1,0,3,4,5]

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def insertionSortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        h = head
        t = head
        cur = head.next
        t.next = None
        while cur:
            nx = cur.next
            if cur.val <= h.val:
                cur.next = h
                h = cur
            elif cur.val >= t.val:
                cur.next = None
                t.next = cur
                t = t.next
            else:
                pre = h
                h1 = h
                while h1.val < cur.val:
                    pre = h1
                    h1 = h1.next
                pre.next = cur
                cur.next = h1
            cur = nx
        return h

```


#### 148. Sort List

```
Given the head of a linked list, return the list after sorting it in ascending order.

 

Example 1:


Input: head = [4,2,1,3]
Output: [1,2,3,4]
Example 2:


Input: head = [-1,5,3,4,0]
Output: [-1,0,3,4,5]
Example 3:

Input: head = []
Output: []

```

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        
        if not head or not head.next:
            return head
        
        pre = head
        s, f = head.next, head.next.next
        while f and f.next:
            f = f.next.next
            pre = s
            s = s.next
            
        if pre:
            pre.next = None
        # print('head:', head)
        # print('s:', s)
            
        l = self.sortList(head)
        r = self.sortList(s)
        
        pre = ListNode(-1)
        h1 = pre
        
        while l or r:
            if l is None:
                pre.next = r
                break
            elif r is None:
                pre.next = l
                break
            else:
                if l.val > r.val:
                    pre.next = r
                    r = r.next
                else:
                    pre.next = l
                    l = l.next
                pre = pre.next
        return h1.next
        

```


#### 160. Intersection of Two Linked Lists

```
160. Intersection of Two Linked Lists
Easy

10519

980

Add to List

Share
Given the heads of two singly linked-lists headA and headB, return the node at which the two lists intersect. If the two linked lists have no intersection at all, return null.

For example, the following two linked lists begin to intersect at node c1:


The test cases are generated such that there are no cycles anywhere in the entire linked structure.

Note that the linked lists must retain their original structure after the function returns.

Custom Judge:

The inputs to the judge are given as follows (your program is not given these inputs):

intersectVal - The value of the node where the intersection occurs. This is 0 if there is no intersected node.
listA - The first linked list.
listB - The second linked list.
skipA - The number of nodes to skip ahead in listA (starting from the head) to get to the intersected node.
skipB - The number of nodes to skip ahead in listB (starting from the head) to get to the intersected node.
The judge will then create the linked structure based on these inputs and pass the two heads, headA and headB to your program. If you correctly return the intersected node, then your solution will be accepted.

 

Example 1:


Input: intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
Output: Intersected at '8'
Explanation: The intersected node's value is 8 (note that this must not be 0 if the two lists intersect).
From the head of A, it reads as [4,1,8,4,5]. From the head of B, it reads as [5,6,1,8,4,5]. There are 2 nodes before the intersected node in A; There are 3 nodes before the intersected node in B.
Example 2:


Input: intersectVal = 2, listA = [1,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
Output: Intersected at '2'
Explanation: The intersected node's value is 2 (note that this must not be 0 if the two lists intersect).
From the head of A, it reads as [1,9,1,2,4]. From the head of B, it reads as [3,2,4]. There are 3 nodes before the intersected node in A; There are 1 node before the intersected node in B.
Example 3:


Input: intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
Output: No intersection
Explanation: From the head of A, it reads as [2,6,4]. From the head of B, it reads as [1,5]. Since the two lists do not intersect, intersectVal must be 0, while skipA and skipB can be arbitrary values.
Explanation: The two lists do not intersect, so return null.
 

Constraints:

The number of nodes of listA is in the m.
The number of nodes of listB is in the n.
1 <= m, n <= 3 * 104
1 <= Node.val <= 105
0 <= skipA < m
0 <= skipB < n
intersectVal is 0 if listA and listB do not intersect.
intersectVal == listA[skipA] == listB[skipB] if listA and listB intersect.
 

Follow up: Could you write a solution that runs in O(m + n) time and use only O(1) memory?
```

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        12345412345
        41234512345
         12345
        412345
        """
        p1, p2 = headA, headB
        
        while p1 != p2:
            if not p1:
                p1 = headB
            else:
                p1 = p1.next
            
            if not p2:
                p2 = headA
            else:
                p2 = p2.next
                
        return p1
        
        

```


#### 234. Palindrome Linked List

```
234. Palindrome Linked List
Easy

10191

616

Add to List

Share
Given the head of a singly linked list, return true if it is a palindrome.

 

Example 1:


Input: head = [1,2,2,1]
Output: true
Example 2:


Input: head = [1,2]
Output: false
```


```python

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        
                f
            s
        
        1 2 2 1
        
        
                f
            s
        1 2 1 2 1
        """
        
        if not head:
            return False
        if not head.next:
            return True
        
        
        s, f = head.next, head.next.next
        
        pre = head
        while f and f.next:
            f = f.next.next
            pre = s
            s = s.next
        
        pre.next = None
        
        tmp = None
        while head:
            nx = head.next
            head.next = tmp
            tmp = head
            head = nx
            
        if f is not None:
            s = s.next
            
        # print('tmp:', tmp)
        # print('s:', s)
    
        while tmp and s:

            if tmp.val != s.val:
                return False
            tmp = tmp.next
            s = s.next
            
        return True
        

```