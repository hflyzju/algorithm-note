### 一、基础性质

1. lowbit(x):x二进制最小位不为0的位置
2. lowbit(x) 取反再加1: ~x + 1 = -x
3. 树状数组index从1开始，这样好计算一点
4. 树状数组每个位置统计的前缀和的个数为lowbit(x)不为0的位置，例如1 -> lowbit(1) = 1, 那么a[1]只包含一个数
5. t[x]的父节点为t[x + lowbit(x)], 例如t[1] -> father = t[1 + 1] = t[2]
6. 学习链接：https://www.bilibili.com/video/BV1pE41197Qj?spm_id_from=333.337.search-card.all.click

0001->长度为1：t[1] = a[1]
0010->长度为2：t[2] = a[1] + a[2]
0011->长度为1：t[3] = a[3]
1000->长度为4：t[4] = a[4] + a[3] + a


### 二、题解

#### 307. 区域和检索 - 数组可修改

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