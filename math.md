
#### 829. 连续整数求和

```python
class Solution(object):
    def consecutiveNumbersSum(self, n):
        """829. 连续整数求和
        :type n: int
        :rtype: int
        
        k为个数, a为起点
        (a + a + k - 1) * k = 2n
        (2a + k - 1) = 2n / k
        
        1. k的上界：2n / k > k => 2n > k**2
        2. 2n/k为整数，2n是k的倍数
        3. 2n/k + 1 - k 是2的倍数
        好像满足这几个条件就确定有解了
给定一个正整数 n，返回 连续正整数满足所有数字之和为 n 的组数 。 
输入: n = 5
输出: 2
解释: 5 = 2 + 3，共有两组连续整数([5],[2,3])求和后为 5。
        """
        n *= 2
        cnt = 0
        k = 1
        while k * k < n:
            if n % k == 0:
                if (n // k - k + 1) % 2 == 0:
                    cnt += 1
            k += 1
        return cnt

        # 直接法
        # 以start为起点的连续k个数字之和为(strat + (start+k-1))*k // 2 
        # strat = (2*n//k + 1 -k) // 2
        # 首先本身n为一组，再判断2-n个数的组合能否相加=n即可，枚举k=(2到n),根据k和n计算出start，当start<1时结束
        # res = 1
        # k = 1
        # while k * k < 2 * n:
        #     start = (2*n//k + 1 -k) // 2
        #     if start < 1:
        #         break
        #     elif n == (2*start + k - 1) * k // 2:
        #         res += 1
        # return res

# 作者：yhx-w
# 链接：https://leetcode.cn/problems/consecutive-numbers-sum/solution/shu-xue-by-yhx-w-fz0w/
# 来源：力扣（LeetCode）
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


```