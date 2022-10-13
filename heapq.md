

#### 295. 数据流的中位数

```python

class MedianFinder:

    def __init__(self):
        self.max_heapq = []
        self.min_heapq = []


    def addNum(self, num: int) -> None:
        if len(self.max_heapq) > len(self.min_heapq):
            # 需要往最小堆里面添加，先加到最大堆(其实也是以最小堆实现)，然后把最大堆里面的最小值拿出来
            heapq.heappush(self.max_heapq, num)
            min_val = heapq.heappop(self.max_heapq)
            heapq.heappush(self.min_heapq, -min_val)
        else:
            # 需要往最大堆里面提那家，先加到最小堆，然后把最小堆里面最大的元素拿出来
            heapq.heappush(self.min_heapq, -num)
            max_val = -heapq.heappop(self.min_heapq)
            heapq.heappush(self.max_heapq, max_val)
        # print("add:", num)
        # print("max_heapq:", self.max_heapq)
        # print("min_heapq:", self.min_heapq)
    def findMedian(self) -> float:
        """数据流的中位数
addNum(1)
addNum(2)
findMedian() -> 1.5
addNum(3) 
findMedian() -> 2

        题解：
            1. 用最大堆和最小堆维护大的一半和小的一半，例如:max_heapq=[3,4,5], min_heapq=[-2, -2, -1] 
            2. 如何保证大的在大堆，小的在小堆？往最大堆添加数字的时候，要先加入到最小堆，然后从最小堆拿出最大的元素放到最大堆，加入最小堆要反过来
        """
        if len(self.max_heapq) > len(self.min_heapq):
            return self.max_heapq[0]
        elif len(self.max_heapq) < len(self.min_heapq):
            return self.min_heapq[-1]
        else:
            return (self.max_heapq[0] - self.min_heapq[0]) / 2






# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
```

#### 407. 接雨水 II

```python
class Solution:
    def trapRainWater(self, heightMap: List[List[int]]) -> int:
        """二维接雨水

        输入: heightMap = [[1,4,3,1,3,2],
                          [3,2,1,3,2,4],
                          [2,3,3,2,3,1]]
        输出: 4
        解释: 下雨后，雨水将会被上图蓝色的方块中。总的接雨水量为1+2+1=4。

        来源：力扣（LeetCode）
        链接：https://leetcode.cn/problems/trapping-rain-water-ii
        著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

        题解：
        1. 利用优先队列，从最外圈往里面搜索，找到外圈最小的桶，然后计算相邻内部的最大盛水体积。
        2. 注意更新相邻内部节点的高度为max(点前外部节点高度，相邻内部节点高度)
        """

        m, n = len(heightMap), len(heightMap[0])

        cache = []
        visited = set()
        # 1. 最外面一圈入优先队列
        for i in range(m):
            heapq.heappush(cache, [heightMap[i][0], i, 0])
            heapq.heappush(cache, [heightMap[i][n-1], i, n-1])
            visited.add((i, 0))
            visited.add((i,n-1))
        for j in range(n):
            heapq.heappush(cache, [heightMap[0][j], 0, j])
            heapq.heappush(cache, [heightMap[m-1][j], m-1, j])
            visited.add((0,j))
            visited.add((m-1,j))

        s = 0
        while cache:
            cur_h, cur_x, cur_y = heapq.heappop(cache)
            for dx, dy in [[0, -1], [0, 1], [-1, 0], [1, 0]]:
                nx, ny = cur_x + dx, cur_y + dy
                if 0 <= nx < m and 0 <= ny < n and (nx, ny) not in visited:
                    tmp = cur_h - heightMap[nx][ny]
                    if tmp > 0:
                        s += tmp
                    visited.add((nx, ny))
                    # 高度取最高的那个
                    heapq.heappush(cache, [max(heightMap[nx][ny], cur_h), nx, ny])
        return s



```


#### 451. 根据字符出现频率排序

```python
class Solution:
    def frequencySort(self, s: str) -> str:
        word_cnt = defaultdict(int)
        for si in s:
            word_cnt[si] += 1

        cache = []
        for letter, cnt in word_cnt.items():
            heapq.heappush(cache, [-cnt, letter])

        res = []
        while cache:
            cnt, letter = heapq.heappop(cache)
            cnt = abs(cnt)
            while cnt:
                res.append(letter)
                cnt -= 1

        return ''.join(res)

```


#### 480. 滑动窗口中位数

```java

class Solution {
    public double[] medianSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        int cnt = n - k + 1;
        double[] ans = new double[cnt];
        // 如果是奇数滑动窗口，让 right 的数量比 left 多一个
        PriorityQueue<Integer> left  = new PriorityQueue<>((a,b)->Integer.compare(b,a)); // 滑动窗口的左半部分
        PriorityQueue<Integer> right = new PriorityQueue<>((a,b)->Integer.compare(a,b)); // 滑动窗口的右半部分
        for (int i = 0; i < k; i++) right.add(nums[i]);
        for (int i = 0; i < k / 2; i++) left.add(right.poll());
        ans[0] = getMid(left, right);
        for (int i = k; i < n; i++) {
            // 人为确保了 right 会比 left 多，因此，删除和添加都与 right 比较（left 可能为空）
            int add = nums[i], del = nums[i - k];
            if (add >= right.peek()) {
                right.add(add);
            } else {
                left.add(add);
            }
            if (del >= right.peek()) {
                right.remove(del);
            } else {
                left.remove(del);
            }
            adjust(left, right);
            ans[i - k + 1] = getMid(left, right);
        }
        return ans;
    }
    void adjust(PriorityQueue<Integer> left, PriorityQueue<Integer> right) {
        while (left.size() > right.size()) right.add(left.poll());
        while (right.size() - left.size() > 1) left.add(right.poll());
    }
    double getMid(PriorityQueue<Integer> left, PriorityQueue<Integer> right) {
        if (left.size() == right.size()) {
            return (left.peek() / 2.0) + (right.peek() / 2.0);
        } else {
            return right.peek() * 1.0;
        }
    }
}

// 作者：AC_OIer
// 链接：https://leetcode.cn/problems/sliding-window-median/solution/xiang-jie-po-su-jie-fa-you-xian-dui-lie-mo397/
// 来源：力扣（LeetCode）
// 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```


#### 630. 课程表 III


```python
class Solution:
    def scheduleCourse(self, courses: List[List[int]]) -> int:
        """这里有 n 门不同的在线课程，按从 1 到 n 编号。给你一个数组 courses ，其中 courses[i] = [durationi, lastDayi] 表示第 i 门课将会 持续 上 durationi 天课，并且必须在不晚于 lastDayi 的时候完成。你的学期从第 1 天开始。且不能同时修读两门及两门以上的课程。返回你最多可以修读的课程数目。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/course-schedule-iii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
输入：courses = [[100, 200], [200, 1300], [1000, 1250], [2000, 3200]]
输出：3
解释：
这里一共有 4 门课程，但是你最多可以修 3 门：
首先，修第 1 门课，耗费 100 天，在第 100 天完成，在第 101 天开始下门课。
第二，修第 3 门课，耗费 1000 天，在第 1100 天完成，在第 1101 天开始下门课程。
第三，修第 2 门课，耗时 200 天，在第 1300 天完成。
第 4 门课现在不能修，因为将会在第 3300 天完成它，这已经超出了关闭日期。

来源：力扣（LeetCode）
链接：https://leetcode.cn/problems/course-schedule-iii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
        题解：
            1. 先按结束时间从小到大排序，因为从小到大排列，那么前面能做的话，后面肯定能做
            2. 尝试修课程，不能修的话，尝试替换前面耗时长的课程，为后面的课程提供空间
            https://www.bilibili.com/video/BV1gy4y1B7uF?spm_id_from=333.337.search-card.all.click
        """

        courses.sort(key=lambda x:x[1])
        # print(courses)
        cache = []
        cur_day = 0
        for du, la in courses:
            if cur_day + du <= la:
                # 能做，直接放进去，更新cur_day
                heapq.heappush(cache, -du)
                cur_day += du
            else:
                # 当前因为填满了不能做了，但是如果当前的时间比最长的要小，可以考虑把前面最长的时间替换下来，给后面给大的空间
                if cache and du < -cache[0]:
                    pre_du = heapq.heappop(cache)
                    heapq.heappush(cache, -du)
                    cur_day -= (-pre_du - du)
            # print('du:', du, 'la:', la, 'cache:', cache, 'cur:', cur_day)
        return len(cache)

```