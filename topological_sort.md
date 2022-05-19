

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