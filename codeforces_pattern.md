

### binary search
#### search x in arr
```c++
int find_position(const vector<int>& a, int x) {
    int l = 0;
    int r = (int)a.size() - 1;   // [l, r] is our search space
    while (l <= r) {             // search space is non-empty
        int m = l + (r - l) / 2; // "middle" position in the range
        if (a[m] == x) return m; // found!
        else if (a[m] < x) {
            l = m + 1;           // remove all indices <= m from the search space
        } else {
            r = m - 1;           // remove all indices >= m from the search space
        }
    }
    return n;                    // failure
}

```