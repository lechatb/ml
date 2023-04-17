import taichi as ti
import numpy as np

ti.init()

@ti.kernel
def edit_distance_kernel(s: ti.types.ndarray(dtype=ti.int32), t: ti.types.ndarray(dtype=ti.int32), d: ti.types.ndarray(dtype=ti.int32)):
    n, m = s.shape[0], t.shape[0]
    # Initialize matrix
    for i in range(n + 1):
        d[i, 0] = i
    for j in range(m + 1):
        d[0, j] = j
    # Calculate edit distance
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0
            if s[i - 1] == t[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i, j] = ti.min(d[i - 1,j]+1, d[i,j-1]+1, d[i-1,j-1]+cost)

def edit_distance_ti(s, t):
    n, m = len(s), len(t)
    s_arr = np.array([ord(c) for c in s], dtype=np.int32)
    t_arr = np.array([ord(c) for c in t], dtype=np.int32)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    edit_distance_kernel(s_arr, t_arr, d)
    return d[n, m]

s = "kitten"
t = "sitting"
print(edit_distance(s, t))  # Output: 3