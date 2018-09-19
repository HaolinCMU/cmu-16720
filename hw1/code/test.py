import numpy as np

x = np.arange(16).reshape(4,4)

rows = np.vsplit(x, 2)
print(rows)
cols = []
for row in rows:
    cols = np.hsplit(row, 2)
    print(cols)
