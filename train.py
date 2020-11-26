import os
from sys import argv

step = int(argv[1])
epoch = int(argv[2])

for i in range(step, step + epoch +1):
    os.system("python3 hw4.py 0.3 0.3 %d 1 false" %i)
