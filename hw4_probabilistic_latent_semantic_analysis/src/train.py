import os
from sys import argv

step = int(argv[1])
epoch = int(argv[2])

for i in range(step, step + epoch + 1):
    os.system("python3 hw4.py 0.2 0.6 %d 1 0" %i)
os.system("python3 hw4.py 0.2 0.6 %d 0 1" %step + epoch)
