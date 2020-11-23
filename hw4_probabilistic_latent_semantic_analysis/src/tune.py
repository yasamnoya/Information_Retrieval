import os

for b in [0.85, 0.875, 0.9]:
    for k1 in [2.5, 2.7, 2.9, 3]:
        for k2 in [20]:
            message = "k1 = " + str(k1) + ", k2 = " + str(k2) + ", b = " + str(b)
            running = ("python3 hw2.py " + str(k1) + " " + str(k2) + " " + str(b))
            print(running)
            os.system(running)
            os.system("kaggle competitions submit -c 2020-information-retrieval-and-applications-hw2 -f ../result.csv -m \"" + message + "\"")
    
