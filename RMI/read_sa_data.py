import os
import re
import numpy as np
from struct import unpack, pack
import sys

fp = open(sys.argv[1],"rb")

size = unpack("Q",fp.read(8))

print("Size of SA: ",size[0])

#for i in range(500):
#    val = unpack("Q",fp.read(8))
#    print("Val: ",val[0])

#for i in range(size[0]):
for i in range(300):
    val = unpack("Q",fp.read(8))
    print("Val: ",val[0])
    break
    #if i % 1000 == 0:
    #    print("Val: ",val[0])
    #if i >= size[0] - 2:
    #    print("Last Val: ",val[0])
    #if i == 500:
    #print("half Val: ",val[0])
    #    break

