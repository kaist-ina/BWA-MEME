import os
import sys


import random


global id_, id_2, count
count = 0
seq_num = 1000


id_  = 1000
id_2 = 1004

mutation_list = {}
mutation_list["A"]= ["C","G","T"]
mutation_list["C"]= ["A","G","T"]
mutation_list["G"]= ["A","C","T"]
mutation_list["T"]= ["A","C","G"]



def write_seq(wp, line, randomlist):
    global id_, id_2, count
    for elem in randomlist:

        if 'N' in line[elem:elem+151]:
            continue

        
        count += 1
        id_ += 101
        if count %50 ==1:
            id_2 += 1
            id_ = 1000
        #add variation in query
        
        wp.write("@A00718:106:HN2YVDSXX:3:1101:{}:{} 1:N:0:ATATGGAT+CTGTATTA\n".format(id_, id_2 ))

        query = [elem for elem in line[elem:elem+151]]

        # mutation
        for __ in range(5):
           if random.random() < 0.6:
               mutation_point = random.randint(0, 151) - 1
               query[mutation_point] = mutation_list[query[mutation_point]][random.randint(0,2)]

        # Ambiguous N
        if random.random() < 0.6:
            mutation_point = random.randint(0, 151) - 1
            query[mutation_point] = 'N'
            
        query = "".join(query)
        
        
        wp.write(query)
        
        wp.write("\n+\nFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF\n")
        
    wp.flush()

def write_boundary_seq(wp, line, randomlist):
    global id_, id_2, count
    for elem in randomlist:

        if 'N' in line[elem:elem+151]:
            continue

        
        count += 1
        id_ += 101
        if count %50 ==1:
            id_2 += 1
            id_ = 1000
        #add variation in query
        
        wp.write("@A00718:106:HN2YVDSXX:3:1101:{}:{} 1:N:0:ATATGGAT+CTGTATTA\n".format(id_, id_2 ))

        query = [elem for elem in line[elem:elem+151]]

        # mutation
        for __ in range(5):
           if random.random() < 0.6:
               mutation_point = random.randint(0, 10) - 1
               query[mutation_point] = mutation_list[query[mutation_point]][random.randint(0,2)]

        # Ambiguous N
        if random.random() < 0.6:
            mutation_point = random.randint(145, 151) - 1
            query[mutation_point] = 'N'

        # reverse
        if random.random() < 0.5:
            query.reverse()
            for idx,elem in enumerate(query):
                if elem== "A":
                    query[idx] = "T"
                if elem== "C":
                    query[idx] = "G"
                if elem== "G":
                    query[idx] = "C"
                if elem== "T":
                    query[idx] = "A"
        query = "".join(query)
        
        
        wp.write(query)
        
        wp.write("\n+\nFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF\n")
        
    wp.flush()    



ref_fp = open(sys.argv[1],"r")

query_fp = open("query_"+  os.path.basename(sys.argv[1]),"w")

for line in ref_fp:
    if len(line) < 300:
        continue

    len_ref = len(line)

    randomlist = random.sample(range(0,len(line)-1-151), min(seq_num, len(line)-153 ))
    boundary_list = []
    boundary_list.append(0)
    boundary_list.append(0)
    boundary_list.append(0)
    boundary_list.append(0)
    boundary_list.append(0)
    boundary_list.append(1)
    boundary_list.append(1)
    boundary_list.append(1)
    boundary_list.append(1)
    boundary_list.append(1)
    boundary_list.append(1)
    boundary_list.append(1)
    boundary_list.append(1)
    boundary_list.append(len_ref-1-151)
    boundary_list.append(len_ref-1-151)
    boundary_list.append(len_ref-1-151)
    boundary_list.append(len_ref-1-151)
    boundary_list.append(len_ref-1-151)
    boundary_list.append(len_ref-1-151)
    boundary_list.append(len_ref-1-154)
    boundary_list.append(len_ref-1-154)
    boundary_list.append(len_ref-1-154)
    boundary_list.append(len_ref-1-154)
    boundary_list.append(len_ref-1-154)
    boundary_list.append(len_ref-1-154)
    boundary_list.append(len_ref-1-154)
    write_boundary_seq(query_fp, line,boundary_list)
    write_seq(query_fp, line,randomlist)

    
    
