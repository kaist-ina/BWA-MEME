#!/bin/bash

#length="100 200 300"
#length="151"

#error_rate="0000 0002 0005 0010 0015 0050"
error_rate="15 20 30 40 50 60 70 80 90 100 110 120 130 140 150"
error_rate="11"
mkdir -p results

for e in ${error_rate};do
 
    echo "Length exact , Error rate ${e}"
    echo "Starting EXP for STAR"
    
    echo "" >results/star_query_len_exact_${e}.log
    numactl -m 1 -C 25-26 ./star_align ~/human_ref/human_g1k_v37.fasta  /ssd/wgsim/exact-${e}.fq 5000 2 4 2>> /dev/null >> results/star_query_len_exact_${e}.log
    
    echo "Starting EXP for MEME $f"
    # memtradeoff
    echo "" >results/meme_query_len_exact_${e}.log
    numactl -m 1 -C 25-26 ./meme_align ~/human_ref/human_g1k_v37.fasta  /ssd/wgsim/exact-${e}.fq 5000 2 4 2>> /dev/null >> results/meme_query_len_exact_${e}.log
    
    echo "Starting EXP for FMI $f"
    #FMI
    echo "" >results/fmi_query_len_exact_${e}.log
    numactl -m 1 -C 25-26 ./fmi_seeding_big_read ~/human_ref/human_g1k_v37.fasta  /ssd/wgsim/exact-${e}.fq 5000 2 1 2>> /dev/null >> results/fmi_query_len_exact_${e}.log

done;
