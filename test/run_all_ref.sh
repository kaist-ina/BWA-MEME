#!/bin/bash

echo "" > result_var_ref.txt

numactl -m 1 -C 24-47,72-95 ./learned_seeding_big_read  ~/human_ref/human_g1k_v37.fasta  /ssd/wgsim/seeds_human.fq  5000 48 3 >> result_var_ref.txt

numactl -m 1 -C 24-47,72-95 ./learned_seeding_big_read  /ssd/reference/GCA_000002035.4_GRCz11_genomic.fna  /ssd/wgsim/seeds_zebra.fq  5000 48 3  >> result_var_ref.txt

numactl -m 1 -C 24-47,72-95 ./learned_seeding_big_read  /ssd/reference/GCF_000001635.27_GRCm39_genomic.fna  /ssd/wgsim/seeds_mouse.fq  5000 48 3  >> result_var_ref.txt

numactl -m 1 -C 24-47,72-95 ./learned_seeding_big_read_24  /ssd/reference/GCF_000002985.6_WBcel235_genomic.fna  /ssd/wgsim/seeds_celegan.fq  5000 48 3  >> result_var_ref.txt

numactl -m 1 -C 24-47,72-95 ./fmi_seeding_big_read  ~/human_ref/human_g1k_v37.fasta  /ssd/wgsim/seeds_human.fq  5000 48 3  >> result_var_ref.txt

numactl -m 1 -C 24-47,72-95 ./fmi_seeding_big_read  /ssd/reference/GCA_000002035.4_GRCz11_genomic.fna  /ssd/wgsim/seeds_zebra.fq  5000 48 3  >> result_var_ref.txt

numactl -m 1 -C 24-47,72-95 ./fmi_seeding_big_read  /ssd/reference/GCF_000001635.27_GRCm39_genomic.fna  /ssd/wgsim/seeds_mouse.fq  5000 48 3  >> result_var_ref.txt

numactl -m 1 -C 24-47,72-95 ./fmi_seeding_big_read  /ssd/reference/GCF_000002985.6_WBcel235_genomic.fna  /ssd/wgsim/seeds_celegan.fq  5000 48 3  >> result_var_ref.txt

#numactl -m 1 -C 24-47,72-95 ./learned_seeding_big_read  ~/human_ref/human_g1k_v37.fasta  /ssd/wgsim/seeds_human.fq  5000 48 3
