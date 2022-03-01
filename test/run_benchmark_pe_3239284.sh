#!/bin/bash

#length="100 200 300"

threads="12 48"

NUMA_48="numactl -m 1 -C 24-47,72-95"
NUMA_12="numactl -m 1 -C 24-29,72-77"

READ_1="/ssd2/ERR3239284_1.fastq.gz"
READ_2="/ssd2/ERR3239284_2.fastq.gz"

mkdir -p results


echo "" >results/benchmarks_se_err3239284.log

echo "Starting EXP for STAR"
/usr/bin/time -v $NUMA_12 ~/STAR/bin/Linux_x86_64/STAR --runThreadN 12  --genomeDir  ~/human_ref/ --outFileNamePrefix /ssd/output.sam --readFilesIn $READ_1 $READ_2  --readFilesCommand zcat   --outSAMtype SAM  >> results/benchmarks_se_err3239284.log

rm /ssd/output.samAligned.out.sam

date

echo "Starting EXP for Minimap2"
/usr/bin/time -v  $NUMA_12  ./minimap2/minimap2 -ax sr -t 12  ~/human_ref/human_g1k_v37.fasta $READ_1 $READ_2  > /ssd/output.sam 2>> results/benchmarks_se_err3239284.log

date

echo "Starting EXP for BWA-MEME"
/usr/bin/time -v $NUMA_12 ./BWA-MEME/bwa-mem2.avx512bw mem -7 -t 12 ~/human_ref/human_g1k_v37.fasta $READ_1 $READ_2 -o /ssd/output.sam 2>> results/benchmarks_se_err3239284.log

date

echo "Starting EXP for Whisper2"
ulimit -n 10000
/usr/bin/time -v $NUMA_12 /ssd2/Whisper/src/whisper -rp -out /ssd2/mappings -t 12 -temp /ssd2/Whisper/temp/ /ssd2/Whisper/index/human $READ_1 $READ_2  2>> results/benchmarks_se_err3239284.log

date
echo "Starting EXP for BWA-MEM2"
/usr/bin/time -v $NUMA_12 ./BWA-MEME/bwa-mem2.avx512bw mem -t 12 ~/human_ref/human_g1k_v37.fasta $READ_1 $READ_2 -o /ssd/output.sam 2>> results/benchmarks_se_err3239284.log

date
##### 48 thread experiment
echo "Starting EXP for STAR"
/usr/bin/time -v $NUMA_48 ~/STAR/bin/Linux_x86_64/STAR --runThreadN 48  --genomeDir  ~/human_ref/ --outFileNamePrefix /ssd/output.sam --readFilesIn $READ_1  $READ_2 --readFilesCommand zcat   --outSAMtype SAM  2>> results/benchmarks_se_err3239284.log
rm /ssd/output.samAligned.out.sam

date
echo "Starting EXP for Minimap2"
/usr/bin/time -v $NUMA_48  ./minimap2/minimap2 -ax sr -t 48  ~/human_ref/human_g1k_v37.fasta $READ_1 $READ_2  > /ssd/output.sam 2>> results/benchmarks_se_err3239284.log

date
echo "Starting EXP for BWA-MEME"
/usr/bin/time -v $NUMA_48 ./BWA-MEME/bwa-mem2.avx512bw mem -7 -t 48 ~/human_ref/human_g1k_v37.fasta $READ_1 $READ_2 -o /ssd/output.sam 2>> results/benchmarks_se_err3239284.log

date
echo "Starting EXP for BWA-MEM2"
/usr/bin/time -v $NUMA_48 ./BWA-MEME/bwa-mem2.avx512bw mem -t 48 ~/human_ref/human_g1k_v37.fasta $READ_1 $READ_2 -o /ssd/output.sam 2>> results/benchmarks_se_err3239284.log

date
echo "Starting EXP for bowtie2"
/usr/bin/time -v $NUMA_48 ./bowtie2/bowtie2 -t -p 48 -x ~/human_ref/human_g1k_v37 -1 $READ_1 -2 $READ_2 > /ssd/output.sam 2>> results/benchmarks_se_err3239284.log

date
echo "Starting EXP for bwa"
/usr/bin/time -v $NUMA_48 ~/bwa/bwa mem -t48  ~/human_ref/human_g1k_v37.fasta $READ_1 $READ_2 > /ssd/output.sam 2>> results/benchmarks_se_err3239284.log

date
echo "Starting EXP for bowtie2"
/usr/bin/time -v $NUMA_12 ./bowtie2/bowtie2 -t -p 12 -x ~/human_ref/human_g1k_v37 -1  $READ_1 -2 $READ_2 > /ssd/output.sam 2>> results/benchmarks_se_err3239284.log

date
echo "Starting EXP for bwa"
/usr/bin/time -v $NUMA_12 ~/bwa/bwa mem -t12  ~/human_ref/human_g1k_v37.fasta $READ_1 $READ_2 > /ssd/output.sam 2>> results/benchmarks_se_err3239284.log





