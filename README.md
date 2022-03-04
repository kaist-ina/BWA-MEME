## BWA-MEME: BWA-MEM emulated with a machine learning approach 
- BWA-MEME produces identical results as BWA-MEM2 and is 1.4x faster. 
- BWA-MEME builds upon BWA-MEM2 and includes performance improvements to the seeding. 
- BWA-MEME leverages **learned index** in **suffix array search**.
- BWA-MEME also provides feature to accomodate various memory size in servers.
---

## Contents
- [When to use BWA-MEME](#when-to-use-bwa-meme)
- [Performance of BWA-MEME](#performance-of-bwa-meme)
    + [The seeding module of BWA-MEME uses Learned-index. This, in turn, results in 3.32x higher seeding throughput compared to FM-index of BWA-MEM2.](#the-seeding-module-of-bwa-meme-uses-learned-index-this--in-turn--results-in-332x-higher-seeding-throughput-compared-to-fm-index-of-bwa-mem2)
    + [End-to-end alignment throughput is up to 1.4x higher than BWA-MEM2.](#end-to-end-alignment-throughput-is-up-to-14x-higher-than-bwa-mem2)
- [Getting Started](#getting-started)
  * [Compile the code](#compile-the-code)
  * [Training RMI - prerequisites](#training-rmi---prerequisites)
  * [Reference file download](#reference-file-download)
  * [Build index of the reference DNA sequence](#build-index-of-the-reference-dna-sequence)
  * [(Optional) Download pre-trained P-RMI learned-index model](#-optional--download-pre-trained-p-rmi-learned-index-model)
  * [Run alignment and compare SAM output](#run-alignment-and-compare-sam-output)
  * [Test scripts and executables are available in the test folder](#test-scripts-and-executables-are-available-in-the-test-folder)
- [Changing memory requirement for index in BWA-MEME](#changing-memory-requirement-for-index-in-bwa-meme)
- [Notes](#notes)
- [Citation](#citation)
---
## When to use BWA-MEME
- Anyone who use BWA-MEM in CPU-only machine (BWA-MEME requires 38GB of memory for index at minimal mode)
- Building high-throughput NGS alignment cluster with low Cost per Alignment throughput. CPU alignment can be cheaper than using hardware acceleration (GPU, FPGA)


## Performance of BWA-MEME
#### The seeding module of BWA-MEME uses Learned-index. This, in turn, results in 3.32x higher seeding throughput compared to FM-index of BWA-MEM2.
<img src="https://github.com/kaist-ina/BWA-MEME/blob/dev/images/BWA-MEME-SeedingResults.jpg" width="50%"/>

#### End-to-end alignment throughput is up to 1.4x higher than BWA-MEM2.
<img src="https://github.com/kaist-ina/BWA-MEME/blob/dev/images/BWA-MEME-AlignmentResults.png" width="50%" />

## Getting Started
### Compile the code
```sh
# Compile from source
git clone https://github.com/kaist-ina/BWA-MEME.git BWA-MEME
cd BWA-MEME

# To find out vectorization features supported in your machine
cat /proc/cpuinfo

# If AVX512BW (512-bit SIMD) is supported
make clean
make -j<num_threads> arch=avx512

# If AVX2 (256-bit SIMD) is supported
make clean
make -j<num_threads> arch=avx2

# If SSE4.1 (128-bit SIMD) is supported (default)
make -j<num_threads>
```
### Training RMI - prerequisites
To use the RMI train code, please [install Rust](https://rustup.rs/).

### Reference file download
You can download the reference using the command below.
```sh
# Download human_g1k_v37.fasta human genome and decompress it
wget ftp://ftp-trace.ncbi.nih.gov/1000genomes/ftp/technical/reference/human_g1k_v37.fasta.gz
gunzip human_g1k_v37.fasta.gz
```

### Build index of the reference DNA sequence
- Suffix array, Inverse suffix array 
```sh
# Build index (Takes ~4 hr for human genome with 32 threads. 1 hr for BWT, 3 hr for BWA-MEME)
./bwa-mem2 index -a meme -t <num_threads> <input.fasta>
```
- P-RMI learned-index model
```sh
# Run code below to train P-RMI, suffix array is required which is generated in index build code
./build_rmis_dna.sh <input.fasta>
```

### (Optional) Download pre-trained P-RMI learned-index model
```sh
# We provide the pretrained models for human_g1k_v37.fasta, please download in the link below.
# Two P-RMI model parameter files are required to run BWA-MEME
https://ina.kaist.ac.kr/~bwa-meme/
```


### Run alignment and compare SAM output
```sh
# Perform alignment with BWA-MEME, add -7 option
./bwa-mem2 mem -Y -K 100000000 -t <num_threads> -7 <input.fasta> <input_1.fastq> -o <output_meme.sam>

# To verify output with BWA-MEM2
./bwa-mem2 mem -Y -K 100000000 -t <num_threads> <input.fasta> <input_1.fastq> -o <output_mem2.sam>

# Compare output SAM files
diff <output_mem2.sam> <output_meme.sam>

# To diff large SAM files use https://github.com/unhammer/diff-large-files

```
### Test scripts and executables are available in the test folder

## Changing memory requirement for index in BWA-MEME 
1. Change ```#define MODE ``` number in src/LearnedIndex_seeding.h (set 1 for 38GB index size, 2 for 88GB index size, 3 for 118GB index size)
2. Recompile the code
3. Run alignment

## Notes

* BWA-MEME requires at least 64 GB RAM (with minimal acceleration BWA-MEME requires 38GB of memory). For WGS runs on human genome (>32 threads) with full acceleration of BWA-MEME, it is recommended to have 140-192 GB RAM.

* When deploying BWA-MEME with many threads (>72), mimalloc library is recommended to avoid performance drop issue.

## Citation

If you use BWA-MEME, please cite the following [paper](https://www.biorxiv.org/content/10.1101/2021.09.01.457579v1)
> **Youngmok Jung and Dongu Han. *BWA-MEME: BWA-MEM accelerated with a machine learning approach.*  (biorxiv).**



