# BWA-MEME: BWA-MEM emulated with a machine learning approach 
<div align="center">
<img src="images/DALL·E_logo_bwa.png" width="30%"  style="margin-left: auto; margin-right: auto; display: block;" />
</div>

- BWA-MEME generates **the same SAM output** as **BWA-MEM2** and the original **bwa mem** 0.7.17.
- BWA-MEME is optimized for CPU usage, achieving up to **1.4x higher** alignment throughput, no specialized hardware is required.
- The seeding throughput of BWA-MEME is up to **3.32x faster** than BWA-MEM2.
- BWA-MEME can adapt to a wide range of server memory sizes, from 38GB to 128GB.
- BWA-MEME provides runtime index-building that is equivalent to disk read speed of 3-5 GB per second.
  
Contact: [Youngmok Jung](https://quito418.github.io/quito418/), Dongsu Han

Email: tom418@kaist.ac.kr, dhan.ee@kaist.ac.kr

---

## Contents
* [When to use BWA-MEME](#when-to-use-bwa-meme)
* [Performance of BWA-MEME](#performance-of-bwa-meme)
* [Getting Started](#getting-started)
  + [Option 1. Bioconda](#install-option-1-bioconda)
  + [Option 2. Build locally](#install-option-2-build-locally)
* [Changing memory requirement for index in BWA-MEME](#changing-memory-requirement-for-index-in-bwa-meme)
* [Notes](#notes)
  + [Building pipeline with Samtools](#building-pipeline-with-samtools)
  + [Download BWA MEME indices and pretrained P-RMI model](#download-meme-indices-and-pretrained-p-rmi-model)
  + [Reference file download](#reference-file-download)
* [Citation](#citation)
---
## When to use BWA-MEME
- Anyone who use BWA-MEM or BWA-MEM2 in CPU-only machine (BWA-MEME requires 38GB of memory for index at minimal mode)
- Building high-throughput NGS alignment cluster with low cost/throughput. CPU-only alignment can be cheaper than using hardware acceleration (GPU, FPGA).
- Just add single option "-7" to deploy BWA-MEME instead of BWA-MEM2 (BWA-MEME does not change anything, except the speed).

## Performance of BWA-MEME
#### The seeding module of BWA-MEME uses Learned-index. This results in 3.32x higher seeding throughput compared to FM-index of BWA-MEM2.
<img src="https://github.com/kaist-ina/BWA-MEME/blob/master/images/BWA-MEME-SeedingResults.jpg" width="50%"/>

#### End-to-end alignment throughput is up to 1.4x higher than BWA-MEM2.
<img src="https://github.com/kaist-ina/BWA-MEME/blob/master/images/BWA-MEME-alignment_throughput.png" width="50%" />

---
## Getting Started
### Install Option 1. Bioconda
```sh
# Install with conda, bwa-meme and the learned-index train script "build_rmis_dna.sh" will be installed
conda install -c conda-forge -c bioconda bwa-meme

# Print version and Mode of compiled binary executable
# bwa-meme binary automatically choose the binary based on the SIMD instruction supported (SSE, AVX2, AVX512 ...)
# Other modes of bwa-meme is available as bwa-meme_mode1 or bwa-meme_mode2
bwa-meme version

```
### Build index of the reference DNA sequence
```sh
# Build index (Takes ~1hr for human genome)
# we recommend using at least 8 threads
bwa-meme index -a meme <input.fasta> -t <thread number>
```
### Training P-RMI
```sh
# Run code below to train P-RMI, suffix array is required which is generated in index build code
# takes about 15 minute for human genome with single thread
build_rmis_dna.sh <input.fasta>
```

### Run alignment and compare SAM output with BWA-MEM2
```sh
# Perform alignment with BWA-MEME, add -7 option
bwa-meme mem -7 -Y -K 100000000 -t <num_threads> <input.fasta> <input_1.fastq> -o <output_meme.sam>

# Below runs alignment with BWA-MEM2, without -7 option
bwa-meme mem -Y -K 100000000 -t <num_threads> <input.fasta> <input_1.fastq> -o <output_mem2.sam>

# Compare output SAM files
diff <output_mem2.sam> <output_meme.sam>

# To diff large SAM files use https://github.com/unhammer/diff-large-files
```

---
### Install Option 2. Build locally
#### Compile the code
```sh
# Compile from source
git clone https://github.com/kaist-ina/BWA-MEME.git BWA-MEME
cd BWA-MEME

# To compile all binary executables run below command. 
# Put the highest number of available vCPU cores
# You should also have cmake installed. Download by sudo apt-get install cmake
make -j<num_threads>

# Print version and Mode of compiled binary executable
# bwa-meme binary automatically choose the binary based on the SIMD instruction supported (SSE, AVX2, AVX512 ...)
# Other modes of bwa-meme is available as bwa-meme_mode1 or bwa-meme_mode2
./bwa-meme version

# For bwa-meme with mode 1 or 2 see below
```
### Build index of the reference DNA sequence
```sh
# Build index (Takes ~1hr for human genome)
# we recommend using 32 threads
./bwa-meme index -a meme <input.fasta> -t <thread number>
```
### Training P-RMI 
Prerequisites for building locally: To use the train code, please [install Rust](https://rustup.rs/).
```sh
# Run code below to train P-RMI, suffix array is required which is generated in index build code
# takes about 15 minute for human genome with single thread
./build_rmis_dna.sh <input.fasta>
```

### Run alignment and compare SAM output with BWA-MEM2
```sh
# Perform alignment with BWA-MEME, add -7 option
./bwa-meme mem -7 -Y -K 100000000 -t <num_threads> <input.fasta> <input_1.fastq> -o <output_meme.sam>

# Below runs alignment with BWA-MEM2, without -7 option
./bwa-meme mem -Y -K 100000000 -t <num_threads> <input.fasta> <input_1.fastq> -o <output_mem2.sam>

# Compare output SAM files
diff <output_mem2.sam> <output_meme.sam>

# To diff large SAM files use https://github.com/unhammer/diff-large-files
```

### Test scripts and executables are available in the BWA-MEME/test folder
---
## Changing memory requirement for index in BWA-MEME 
```sh
# You can check the MODE value by running version command
# mode 1: 38GB in index size
./bwa-meme_mode1 version
# mode 2: 88GB in index size
./bwa-meme_mode2 version
# mode 3: 118GB in index size, fastest mode
./bwa-meme  version

# If binary executable does not exist, run below command to compile
make clean
make -j<number of threads>

```
---
## Notes

* BWA-MEME requires at least 64 GB RAM (with minimal acceleration BWA-MEME requires 38GB of memory). For WGS runs on human genome (>32 threads) with full acceleration of BWA-MEME, it is recommended to have 140-192 GB RAM.

* When deploying BWA-MEME with many threads, mimalloc library is recommended for a better performance (Enabled at default).

### Building pipeline with Samtools
Credits to @keiranmraine, see issue [#10](../../issues/10)

- Due to increased alignment throughput, given enough threads the bottleneck moves from `alignment` to `Samtools sorting`. As a result BWA-MEME might require additional pipeline modification (not a simple drop-in replacement)
- To reduce the CPU waste, you might want to use `mbuffer` in the pipeline or write alignment outputs to a file with fast compression. 
```
# mbuffer size should be determined by memory option given to samtools.
# ex) samtools sort uses 20 threads, 1G per each thread, so mbuffer size should be 20G (= -m 1G x -@ 20)
bwa-meme mem -7 -K 100000000 -t 32 \
 <reference> <fastq 1> <fastq 2> \
 | mbuffer -m 20G \
 | samtools sort -m 1G --output-fmt bam,level=1 -T ./sorttmp -@ 20 - > sorted.bam
```

### Reference file download
You can download the reference using the command below.
```sh
# Download human_g1k_v37.fasta human genome and decompress it
wget -c ftp://ftp-trace.ncbi.nih.gov/1000genomes/ftp/technical/reference/human_g1k_v37.fasta.gz
gunzip human_g1k_v37.fasta.gz

# hg38 human reference
wget -c https://storage.googleapis.com/genomics-public-data/references/hg38/v0/Homo_sapiens_assembly38.fasta
```

### Download MEME indices and pretrained P-RMI model
```sh
# We provide the pretrained models and all indices required alignment (for hg37 and hg38 human reference) 
# you can download in the link below.
https://web.inalab.net/~bwa-meme/

# Indices of MEME and models should be in the same folder, we follow the prefix-based loading in bwa-mem
```

## Citation

If you use BWA-MEME, please cite the following [paper](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btac137/6543607)
> **Youngmok Jung, Dongsu Han, BWA-MEME: BWA-MEM emulated with a machine learning approach, Bioinformatics, Volume 38, Issue 9, 1 May 2022, Pages 2404–2413, https://doi.org/10.1093/bioinformatics/btac137**


```
@article{10.1093/bioinformatics/btac137,
    author = {Jung, Youngmok and Han, Dongsu},
    title = "{BWA-MEME: BWA-MEM emulated with a machine learning approach}",
    journal = {Bioinformatics},
    volume = {38},
    number = {9},
    pages = {2404-2413},
    year = {2022},
    month = {03},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btac137},
    url = {https://doi.org/10.1093/bioinformatics/btac137},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/38/9/2404/43480985/btac137.pdf},
}

```

<!-- ## Todo

* Support BAM output

* Support Sorting

* Support Markduplicate -->
