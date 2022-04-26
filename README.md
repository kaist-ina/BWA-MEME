# BWA-MEME: BWA-MEM emulated with a machine learning approach 
- BWA-MEME produces identical results as BWA-MEM2 and achieves 1.4x higher alignment throughput.
- Seeding throughput of BWA-MEME is up to 3.32x higher than BWA-MEM2.
- BWA-MEME builds upon BWA-MEM2 and includes performance improvements to the seeding. 
- BWA-MEME leverages **learned index** in **suffix array search**.
- BWA-MEME also provides feature to accomodate various memory size in servers.
---

## Contents
* [When to use BWA-MEME](#when-to-use-bwa-meme)
* [Performance of BWA-MEME](#performance-of-bwa-meme)
* [Getting Started](#getting-started)
  + [Option 1. Bioconda](#install-option-1-bioconda)
  + [Option 2. Build locally](#install-option-2-build-locally)
* [Changing memory requirement for index in BWA-MEME](#changing-memory-requirement-for-index-in-bwa-meme)
* [Notes](#notes)
  + [(Optional) Reference file download](#-optional--reference-file-download)
  + [(Optional) Download pre-trained P-RMI learned-index model](#-optional--download-pre-trained-p-rmi-learned-index-model)
* [Citation](#citation)
---
## When to use BWA-MEME
- Anyone who use BWA-MEM or BWA-MEM2 in CPU-only machine (BWA-MEME requires 38GB of memory for index at minimal mode)
- Building high-throughput NGS alignment cluster with low cost/throughput. CPU-only alignment can be cheaper than using hardware acceleration (GPU, FPGA).
- Just add single option "-7" to deploy BWA-MEME instead of BWA-MEM2 (BWA-MEME does not change anything, except the speed).

## Performance of BWA-MEME
#### The seeding module of BWA-MEME uses Learned-index. This, in turn, results in 3.32x higher seeding throughput compared to FM-index of BWA-MEM2.
<img src="https://github.com/kaist-ina/BWA-MEME/blob/dev/images/BWA-MEME-SeedingResults.jpg" width="50%"/>

#### End-to-end alignment throughput is up to 1.4x higher than BWA-MEM2.
<img src="https://github.com/kaist-ina/BWA-MEME/blob/dev/images/BWA-MEME-AlignmentResults.png" width="50%" />

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
- Building Suffix array, Inverse suffix array 
```sh
# Build index (Takes ~4 hr for human genome with 32 threads. 1 hr for BWT, 3 hr for BWA-MEME)
bwa-meme index -a meme -t <num_threads> <input.fasta>
```
### Training P-RMI
Prerequisites: To use the train code, please [install Rust](https://rustup.rs/).
```sh
# Run code below to train P-RMI, suffix array is required which is generated in index build code
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
make -j<num_threads>

# Print version and Mode of compiled binary executable
# bwa-meme binary automatically choose the binary based on the SIMD instruction supported (SSE, AVX2, AVX512 ...)
./bwa-meme version

# For bwa-meme with mode 1 or 2 see below
```
### Build index of the reference DNA sequence
- Building Suffix array, Inverse suffix array 
```sh
# Build index (Takes ~4 hr for human genome with 32 threads. 1 hr for BWT, 3 hr for BWA-MEME)
./bwa-meme index -a meme -t <num_threads> <input.fasta>
```
### Training P-RMI
Prerequisites: To use the train code, please [install Rust](https://rustup.rs/).
```sh
# Run code below to train P-RMI, suffix array is required which is generated in index build code
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

* When deploying BWA-MEME with many threads (>72), mimalloc library is recommended to avoid performance drop issue.

### (Optional) Reference file download
You can download the reference using the command below.
```sh
# Download human_g1k_v37.fasta human genome and decompress it
wget ftp://ftp-trace.ncbi.nih.gov/1000genomes/ftp/technical/reference/human_g1k_v37.fasta.gz
gunzip human_g1k_v37.fasta.gz
```

### (Optional) Download pre-trained P-RMI learned-index model
```sh
# We provide the pretrained models for human_g1k_v37.fasta, please download in the link below.
# Two P-RMI model parameter files are required to run BWA-MEME
https://web.inalab.net/~bwa-meme/
```

## Citation

If you use BWA-MEME, please cite the following [paper](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btac137/6543607)
> **Youngmok Jung, Dongsu Han, BWA-MEME: BWA-MEM emulated with a machine learning approach, Bioinformatics, 2022;, btac137, https://doi.org/10.1093/bioinformatics/btac137**


```
@article{10.1093/bioinformatics/btac137,
    author = {Jung, Youngmok and Han, Dongsu},
    title = "{BWA-MEME: BWA-MEM emulated with a machine learning approach}",
    journal = {Bioinformatics},
    year = {2022},
    month = {03},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btac137},
    url = {https://doi.org/10.1093/bioinformatics/btac137},
    note = {btac137},
    eprint = {https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btac137/42752427/btac137.pdf},
}

```

<!-- ## Todo

* Support BAM output

* Support Sorting

* Support Markduplicate -->
