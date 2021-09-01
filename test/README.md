### Testing BWA-MEME
To evalute BWA-MEME, BWA-MEM2, and ERT.
We provide sample run script.
### Compiling test code and running test scripts
```sh
# Compile the test code
make -j<num_threads> learned_seeding_big_read  ert_seeding_big_read  fmi_seeding_big_read

# Run test script
bash ./script_fmi.sh
bash ./script_ert.sh
bash ./script_meme.sh


```
## Test dataset
We used the publicly available short read dataset from Illumina platinum genome and 1000 Genomes Project-Phase3.
Each can be downloaded from

https://www.ebi.ac.uk/ena/browser/view/PRJEB3381

https://www.ncbi.nlm.nih.gov/bioproject/527456

## Seeding throughput and alignment throughput measurement
To measure the seeding throughput and alignment throughput, please see the run_seeding.sh and run_end_to_end.sh script


## Analyzing memory characteristics
To analyze the memory characteristics, we use Intel Vtunes profiler and the api provided.
1. [Install oneAPI base Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/all-toolkits.html#base-kit)
2. Fix the Vtune library path in the Makefile (/opt/intel/oneapi/vtune/2021.5.0/ to your path)
3. Compile & Run test script using the Intel Vtunes Profiler.


## Notes

* To force all memory allocation into single socket we use numactl.



