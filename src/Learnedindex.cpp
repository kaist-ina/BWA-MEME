/*
 * The MIT License
 *
 * Copyright (C) 2021  Youngmok Jung, Dongsu Han.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * Authors: Youngmok Jung <tom418@kaist.ac.kr>; Dongsu Han <dhan.ee@kaist.ac.kr>;
 */
#include <stdlib.h>
#include <stdio.h>
#include "sais.h"
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <limits.h>
#include <pthread.h>
#include <errno.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <emmintrin.h>
#include "utils.h"
#include "Learnedindex.h"
#include "LearnedIndex_seeding.h"
#include "memcpy_bwamem.h"
#include <vector>
#ifdef __cplusplus
extern "C" {
#endif
#include "safe_mem_lib.h"
#include "safe_str_lib.h"
#include <snprintf_s.h>
#ifdef __cplusplus
}
#endif

#define assert_not_null(x, size, cur_alloc) \
        if (x == NULL) { fprintf(stderr, "Allocation of %0.2lf GB for " #x " failed.\nCurrent Allocation = %0.2lf GB\n", size * 1.0 /(1024*1024*1024), cur_alloc * 1.0 /(1024*1024*1024)); exit(EXIT_FAILURE); }


int64_t pac_seq_len_(const char *fn_pac)
{
	FILE *fp;
	int64_t pac_len;
	uint8_t c;
	fp = xopen(fn_pac, "rb");
	err_fseek(fp, -1, SEEK_END);
	pac_len = err_ftell(fp);
	err_fread_noeof(&c, 1, 1, fp);
	err_fclose(fp);
	assert(c >= 0 && c <= 255);
	return (pac_len - 1) * 4 + (int)c;
}

void pac2nt_(const char *fn_pac, std::string &reference_seq)
{
	uint8_t *buf2;
	int64_t i, pac_size, seq_len;
	FILE *fp;

	// initialization
	seq_len = pac_seq_len_(fn_pac);
	assert(seq_len > 0);
	assert(seq_len <= 0x7fffffffffffffffL);
	fp = xopen(fn_pac, "rb");

	// prepare sequence
	pac_size = (seq_len>>2) + ((seq_len&3) == 0? 0 : 1);
	buf2 = (uint8_t*)calloc(pac_size, 1);
	assert(buf2 != NULL);
	err_fread_noeof(buf2, 1, pac_size, fp);
	err_fclose(fp);
	for (i = 0; i < seq_len; ++i) {
		int nt = buf2[i>>2] >> ((3 - (i&3)) << 1) & 3;
        switch(nt)
        {
            case 0:
                reference_seq += "A";
            break;
            case 1:
                reference_seq += "C";
            break;
            case 2:
                reference_seq += "G";
            break;
            case 3:
                reference_seq += "T";
            break;
            default:
                fprintf(stderr, "ERROR! Value of nt is not in 0,1,2,3!");
                exit(EXIT_FAILURE);
        }
	}
    for(i = seq_len - 1; i >= 0; i--)
    {
        char c = reference_seq[i];
        switch(c)
        {
            case 'A':
                reference_seq += "T";
            break;
            case 'C':
                reference_seq += "G";
            break;
            case 'G':
                reference_seq += "C";
            break;
            case 'T':
                reference_seq += "A";
            break;
        }
    }
	free(buf2);
}


void buildSAandLEP(char* prefix, int num_threads){

    uint64_t startTick;
    int status;
    int index_alloc  = 0;
    std::string reference_seq;
    char pac_file_name[PATH_MAX];
    strcpy_s(pac_file_name, PATH_MAX, prefix);
    strcat_s(pac_file_name, PATH_MAX, ".pac"); 
    printf("[Build-LearnedIndexmode] pac2nt function...\n");
    pac2nt_(pac_file_name, reference_seq);
    
    
    
    char binary_ref_name[PATH_MAX];
    strcpy_s(binary_ref_name, PATH_MAX, prefix);
    strcat_s(binary_ref_name, PATH_MAX, ".0123");
    std::fstream binary_ref_stream (binary_ref_name, std::ios::out | std::ios::binary);
    binary_ref_stream.seekg(0);	
    int64_t i;
    int64_t pac_len = reference_seq.length();
    int64_t max_a_len = 0, max_t_len = 0, tmp_a_len = 0, tmp_t_len = 0;

    for(i = 0; i < pac_len; i++)
    {
        switch(reference_seq[i])
        {
            case 'A':
                tmp_a_len += 1;
                tmp_t_len = 0;
                break;
            case 'C':
                tmp_a_len = 0;
                tmp_t_len = 0;
                break;
            case 'G':
                tmp_a_len = 0;
                tmp_t_len = 0;
                break;
            case 'T':
                tmp_a_len = 0;
                tmp_t_len += 1;
                break;
            default:
                assert(0);
        }
        if (tmp_a_len > max_a_len){
            max_a_len = tmp_a_len;
        }
        if (tmp_t_len > max_t_len){
            max_t_len = tmp_t_len;
        }
    }

    int64_t padded_t_len = std::max(max_a_len+1,max_t_len+1);
    
    int64_t size = (reference_seq.length() + padded_t_len) * sizeof(char);
    char *binary_ref_seq = (char *)_mm_malloc(size , 64);
    index_alloc += size;
    assert_not_null(binary_ref_seq, size, index_alloc);

    

    for(i = 0; i < pac_len; i++)
    {
        switch(reference_seq[i])
        {
            case 'A':
                binary_ref_seq[i] = 0;
                break;
            case 'C':
                binary_ref_seq[i] = 1;
                break;
            case 'G':
                binary_ref_seq[i] = 2;
                break;
            case 'T':
                binary_ref_seq[i] = 3;
                break;
            default:
                assert(0);
                binary_ref_seq[i] = 4;
        }
    }
    
    printf("[Build-LearnedIndexmode] ref seq len = %ld\n", pac_len);
    binary_ref_stream.write(binary_ref_seq, pac_len * sizeof(char));
    binary_ref_stream.close();
    assert(pac_len%2 ==0 );
    pac_len = pac_len + padded_t_len;
    for (i;i<pac_len;i++){
        binary_ref_seq[i] = 3;  
        reference_seq += "T";
    }
    printf("[Build-LearnedIndexmode] padded ref len = %ld\n", pac_len);
    //build suffix array
    size = (pac_len + 2) * sizeof(int64_t);
    int64_t *suffix_array=(int64_t *)_mm_malloc(size, 64);
    index_alloc += size;
    assert_not_null(suffix_array, size, index_alloc);
    startTick = __rdtsc();


    printf("[Build-LearnedIndexmode] Building Suffix array\n");
    uint64_t query_k_mer = 32; // fix this to 32, use front 32 character for Learned Index model inference
    
    status = saisxx(reference_seq.c_str(), suffix_array, pac_len);

    // [    Ref      ][   Ref_complement   ][TTTTTTTTTT...T]
    
    uint64_t total_sa_num = pac_len - padded_t_len;//
    
    char filename[PATH_MAX];
    strcpy_s(filename, PATH_MAX, prefix);
    if (query_k_mer <= 32){
        strcat_s(filename, PATH_MAX, ".suffixarray_uint64");
    }
    else if (query_k_mer <= 64){
        strcat_s(filename, PATH_MAX, ".suffixarray_uint128");
    }
    
    std::ofstream sa_out(filename, std::ios_base::trunc | std::ios::binary);

    if (!sa_out.is_open()) {
      std::cerr << "unable to open " << filename << std::endl;
      exit(EXIT_FAILURE);
    }
    sa_out.write(reinterpret_cast<const char*>(&total_sa_num), sizeof(uint64_t));

    char pos_filename[PATH_MAX];
    strcpy_s(pos_filename, PATH_MAX, prefix);

    strcat_s(pos_filename, PATH_MAX, ".possa_packed");
    std::ofstream possa_out(pos_filename, std::ios_base::trunc | std::ios::binary);

    if (!possa_out.is_open()) {
      std::cerr << "unable to open " << pos_filename << std::endl;
      exit(EXIT_FAILURE);
    }

    // open pos_packed
    strcpy_s(pos_filename, PATH_MAX, prefix);
    strcat_s(pos_filename, PATH_MAX, ".pos_packed");
    std::ofstream pos_out(pos_filename, std::ios_base::trunc | std::ios::binary);

    if (!pos_out.is_open()) {
      std::cerr << "unable to open " << pos_filename << std::endl;
      exit(EXIT_FAILURE);
    }
// #if MEM_TRADEOFF && READ_FROM_FILE
    // Design3: ISA
    uint8_t *ref_to_sapos = (uint8_t *)_mm_malloc(5*total_sa_num*sizeof(uint8_t) , 64);
// #endif

    uint8_t c;
    bwtintv_t ik, ok[4];
    uint32_t prevHits;
    uint64_t r;
    uint64_t sa_count=0;
    printf("[Build-LearnedIndexmode] Build and Save SA, Pos\n");
    for (i=0 ; i< pac_len; i++){
        
        if ( pac_len - padded_t_len <= suffix_array[i] &&  pac_len > suffix_array[i] ){
            continue;
        }
        
        uint32_t pos_val = suffix_array[i] >>8 ;
        uint8_t ls_val= suffix_array[i] & 0xff;
        pos_out.write(reinterpret_cast<const char*>(&pos_val), sizeof(uint32_t));
        pos_out.write(reinterpret_cast<const char*>(&ls_val), sizeof(uint8_t));
        
        possa_out.write(reinterpret_cast<const char*>(&pos_val), sizeof(uint32_t));
        possa_out.write(reinterpret_cast<const char*>(&ls_val), sizeof(uint8_t));
// #if MEM_TRADEOFF && READ_FROM_FILE

        *(uint32_t*)(ref_to_sapos+ suffix_array[i]*5) = (uint32_t) (sa_count>>8);
        *(ref_to_sapos+ suffix_array[i]*5+4) = (uint8_t)(sa_count & 0xff);
// #endif
        sa_count++;
        assert(suffix_array[i] < pac_len && suffix_array[i]>=0);


        // char c = binary_ref_seq[suffix_array[i]-1];
        uint64_t binary_suffix_array = 0;
        uint64_t reverse_bit = 0;
        for (r =0 ; r < query_k_mer ; r++){
            binary_suffix_array = binary_suffix_array << 2;
            reverse_bit = reverse_bit <<2;
            switch (binary_ref_seq[ (suffix_array[i]+r)%pac_len]){
                case 0:
                    binary_suffix_array = (binary_suffix_array|0);
                    
                    break;
                case 1:
                    binary_suffix_array = (binary_suffix_array|1);
                    break;
                case 2:
                    binary_suffix_array = (binary_suffix_array|2);
                    break;
                case 3:
                    binary_suffix_array = (binary_suffix_array|3);
                    break;

            }
            switch (binary_ref_seq[ (suffix_array[i]+query_k_mer-r-1)%pac_len]){
                case 0:
                    reverse_bit = (reverse_bit|0);
                    break;
                case 1:
                    reverse_bit = (reverse_bit|1);
                    break;
                case 2:
                    reverse_bit = (reverse_bit|2);
                    break;
                case 3:
                    reverse_bit = (reverse_bit|3);
                    break;

            }
        }
        
        
        
        possa_out.write(reinterpret_cast<const char*>(&reverse_bit), sizeof(uint64_t));
        sa_out.write(reinterpret_cast<const char*>(&binary_suffix_array), sizeof(uint64_t));


    }
    //check suffix array number is correct
    assert(sa_count== total_sa_num);
    pos_out.close();
    possa_out.close();
    sa_out.close();

// #if MEM_TRADEOFF && READ_FROM_FILE
    char ref_to_sapos_name[PATH_MAX];
    strcpy_s(ref_to_sapos_name, PATH_MAX, prefix);
    strcat_s(ref_to_sapos_name, PATH_MAX, ".ref2sa_packed");
    std::fstream ref2sa_stream (ref_to_sapos_name, std::ios::out | std::ios::binary);
    ref2sa_stream.seekg(0);	
    ref2sa_stream.write((char*)ref_to_sapos, 5*total_sa_num * sizeof(uint8_t));
    ref2sa_stream.close();
    _mm_free(ref_to_sapos);
// #endif


    // hit_out.close();
    fprintf(stderr, "build suffix-array ticks = %llu\n", __rdtsc() - startTick);
    
    


    _mm_free(binary_ref_seq);
    _mm_free(suffix_array);
    

}
