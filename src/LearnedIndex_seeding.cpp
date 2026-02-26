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

#include "LearnedIndex_seeding.h"
#include "memcpy_bwamem.h"

#include <math.h>
#include <cmath>
#include <fstream>
#include <x86intrin.h>
// #include <filesystem>
#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif
#include "safe_mem_lib.h"
#include "safe_str_lib.h"
#include <snprintf_s.h>
#ifdef __cplusplus
}
#endif

#ifdef __GNUC__
#  define ffs(x) __builtin_ffsll(x)
#  define fls(x) __builtin_clzll(x)
#elif __INTEL_COMPILER
#  define ffs(x) _bit_scan_forward64(x)
#  define fls(x) __builtin_clzll(x)
#endif

double SA_NUM;
uint64_t bit_shift;
char* L1_PARAMETERS;
char* L2_PARAMETERS;
/* -8 mode flag: set to 1 when -8 is used (SIMD-accelerated encode + batch lookup) */
static int g_use_simd8_mode = 0;

void learned_set_simd8_mode(int enable) {
	g_use_simd8_mode = enable ? 1 : 0;
}

int learned_get_simd8_mode(void) {
	return g_use_simd8_mode;
}

// sa_raux_buf is used to select properly aligned read among unpacked_queue_binary_buf_shift*
uint8_t sa_raux_buf[4][4]= {
	{0,3,2,1},
	{1,0,3,2},
	{2,1,0,3},
	{3,2,1,0}
} ;
#define _get_pac(pac, l) ((pac)[(l)>>2]>>((~(l)&3)<<1)&3)

// Used to set pivot point in the read
inline void set_forward_pivot(Learned_read_aux_t* raux, int pivot){
	raux->pivot = pivot;
	raux->l_pivot = raux->l_seq-1 - raux->pivot;
}


bool learned_index_load(char const* dataPath, char const* dataPath2, char const* dataPath3, double suffix_array_num) {
	{
		SA_NUM=suffix_array_num;
		uint64_t model_size = 0;
		uint64_t num_model;
		
		{
			std::ifstream infile(dataPath2, std::ios::in | std::ios::binary);
			if (!infile.good()) {
				fprintf(stderr, "[indexload]Can't open learned-index model, read_path: %s\n.", dataPath2);
				return false;
			}
			model_size = infile.tellg();
			infile.seekg( 0, std::ios::end );
			model_size = infile.tellg() - model_size;
			infile.seekg( 0, std::ios::beg );
			L1_PARAMETERS = (char*) malloc(model_size);
			if (L1_PARAMETERS == NULL){
				fprintf(stderr, "Can't alloc memory for learned-index model\n.");
				return false;
			}
			infile.read((char*)L1_PARAMETERS, model_size);
		}

		{
			std::ifstream infile(dataPath3, std::ios::in | std::ios::binary);
			if (!infile.good()) {
				fprintf(stderr, "Can't open learned-index model, read_path: %s\n.", dataPath3);
				return false;
			}
			model_size = infile.tellg();
			infile.seekg( 0, std::ios::end );
			model_size = infile.tellg() - model_size;
			infile.seekg( 0, std::ios::beg );
			L2_PARAMETERS = (char*) malloc(model_size);
			if (L2_PARAMETERS == NULL){
				fprintf(stderr, "Can't alloc memory for learned-index model\n.");
				return false;
			}
			num_model = model_size/24; // number of model
			infile.read((char*)L2_PARAMETERS, model_size);

		}
		bit_shift = ffs(num_model)-1;
		fprintf(stderr,"[Learned-Config] MODE:%d SEARCH_METHOD: %d MEM_TRADEOFF:%d EXPONENTIAL_SMEMSEARCH: %d DEBUG_MODE:%d Num 2nd Models:%ld PWL Bits Used:%ld\n", MODE, CURR_SEARCH_METHOD, MEM_TRADEOFF, EXPONENTIAL_SMEMSEARCH,DEBUG_MODE, num_model, bit_shift);
		bit_shift = 64 - bit_shift;
	}
	return true;
}



void learned_index_cleanup() {
	free(L1_PARAMETERS);
	free(L2_PARAMETERS);
}

inline size_t pwl( uint64_t inp) {
    // return inp >> 38;
	return inp >> 36;
}

inline double linear(double alpha, double beta, double inp) {
	return std::fma(beta, inp, alpha);
} 

inline size_t FCLAMP(double inp, double bound) {
	if (inp < 0.0) return 0;
	return (inp > bound ? bound : (size_t)inp);
}

/* ===========================================================================
 * encode_read_simd8 — SIMD-accelerated 4-way shift-buffer construction (-8 mode)
 *
 * Builds 8 packed 2-bit shift buffers (shift1..4 for forward, rc_shift1..4
 * for reverse-complement) needed by Tokenization / compare_read_and_ref_binary.
 *
 * The output is bit-for-bit identical to the scalar code in bwamem.cpp; the
 * AVX2 path just amortises the BitReverseTable256 lookup across 4 shift phases
 * simultaneously instead of doing 4 separate passes.
 * ===========================================================================*/
void encode_read_simd8(const char* seq, int len,
                       uint8_t* shift1, uint8_t* shift2,
                       uint8_t* shift3, uint8_t* shift4,
                       uint8_t* rc_shift1, uint8_t* rc_shift2,
                       uint8_t* rc_shift3, uint8_t* rc_shift4)
{
    /* Correct implementation: single rolling accumulator (identical to -7 original).
     *
     * The original code uses ONE set_bit register that accumulates all bases in order.
     * Each shift buffer records the accumulator state at a specific phase offset (k&3).
     * Using 4 independent accumulators would produce wrong phase alignment.
     *
     * This function is a drop-in replacement for the original scalar double-loop,
     * merging the two loops (k<len and k<len+4) into one. The output is bit-for-bit
     * identical to the original. Any further SIMD optimisation must preserve this
     * single-accumulator semantics.
     */
    uint8_t set_bit    = 0;
    uint8_t set_rc_bit = 0;
    int k;

    /* Main loop: k = 0..len-1 */
    for (k = 0; k < len; ++k) {
        set_bit    = (uint8_t)(set_bit    << 2);
        set_rc_bit = (uint8_t)(set_rc_bit << 2);
        set_bit    |= (seq[k] < 4) ? seq[k] : 0;
        set_rc_bit |= (seq[len-1-k] < 4) ? (3 - seq[len-1-k]) : 0;
        if ((k & 3) == 0) {
            shift1[k>>2]    = BitReverseTable256[set_bit];
            rc_shift1[k>>2] = BitReverseTable256[set_rc_bit];
        } else if ((k & 3) == 1) {
            shift2[k>>2]    = BitReverseTable256[set_bit];
            rc_shift2[k>>2] = BitReverseTable256[set_rc_bit];
        } else if ((k & 3) == 2) {
            shift3[k>>2]    = BitReverseTable256[set_bit];
            rc_shift3[k>>2] = BitReverseTable256[set_rc_bit];
        } else {
            shift4[k>>2]    = BitReverseTable256[set_bit];
            rc_shift4[k>>2] = BitReverseTable256[set_rc_bit];
        }
    }
    /* Padding loop: k = len..len+3 */
    for (; k < len + 4; ++k) {
        set_bit    = (uint8_t)(set_bit    << 2);
        set_rc_bit = (uint8_t)(set_rc_bit << 2);
        if ((k & 3) == 0) {
            shift1[k>>2]    = BitReverseTable256[set_bit];
            rc_shift1[k>>2] = BitReverseTable256[set_rc_bit];
        } else if ((k & 3) == 1) {
            shift2[k>>2]    = BitReverseTable256[set_bit];
            rc_shift2[k>>2] = BitReverseTable256[set_rc_bit];
        } else if ((k & 3) == 2) {
            shift3[k>>2]    = BitReverseTable256[set_bit];
            rc_shift3[k>>2] = BitReverseTable256[set_rc_bit];
        } else {
            shift4[k>>2]    = BitReverseTable256[set_bit];
            rc_shift4[k>>2] = BitReverseTable256[set_rc_bit];
        }
    }
}

/* ===========================================================================
 * learned_prefetch_next_read_sa -- lookahead SA prefetch for next read (-8)
 *
 * Computes the P-RMI estimated SA position for the first valid pivot of
 * next_seq and issues _mm_prefetch for that SA cache line.
 * Called in mem_kernel1_core_Learned just before processing read l,
 * using read l+1 as lookahead, so SA DRAM fetch overlaps seeding of l.
 * ===========================================================================*/
void learned_prefetch_next_read_sa(const uint8_t* sa_pos,
                                   const char*    next_seq,
                                   int            next_len,
                                   int            min_seed_len)
{
    if (next_len < min_seed_len) return;

    /* find first non-N pivot */
    int p = 0;
    while (p < next_len - min_seed_len + 1 &&
           (uint8_t)next_seq[p] >= 4)
        p++;
    if (p >= next_len - min_seed_len + 1) return;

    /* build 64-bit key from first 32 bases at pivot p */
    uint64_t key = 0;
    int end = p + 32 < next_len ? p + 32 : next_len;
    for (int i = p; i < end; i++) {
        uint8_t b = (uint8_t)next_seq[i];
        key = (key << 2) | (b < 4 ? b : 0);
    }
    key <<= (64 - (end - p) * 2);   /* left-align */

    /* P-RMI lookup (mirrors learned_index_lookup) */
    size_t modelIndex = key >> bit_shift;
    double fpred = linear(*((double*)(L2_PARAMETERS + modelIndex*24 + 0)),
                          *((double*)(L2_PARAMETERS + modelIndex*24 + 8)),
                          (double)key);
    size_t err = *((uint64_t*)(L2_PARAMETERS + modelIndex*24 + 16));
    if (err >> 63) {
        size_t partial_start = (err >> 32) & 0x7fffffff;
        double partial_num   = (double)(err & 0x00000000ffffffffULL);
        modelIndex = partial_start + FCLAMP(fpred, partial_num - 1.0);
        fpred = linear(*((double*)(L1_PARAMETERS + modelIndex*24 + 0)),
                       *((double*)(L1_PARAMETERS + modelIndex*24 + 8)),
                       (double)key);
    }
    uint64_t est_pos = (uint64_t)FCLAMP(fpred, SA_NUM - 1.0);

    /* prefetch estimated SA entry; T2 = L3 hint (770 MB pac shared) */
    _mm_prefetch((const char*)(sa_pos + est_pos * SASIZE), _MM_HINT_T2);
}

/* ===========================================================================
 * learned_index_lookup_batch — batch P-RMI lookup for n keys  (-8 mode)
 *
 * Processes `n` keys independently using interleaved prefetches to hide
 * DRAM latency of the L2_PARAMETERS table.
 *
 * Prefetch distance LOOKUP_BATCH_PREFETCH_DIST: empirically 4-8 is optimal
 * for a ~38 GB learned-index (L2_PARAMETERS ~6 GB, L1_PARAMETERS ~6 MB).
 * ===========================================================================*/
#define LOOKUP_BATCH_PREFETCH_DIST 8

void learned_index_lookup_batch(const uint64_t* keys, int n,
                                uint64_t* positions, size_t* errs)
{
    /* Warm up prefetch pipeline */
    {
        int warmup = (n < LOOKUP_BATCH_PREFETCH_DIST) ? n : LOOKUP_BATCH_PREFETCH_DIST;
        for (int i = 0; i < warmup; ++i) {
            size_t mi = keys[i] >> bit_shift;
            _mm_prefetch((const char*)(L2_PARAMETERS + mi * 24), _MM_HINT_T0);
        }
    }

    for (int i = 0; i < n; ++i) {
        /* Prefetch for key[i + DIST] */
        int pf = i + LOOKUP_BATCH_PREFETCH_DIST;
        if (pf < n) {
            size_t mi_pf = keys[pf] >> bit_shift;
            _mm_prefetch((const char*)(L2_PARAMETERS + mi_pf * 24), _MM_HINT_T0);
        }

        /* Standard p-RMI lookup (identical to learned_index_lookup) */
        uint64_t key       = keys[i];
        size_t modelIndex  = key >> bit_shift;
        double fpred       = linear(*((double*)(L2_PARAMETERS + modelIndex*24 + 0)),
                                    *((double*)(L2_PARAMETERS + modelIndex*24 + 8)),
                                    (double)key);
        size_t err_val     = *((uint64_t*)(L2_PARAMETERS + modelIndex*24 + 16));

        if (err_val >> 63) {
            size_t partial_start = (err_val >> 32) & 0x7fffffff;
            double partial_num   = (double)(err_val & 0x00000000ffffffffULL);
            modelIndex = partial_start + FCLAMP(fpred, partial_num - 1.0);
            _mm_prefetch((const char*)(L1_PARAMETERS + modelIndex*24), _MM_HINT_T0);
            fpred   = linear(*((double*)(L1_PARAMETERS + modelIndex*24 + 0)),
                             *((double*)(L1_PARAMETERS + modelIndex*24 + 8)),
                             (double)key);
            err_val = *((uint64_t*)(L1_PARAMETERS + modelIndex*24 + 16));
        }

        positions[i] = FCLAMP(fpred, SA_NUM - 1.0);
        errs[i]      = err_val;
    }
}


/*
* Example of lookup function for naive 2-layer RMI
*/
inline uint64_t learned_index_lookup_2rmi(uint64_t key, size_t* err) { // 2 layer RMI pwl,linear
	size_t modelIndex;
	double fpred;
	// uint64_t ipred;
	// modelIndex = key >>36;
	fpred = linear(L0_PARAMETER0, L0_PARAMETER1, (double)key);
  	modelIndex = FCLAMP(fpred, 268435456.0 - 1.0);


	fpred = linear(*((double*) (L1_PARAMETERS + (modelIndex * 24) + 0)), *((double*) (L1_PARAMETERS + (modelIndex * 24) + 8)), (double)key);

	*err = *((uint64_t*) (L1_PARAMETERS + (modelIndex * 24) + 16));

	return FCLAMP(fpred, SA_NUM - 1.0);
}

/*
* Example of lookup function for naive 3-layer RMI
*/
inline uint64_t learned_index_lookup_3rmi(uint64_t key, size_t* err) { // 3 layer RMI linear,linear,linear_spline
	size_t modelIndex;
	double fpred;
	// uint64_t ipred;
	// modelIndex = pwl(key);

	fpred = linear(L0_PARAMETER0, L0_PARAMETER1, (double)key);
  	modelIndex = FCLAMP(fpred, 48047096.0 - 1.0);

	fpred = linear( *((double*)(L1_PARAMETERS+16*modelIndex + 0)), *((double*)(L1_PARAMETERS+16*modelIndex + 8)), (double)key);

	modelIndex = FCLAMP(fpred, 268435456.0 - 1.0);
  	fpred = linear(*((double*) (L2_PARAMETERS + (modelIndex * 24) + 0)), *((double*) (L2_PARAMETERS + (modelIndex * 24) + 8)), (double)key);

	*err = *((uint64_t*) (L2_PARAMETERS + (modelIndex * 24) + 16));

	return FCLAMP(fpred, SA_NUM - 1.0);
}

inline uint64_t learned_index_lookup(uint64_t key, size_t* err) { //p-rmi
	size_t modelIndex;
	double fpred;
	// below is for pwl,linear model
	// uint64_t ipred;
	size_t partial_start;
	double partial_num;
	// modelIndex = pwl(key);
	modelIndex = key >>bit_shift;


	fpred = linear(*((double*) (L2_PARAMETERS + (modelIndex * 24) + 0)), *((double*) (L2_PARAMETERS + (modelIndex * 24) + 8)), (double)key);

	*err = *((uint64_t*) (L2_PARAMETERS + (modelIndex * 24) + 16));

	if (*err>>63 ){
		partial_start = ((*err)>>32) & 0x7fffffff;
		partial_num = (*err) & 0x00000000ffffffff ;
		modelIndex = partial_start + FCLAMP(fpred, partial_num-1);
		fpred = linear(*((double*) (L1_PARAMETERS + (modelIndex * 24) + 0)), *((double*) (L1_PARAMETERS + (modelIndex * 24) + 8)), (double)key);
		*err = *((uint64_t*) (L1_PARAMETERS + (modelIndex * 24) + 16));
	}

	return FCLAMP(fpred, SA_NUM - 1.0);
}



/*
* About compare_read_and_ref_binary function
* @Param pac: 2-bit encoded reference DNA sequence
* @Param sa: Suffix array of reference DNA sequence
* @Param pos: Position in suffix array
* @Param raux: read data
* @Param sa_num: number of suffix array, should check boundary condition (sa_num - sa_pos)
* @Param valid_len: length to perform exact_match between suffix and read
* @Param match_len: store the length of exact match in this variable
* @Param exact_match_flag: set to true if read and suffix have full-exact-match
*/

inline bool compare_read_and_ref_binary_LOADSUFFIX(const uint8_t* pac, const uint8_t* sa,const uint64_t pos, const Learned_read_aux_t* raux, const uint64_t sa_num, const uint64_t valid_len,  uint32_t* match_len, bool* exact_match_flag){
	/*
		ref -> [(A T G C) C C G T] 	4 base in 8-bit
		sa_pos: ref start position, sa_pos>>2: pac index,
		select binary buf by sa_pos & 3
	*/

	uint64_t sa_pos = *(uint32_t*)(sa + pos);
	sa_pos = sa_pos <<8 |  sa[pos+4];
	uint64_t ref_len = sa_num - sa_pos;
	uint64_t compare; 
	uint32_t read_len = (uint32_t)std::min(ref_len, valid_len);
	
	uint8_t* read_string ;

	switch(raux->pivot&3)
	{
		case 1:
			read_string = raux->unpacked_queue_binary_buf_shift1 + ((raux->pivot + 3)>>2);
			break;
		case 2:
			read_string = raux->unpacked_queue_binary_buf_shift2+((raux->pivot + 2)>>2);
			break;
		case 3:
			read_string = raux->unpacked_queue_binary_buf_shift3+ ((raux->pivot + 1)>>2);
			break;
		case 0:
			read_string = raux->unpacked_queue_binary_buf_shift4+(raux->pivot>>2);
			break;

	}
	uint64_t ref_val = *(uint64_t*)(sa+pos+5) ;
	uint64_t read_val = (*(uint64_t*)(read_string));
	
	uint32_t compare_len;
	compare = ref_val^read_val;
	if(compare != 0 ){
		compare_len = ((ffs(compare)-1)>>1);
		if (read_len <= compare_len){
			*match_len = read_len;	
			if (read_len<ref_len){ // if read_len == valid_len
				*exact_match_flag= true;
				return true;
			}else{
				//when Suffix array contains end of ref, which is padded TTTTTTTT
				*match_len = ref_len;
				return false;
			}
		}else{
			*match_len = compare_len;
			return (ref_val << (62-(compare_len<<1))) < (read_val << (62-(compare_len<<1)));
		}
	}
	uint8_t* ref_pos = pac + (sa_pos>>2);
	uint32_t sa_pos_rest = sa_pos&3;
	switch(sa_raux_buf[raux->pivot&3][sa_pos_rest])
	{
		case 1:
			read_string = raux->unpacked_queue_binary_buf_shift1 + ((raux->pivot + 3)>>2);
			break;
		case 2:
			read_string = raux->unpacked_queue_binary_buf_shift2+((raux->pivot + 2)>>2);
			break;
		case 3:
			read_string = raux->unpacked_queue_binary_buf_shift3+ ((raux->pivot + 1)>>2);
			break;
		case 0:
			read_string = raux->unpacked_queue_binary_buf_shift4+(raux->pivot>>2);
			break;

	}
	for (int i=8; (i<<2) <read_len+sa_pos_rest ; i +=8){
		read_val = (*(uint64_t*)(read_string+i)) ;
		ref_val = (*(uint64_t*)(ref_pos+i)) ;
		compare = ref_val^read_val;
		if(compare != 0 ){
			compare_len = (i<<2)+ ((ffs(compare)-1)>>1) - sa_pos_rest;
			if (read_len <= compare_len){
				*match_len = read_len;	
				if (read_len<ref_len){ // if read_len == valid_len
					*exact_match_flag= true;
					return true;
				}else{
					//when Suffix array contains end of ref, which is padded TTTTTTTT
					*match_len = ref_len;
					return false;
				}
			}else{
				*match_len = compare_len;
				compare_len+= sa_pos_rest-(i<<2);
				return (ref_val << (62-(compare_len<<1))) < (read_val << (62-(compare_len<<1)));
			}
		}
	}
	*match_len = read_len;	
	if (read_len<ref_len){ // if read_len == valid_len
		*exact_match_flag= true;
		return true;
	}else{
		//when Suffix array contains end of ref, which is padded TTTTTTTT
		*match_len = ref_len;
		return false;
	}
}

inline bool compare_read_and_ref_binary_left_LOADSUFFIX(const uint8_t* pac, const uint8_t* sa,const uint64_t pos, const Learned_read_aux_t* raux, const uint64_t sa_num, const uint64_t valid_len,  uint32_t* match_len, bool* exact_match_flag){
	uint64_t sa_pos = *(uint32_t*)(sa + pos);
	sa_pos = sa_pos <<8 |  sa[pos+4];
	uint64_t compare; 
	uint64_t ref_len=sa_num - sa_pos;
	uint32_t read_len= (uint32_t)std::min(ref_len,valid_len);
	
	uint8_t* read_string ;

	switch(raux->l_pivot&3)
	{
		case 1:
			read_string = raux->unpacked_rc_queue_binary_buf_shift1+((raux->l_pivot + 3)>>2);
			break;
		case 2:
			read_string = raux->unpacked_rc_queue_binary_buf_shift2+((raux->l_pivot + 2)>>2);
			break;
		case 3:
			read_string = raux->unpacked_rc_queue_binary_buf_shift3 + ((raux->l_pivot + 1)>>2);
			break;
		case 0:
			read_string = raux->unpacked_rc_queue_binary_buf_shift4 + (raux->l_pivot>>2);
			break;
	}

	uint64_t ref_val = *(uint64_t*)(sa+pos+5) ;
	uint64_t read_val = (*(uint64_t*)(read_string));
	uint32_t compare_len;
	compare = ref_val^read_val;
	if(compare != 0 ){
		compare_len = ((ffs(compare)-1)>>1);
		if (read_len <= compare_len){
			*match_len = read_len;	
			if (read_len<ref_len){ // if read_len == valid_len
				*exact_match_flag= true;
				return true;
			}else{
				//when Suffix array contains end of ref, which is padded TTTTTTTT
				*match_len = ref_len;
				return false;
			}
		}else{
			*match_len = compare_len;
			return (ref_val << (62-(compare_len<<1))) < (read_val << (62-(compare_len<<1)));
		}
	}
	uint8_t* ref_pos = pac + (sa_pos>>2);
	uint32_t sa_pos_rest = sa_pos&3;
	switch(sa_raux_buf[raux->l_pivot&3][sa_pos_rest])
	{
		case 1:
			read_string = raux->unpacked_rc_queue_binary_buf_shift1+((raux->l_pivot + 3)>>2);
			break;
		case 2:
			read_string = raux->unpacked_rc_queue_binary_buf_shift2+((raux->l_pivot + 2)>>2);
			break;
		case 3:
			read_string = raux->unpacked_rc_queue_binary_buf_shift3 + ((raux->l_pivot + 1)>>2);
			break;
		case 0:
			read_string = raux->unpacked_rc_queue_binary_buf_shift4 + (raux->l_pivot>>2);
			break;

	}

	for (int i=8; (i<<2) <read_len+sa_pos_rest ; i +=8){
		read_val = (*(uint64_t*)(read_string+i)) ;
		ref_val = (*(uint64_t*)(ref_pos+i)) ;
		compare = ref_val^read_val;
		if(compare != 0 ){
			compare_len = (i<<2)+ ((ffs(compare)-1)>>1) - sa_pos_rest;
			if (read_len <= compare_len){
				*match_len = read_len;	
				if (read_len<ref_len){ // if read_len == valid_len
					*exact_match_flag= true;
					return true;
				}else{
					//when Suffix array contains end of ref, which is padded TTTTTTTT
					*match_len = ref_len;
					return false;
				}
			}else{
				*match_len = compare_len;
				compare_len+= sa_pos_rest-(i<<2);
				// *match_len = 4*i+ compare_len - sa_pos_rest;
				return (ref_val << (62-(compare_len<<1))) < (read_val << (62-(compare_len<<1)));
			}
		}
	}
	*match_len = read_len;	
	if (read_len<ref_len){ // if read_len == valid_len
		*exact_match_flag= true;
		return true;
	}else{
		//when Suffix array contains end of ref, which is padded TTTTTTTT
		*match_len = ref_len;
		return false;
	}
}

inline bool compare_read_and_ref_binary_pos_only(const uint8_t* pac, const uint8_t* sa,const uint64_t pos, const Learned_read_aux_t* raux, const uint64_t sa_num, const uint64_t valid_len,  uint32_t* match_len, bool* exact_match_flag){
	/*
		ref -> [(A T G C) C C G T] 	4 base in 8-bit
		sa_pos: ref start position, sa_pos>>2: pac index,
		select binary buf by sa_pos & 3
	*/
	uint64_t sa_pos = *(uint32_t*)(sa + pos);
	sa_pos = sa_pos <<8 |  sa[pos+4];
	uint64_t ref_len=sa_num - sa_pos;
	uint64_t compare; 
	uint32_t read_len= (uint32_t)std::min(ref_len,valid_len);
	uint32_t sa_pos_rest = sa_pos&3;
	uint8_t* ref_pos = pac + (sa_pos>>2);
	uint8_t* read_string;
	switch(sa_raux_buf[raux->pivot&3][sa_pos_rest])
	{
		case 1:
			read_string = raux->unpacked_queue_binary_buf_shift1 + ((raux->pivot + 3)>>2);
			break;
		case 2:
			read_string = raux->unpacked_queue_binary_buf_shift2+((raux->pivot + 2)>>2);
			break;
		case 3:
			read_string = raux->unpacked_queue_binary_buf_shift3+ ((raux->pivot + 1)>>2);
			break;
		case 0:
			read_string = raux->unpacked_queue_binary_buf_shift4+(raux->pivot>>2);
			break;
	}
	
	uint64_t ref_val = (*(uint64_t*)(ref_pos)) ;
	uint64_t read_val = (*(uint64_t*)(read_string));
	uint32_t compare_len;
	compare = ref_val^read_val;
	compare = compare >> (sa_pos_rest<<1);
	if( compare != 0 ){
		compare_len = ((ffs(compare)-1)>>1);
		if (read_len <= compare_len){
			*match_len = read_len;	
			// if ( read_len<ref_len ){ 
			if ( read_len<ref_len ){ 
				*exact_match_flag= true;
				return true;
			}else{
				//when Suffix array contains end of ref, which is padded TTTTTTTT
				*match_len = ref_len;
				return false;
			}
		}else{
			*match_len = compare_len;
			compare_len += sa_pos_rest;
			return (ref_val << (62-(compare_len<<1))) < (read_val << (62-(compare_len<<1)));
		}
	}
	for (uint64_t i=8; (i<<2) <read_len+sa_pos_rest ; i +=8){
		ref_val = (*(uint64_t*)(ref_pos+i)) ;
		read_val = (*(uint64_t*)(read_string+i)) ;
		compare = ref_val^read_val;
		if(compare != 0 ){
			compare_len = (i<<2)+ ((ffs(compare)-1)>>1) - sa_pos_rest;
			if (read_len <= compare_len){
				*match_len = read_len;	
				if (read_len<ref_len){ // if read_len == valid_len
					*exact_match_flag= true;
					return true;
				}else{
					//when Suffix array contains end of ref, which is padded TTTTTTTT
					*match_len = ref_len;
					return false;
				}
			}else{
				*match_len = compare_len;
				compare_len +=  sa_pos_rest-(i<<2);
				return (ref_val << (62-(compare_len<<1))) < (read_val << (62-(compare_len<<1)));
			}
		}
	}
	*match_len = read_len;	
	if (read_len<ref_len){ // if read_len == valid_len
		*exact_match_flag= true;
		return true;
	}else{
		//when Suffix array contains end of ref, which is padded TTTTTTTT
		*match_len = ref_len;
		return false;
	}
}

inline bool compare_read_and_ref_binary_left_pos_only(const uint8_t* pac, const uint8_t* sa,const uint64_t pos, const Learned_read_aux_t* raux, const uint64_t sa_num, const uint64_t valid_len,  uint32_t* match_len, bool* exact_match_flag){
	uint64_t sa_pos = *(uint32_t*)(sa + pos);
	sa_pos = sa_pos <<8 |  sa[pos+4];
	uint64_t ref_len=sa_num - sa_pos;
	uint64_t compare; 
	uint32_t read_len= (uint32_t)std::min(ref_len,valid_len);
	uint32_t sa_pos_rest = sa_pos&3;
	uint8_t* ref_pos = pac + (sa_pos>>2);
	uint8_t* read_string;
	switch(sa_raux_buf[raux->l_pivot&3][sa_pos_rest])
	{
		case 1:
			read_string = raux->unpacked_rc_queue_binary_buf_shift1+((raux->l_pivot + 3)>>2);
			break;
		case 2:
			read_string = raux->unpacked_rc_queue_binary_buf_shift2+((raux->l_pivot + 2)>>2);
			break;
		case 3:
			read_string = raux->unpacked_rc_queue_binary_buf_shift3 + ((raux->l_pivot + 1)>>2);
			break;
		case 0:
			read_string = raux->unpacked_rc_queue_binary_buf_shift4 + (raux->l_pivot>>2);
			break;

	}
	uint64_t ref_val = (*(uint64_t*)(ref_pos)) ;
	uint64_t read_val = (*(uint64_t*)(read_string));
	uint32_t compare_len;
	compare = ref_val^read_val;
	compare = compare >> (sa_pos_rest<<1);
	if(compare != 0 ){
		compare_len = ((ffs(compare)-1)>>1);
		if (read_len <= compare_len){
			*match_len = read_len;	
			if (read_len<ref_len){ // if read_len == valid_len
				*exact_match_flag= true;
				return true;
			}else{
				//when Suffix array contains end of ref, which is padded TTTTTTTT
				*match_len = ref_len;
				return false;
			}
		}else{
			*match_len = compare_len;
			compare_len += sa_pos_rest;
			return (ref_val << (62-(compare_len<<1))) < (read_val << (62-(compare_len<<1)));
		}
	}

	for (uint64_t i=8; (i<<2) <read_len+sa_pos_rest ; i +=8){
		ref_val = (*(uint64_t*)(ref_pos+i)) ;
		read_val = (*(uint64_t*)(read_string+i)) ;
		compare = ref_val^read_val;
		if(compare != 0 ){
			compare_len = (i<<2)+ ((ffs(compare)-1)>>1) - sa_pos_rest;
			if (read_len <= compare_len){
				*match_len = read_len;	
				if (read_len<ref_len){ // if read_len == valid_len
					*exact_match_flag= true;
					return true;
				}else{
					//when Suffix array contains end of ref, which is padded TTTTTTTT
					*match_len = ref_len;
					return false;
				}
			}else{
				*match_len = compare_len;
				compare_len+= sa_pos_rest -(i<<2);
				return (ref_val << (62-(compare_len<<1))) < (read_val << (62-(compare_len<<1)));
			}
		}

	}
	*match_len = read_len;	
	if (read_len<ref_len){ // if read_len == valid_len
		*exact_match_flag= true;
		return true;
	}else{
		//when Suffix array contains end of ref, which is padded TTTTTTTT
		*match_len = ref_len;
		return false;
	}
}

#if LOADSUFFIX
	#define compare_read_and_ref_binary(pac, sa_pos, iter_pos, raux, sa_num, read_valid_len, match_len, exact_match_flag) compare_read_and_ref_binary_LOADSUFFIX(pac, sa_pos, iter_pos*SASIZE, raux, sa_num, read_valid_len, match_len, exact_match_flag) 
	#define compare_read_and_ref_binary_left(pac, sa_pos, iter_pos, raux, sa_num, read_valid_len, match_len, exact_match_flag) compare_read_and_ref_binary_left_LOADSUFFIX(pac, sa_pos, iter_pos*SASIZE, raux, sa_num, read_valid_len, match_len, exact_match_flag) 
#else
	#define compare_read_and_ref_binary(pac, sa_pos, iter_pos, raux, sa_num, read_valid_len, match_len, exact_match_flag) compare_read_and_ref_binary_pos_only(pac, sa_pos, iter_pos*5, raux, sa_num, read_valid_len, match_len, exact_match_flag) 
	#define compare_read_and_ref_binary_left(pac, sa_pos, iter_pos, raux, sa_num, read_valid_len, match_len, exact_match_flag) compare_read_and_ref_binary_left_pos_only(pac, sa_pos, iter_pos*5, raux, sa_num, read_valid_len, match_len, exact_match_flag) 
#endif




#if __AVX512BW__
inline uint64_t Tokenization( Learned_read_aux_t* raux, bool right_forward, uint32_t* ambiguous_pos, bool hasN){
	// make key from read
	// 1. if (read length - pivot) is smaller than 32 (query_k_mer_size)
	// 2. if (unambiguous base N appears)
	uint64_t key=0, len, r, ffs_pos;
	uint32_t pivot;
	*ambiguous_pos = raux->l_seq;
	
	if (right_forward){
		if (hasN){
			__m512i ambiguous_comp = _mm512_maskz_set1_epi8(0xFFFFFFFFFFFFFFFF , 4);
			switch((raux->pivot&3))
			{
				case 0:
					pivot= raux->pivot>>2;
					for (r=pivot;r<pivot+8;r++){
						key = key <<8;
						key |= BitReverseTable256[*(raux->unpacked_queue_binary_buf_shift4 + r)] ;
					}
					break;	
				case 1:
					pivot= ((raux->pivot+3)>>2) ;
					for (r=pivot;r<pivot+8;r++){
						key = key <<8;
						key |= BitReverseTable256[*(raux->unpacked_queue_binary_buf_shift1  + r)] ;
					}
					break;	
				case 2:
					pivot= ((raux->pivot+2)>>2);
					for (r=pivot;r<pivot+8;r++){
						key = key <<8;
						key |= BitReverseTable256[*(raux->unpacked_queue_binary_buf_shift2  + r)] ;
					}
					break;	
				case 3:
					pivot= ((raux->pivot+1)>>2);
					for (r=pivot;r<pivot+8;r++){
						key = key <<8;
						key |= BitReverseTable256[*(raux->unpacked_queue_binary_buf_shift3  + r)] ;
					}
					break;	
			}

			for (r=raux->pivot; r<raux->l_seq; r+=64){
				// find ambiguous pos starting from raux->pivot
				__mmask64 mask_val = 0xFFFFFFFFFFFFFFFF;
				if (raux->l_seq - r < 64){
					mask_val >>=  64-(raux->l_seq - r);
				}
				__m512i  read_val_512 = _mm512_maskz_loadu_epi8(mask_val, raux->unpacked_queue_buf+ r);
				__mmask64 result =  _mm512_cmpeq_epi8_mask(ambiguous_comp, read_val_512);
				ffs_pos = ffs(result);
				if (ffs_pos ){
					*ambiguous_pos = std::min(ffs_pos-1 + r, (uint64_t)raux->l_seq);
					break;
				}
			}

			return key;

		}else{
			switch((raux->pivot&3))
			{
				case 0:
					pivot= raux->pivot>>2;
					for (r=pivot;r<pivot+8;r++){
						key = key <<8;
						key |= BitReverseTable256[*(raux->unpacked_queue_binary_buf_shift4 + r)] ;
					}
					break;	
				case 1:
					pivot= ((raux->pivot+3)>>2) ;
					for (r=pivot;r<pivot+8;r++){
						key = key <<8;
						key |= BitReverseTable256[*(raux->unpacked_queue_binary_buf_shift1  + r)] ;
					}
					break;	
				case 2:
					pivot= ((raux->pivot+2)>>2);
					for (r=pivot;r<pivot+8;r++){
						key = key <<8;
						key |= BitReverseTable256[*(raux->unpacked_queue_binary_buf_shift2  + r)] ;
					}
					break;	
				case 3:
					pivot= ((raux->pivot+1)>>2);
					for (r=pivot;r<pivot+8;r++){
						key = key <<8;
						key |= BitReverseTable256[*(raux->unpacked_queue_binary_buf_shift3  + r)] ;
					}
					break;	
			}
			return key;
		}
	}else{
		if (hasN){
			__m512i ambiguous_comp = _mm512_maskz_set1_epi8(0xFFFFFFFFFFFFFFFF , 4);
			switch((raux->l_pivot&3))
			{
				case 0:
					pivot= raux->l_pivot>>2;
					for (r=pivot;r<pivot+8;r++){
						key = key <<8;
						key |= BitReverseTable256[*(raux->unpacked_rc_queue_binary_buf_shift4 + r)] ;
					}
					break;	
				case 1:
					pivot= ((raux->l_pivot+3)>>2) ;
					for (r=pivot;r<pivot+8;r++){
						key = key <<8;
						key |= BitReverseTable256[*(raux->unpacked_rc_queue_binary_buf_shift1  + r)] ;
					}
					break;	
				case 2:
					pivot= ((raux->l_pivot+2)>>2);
					for (r=pivot;r<pivot+8;r++){
						key = key <<8;
						key |= BitReverseTable256[*(raux->unpacked_rc_queue_binary_buf_shift2  + r)] ;
					}
					break;	
				case 3:
					pivot= ((raux->l_pivot+1)>>2);
					for (r=pivot;r<pivot+8;r++){
						key = key <<8;
						key |= BitReverseTable256[*(raux->unpacked_rc_queue_binary_buf_shift3  + r)] ;
					}
					break;	
			}
			for (r=raux->l_pivot; r<raux->l_seq; r+=64){
				// find ambiguous pos starting from raux->pivot
				__mmask64 mask_val = 0xFFFFFFFFFFFFFFFF;
				if (raux->l_seq - r < 64){
					mask_val >>=  64-(raux->l_seq - r);
				}
				__m512i  read_val_512 = _mm512_maskz_loadu_epi8(mask_val, raux->unpacked_rc_queue_buf+ r);
				__mmask64 result =  _mm512_cmpeq_epi8_mask(ambiguous_comp, read_val_512);
				ffs_pos = ffs(result);
				if (ffs_pos ){
					*ambiguous_pos = std::min(ffs_pos-1 + r, (uint64_t)raux->l_seq);
					break;
				}
			}
			return key;
		}
		else{
			switch((raux->l_pivot&3))
			{
				case 0:
					pivot= raux->l_pivot>>2;
					for (r=pivot;r<pivot+8;r++){
						key = key <<8;
						key |= BitReverseTable256[*(raux->unpacked_rc_queue_binary_buf_shift4 + r)] ;
					}
					break;	
				case 1:
					pivot= ((raux->l_pivot+3)>>2) ;
					for (r=pivot;r<pivot+8;r++){
						key = key <<8;
						key |= BitReverseTable256[*(raux->unpacked_rc_queue_binary_buf_shift1  + r)] ;
					}
					break;	
				case 2:
					pivot= ((raux->l_pivot+2)>>2);
					for (r=pivot;r<pivot+8;r++){
						key = key <<8;
						key |= BitReverseTable256[*(raux->unpacked_rc_queue_binary_buf_shift2  + r)] ;
					}
					break;	
				case 3:
					pivot= ((raux->l_pivot+1)>>2);
					for (r=pivot;r<pivot+8;r++){
						key = key <<8;
						key |= BitReverseTable256[*(raux->unpacked_rc_queue_binary_buf_shift3  + r)] ;
					}
					break;
			}
			return key;
		}
	}
}

#else
inline uint64_t Tokenization( Learned_read_aux_t* raux, bool right_forward, uint32_t* ambiguous_pos, bool hasN){
	// make key from read
	// 1. if (read length - pivot) is smaller than 32 (query_k_mer_size)
	// 2. if (unambiguous base N appears)
	uint64_t key=0, len, r;
	*ambiguous_pos = raux->l_seq;
	if (right_forward){
		if (hasN){
			len = raux->l_seq - raux->pivot;
			for (r=0; r< len && r < 32; r++){
				if (raux->unpacked_queue_buf[raux->pivot + r] >= 4){
					//unambiguous base N is found
					*ambiguous_pos = raux->pivot + r;
					break;
				}
				key = key << 2;
				key |= raux->unpacked_queue_buf[raux->pivot + r];
			}
			for (; r < 32; r++){
				key = key << 2;
	#if PAD_1
				key |= 3;
	#endif
			}
			// iterate until end of read, find position of N
			if (*ambiguous_pos == raux->l_seq){
				for (;r<len;r++){
					if (raux->unpacked_queue_buf[raux->pivot + r] >= 4){
						//unambiguous base N is found
						*ambiguous_pos = raux->pivot + r;
						break;
					}
				}
			}
	#if DEBUG_MODE
			assert(len!=0);
	#endif
			return key;
		}else{
			len = raux->l_seq - raux->pivot;
			for (r=0; r< len && r < 32; r++){
				key = key << 2;
				key |= raux->unpacked_queue_buf[raux->pivot + r];
			}
			for (; r < 32; r++){
				key = key << 2;
	#if PAD_1
				key |= 3;
	#endif
			}
	#if DEBUG_MODE
			assert(len!=0);
	#endif
			return key;
		}
	}else{
		if (hasN){
			len = raux->l_seq - raux->l_pivot;
			for (r=0; r< len && r < 32; r++){
				if (raux->unpacked_rc_queue_buf[ raux->l_pivot + r] >= 4){
					//unambiguous base N is found
					*ambiguous_pos = raux->l_pivot + r;
					break;
				}
				key = key << 2;
				key |= raux->unpacked_rc_queue_buf[ raux->l_pivot + r];
			}	
			for (; r < 32; r++){
				key = key << 2;
	#if PAD_1
				key |= 3;
	#endif
			}
			// iterate until end of read, find position of N
			if (*ambiguous_pos == raux->l_seq){
				for (;r<len;r++){
					if (raux->unpacked_rc_queue_buf[raux->l_pivot + r] >= 4){
						//unambiguous base N is found
						*ambiguous_pos = raux->l_pivot + r;
						break;
					}
				}
			}
	#if DEBUG_MODE			
			assert(len!=0);
	#endif			
			return key;
		}
		else{
			len = raux->l_seq - raux->l_pivot;
			for (r=0; r< len && r < 32; r++){
				key = key << 2;
				key |= raux->unpacked_rc_queue_buf[ raux->l_pivot + r];
			}	
			for (; r < 32; r++){
				key = key << 2;
	#if PAD_1
				key |= 3;
	#endif
			}
	#if DEBUG_MODE			
			assert(len!=0);
	#endif	
			return key;
		}
	}
}
#endif

void Learned_getSMEMsAllPosOneThread_step1only(Learned_index_aux_t* iaux, Learned_read_aux_t* raux, mem_tlv* smems, u64v* hits, bool hasN, int split_len, int split_width){
	set_forward_pivot(raux, 0);
	// loop until raux->pivot reaches end of read
	while (raux->pivot < raux->l_seq){
		// seeding step 1
		Learned_getSMEMsOnePosOneThread_step1(iaux,raux, smems, hits, hasN);
	}
}

void Learned_getSMEMsAllPosOneThread(Learned_index_aux_t* iaux, Learned_read_aux_t* raux, mem_tlv* smems, u64v* hits, bool hasN, int split_len, int split_width){
	set_forward_pivot(raux, 0);
	// loop until raux->pivot reaches end of read
	while (raux->pivot < raux->l_seq){
#if MEM_TRADEOFF_CACHED
		// seeding step 1
		int before = smems->n;
		Learned_getSMEMsOnePosOneThread_step1(iaux,raux, smems, hits, hasN);
		int after = smems->n;
		// seeding step 2
		for (int k = before; k < after; ++k) {
			int next_pivot = raux->pivot;
			int original_min_intv_limit = raux->min_intv_limit;
			int qbeg = smems->a[k].start;
			int qend = smems->a[k].end;
			if ((qend - qbeg) < split_len || smems->a[k].hitcount > split_width) {
				set_forward_pivot(raux, next_pivot);
				continue;
			}
			set_forward_pivot(raux, (qbeg + qend) >> 1);
			raux->min_intv_limit = smems->a[k].hitcount+1;
			// heuristic configuration to reuse cached position (Inverse Suffix Array)
			if ( smems->a[k].hitcount >= 1 && smems->a[k].hitcount < 10 ){
				raux->cache_pivot_end = smems->a[k].end;
				raux->cache_pivot = smems->a[k].start;
				raux->cache_refpos = smems->a[k].cache_refpos;
				Learned_getSMEMsOnePosOneThread(iaux,raux, smems, hits, hasN, true);
			}
			else{
				Learned_getSMEMsOnePosOneThread(iaux,raux, smems, hits, hasN, false);
			}

			raux->min_intv_limit = original_min_intv_limit;
			set_forward_pivot(raux, next_pivot);
		}
#else
		// seeding step 1
		int before = smems->n;
		Learned_getSMEMsOnePosOneThread_step1(iaux,raux, smems, hits, hasN);
		int after = smems->n;
		// seeding step 2
		for (int k = before; k < after; ++k) {
			int next_pivot = raux->pivot;
			int original_min_intv_limit = raux->min_intv_limit;
			int qbeg = smems->a[k].start;
			int qend = smems->a[k].end;
			if ((qend - qbeg) < split_len || smems->a[k].hitcount > split_width) {
				set_forward_pivot(raux, next_pivot);
				continue;
			}
			set_forward_pivot(raux, (qbeg + qend) >> 1);
			raux->min_intv_limit = smems->a[k].hitcount+1;
			Learned_getSMEMsOnePosOneThread(iaux,raux, smems, hits, hasN, false);

			raux->min_intv_limit = original_min_intv_limit;
			set_forward_pivot(raux, next_pivot);
		}
#endif
	}
}

/* ===========================================================================
 * next_pivot_key — compute 64-bit P-RMI key for pivot p, no raux mutation
 *
 * Reads directly from the pre-built shift buffers (same logic as Tokenization
 * right_forward=true, hasN=false fast path) to obtain the key used for the
 * L2_PARAMETERS model-index computation.  Used only to issue _mm_prefetch;
 * does NOT set ambiguous_pos.
 * ===========================================================================*/
static inline uint64_t next_pivot_key(const Learned_read_aux_t* raux, int p)
{
    uint64_t key = 0;
    uint32_t pivot;
    switch (p & 3) {
        case 0:
            pivot = (uint32_t)(p >> 2);
            for (uint32_t r = pivot; r < pivot + 8; r++)
                key = (key << 8) | BitReverseTable256[raux->unpacked_queue_binary_buf_shift4[r]];
            break;
        case 1:
            pivot = (uint32_t)((p + 3) >> 2);
            for (uint32_t r = pivot; r < pivot + 8; r++)
                key = (key << 8) | BitReverseTable256[raux->unpacked_queue_binary_buf_shift1[r]];
            break;
        case 2:
            pivot = (uint32_t)((p + 2) >> 2);
            for (uint32_t r = pivot; r < pivot + 8; r++)
                key = (key << 8) | BitReverseTable256[raux->unpacked_queue_binary_buf_shift2[r]];
            break;
        default: /* case 3 */
            pivot = (uint32_t)((p + 1) >> 2);
            for (uint32_t r = pivot; r < pivot + 8; r++)
                key = (key << 8) | BitReverseTable256[raux->unpacked_queue_binary_buf_shift3[r]];
            break;
    }
    return key;
}

/* ===========================================================================
 * Learned_getSMEMsAllPosOneThread_simd8 (-8 mode)
 *
 * Loop-internal next-pivot prefetch.
 *
 * Design:
 *   The outer loop mirrors Learned_getSMEMsAllPosOneThread exactly, but
 *   BEFORE calling step1 for the current pivot it fires a software prefetch
 *   for the L2_PARAMETERS cache line of the *next* non-ambiguous pivot.
 *
 *   Why this is better than the old Phase-1 pre-pass:
 *   - Old pre-pass: prefetched ALL l_seq pivots (e.g. 150), but seeding only
 *     visits ~5 pivots (pivot jumps by match_len ~30 each step).
 *     => 97% of Tokenization calls and prefetches were wasted.
 *   - New approach: prefetch distance = 1 real seeding step ahead, so every
 *     prefetch is actually consumed.  next_pivot_key() costs ~10 cycles vs.
 *     the ~150-cycle DRAM stall it hides.
 *
 *   The prefetch is issued right before the SA binary-search for the current
 *   pivot (~hundreds of cycles), giving enough time for the hardware to fetch
 *   the L2_PARAMETERS line before the next iteration needs it.
 * ===========================================================================*/
void Learned_getSMEMsAllPosOneThread_simd8(Learned_index_aux_t* iaux, Learned_read_aux_t* raux, mem_tlv* smems, u64v* hits, bool hasN, int split_len, int split_width)
{
    set_forward_pivot(raux, 0);

    while (raux->pivot < raux->l_seq) {
        /* ---- prefetch L2_PARAMETERS for the next non-ambiguous pivot ---- */
        {
            /* Estimate next pivot using min_seed_len as lower bound.
             * This is almost always the correct next starting point and
             * ensures every prefetch is consumed before the next iteration. */
            int next_p = raux->pivot + raux->min_seed_len;
            while (next_p < raux->l_seq - raux->min_seed_len + 1 &&
                   raux->unpacked_queue_buf[next_p] >= 4)
                next_p++;
            if (next_p < raux->l_seq - raux->min_seed_len + 1 &&
                raux->unpacked_queue_buf[next_p] < 4) {
                uint64_t nkey = next_pivot_key(raux, next_p);
                size_t mi_pf  = nkey >> bit_shift;
                _mm_prefetch((const char*)(L2_PARAMETERS + mi_pf * 24), _MM_HINT_T0);
            }
        }

        /* ---- step1 + step2: identical logic to getSMEMsAllPosOneThread ---- */
#if MEM_TRADEOFF_CACHED
        int before = smems->n;
        Learned_getSMEMsOnePosOneThread_step1(iaux, raux, smems, hits, hasN);
        int after = smems->n;
        for (int k = before; k < after; ++k) {
            int next_pivot_saved  = raux->pivot;
            int original_min_intv = raux->min_intv_limit;
            int qbeg = smems->a[k].start;
            int qend = smems->a[k].end;
            if ((qend - qbeg) < split_len || smems->a[k].hitcount > split_width) {
                set_forward_pivot(raux, next_pivot_saved);
                continue;
            }
            set_forward_pivot(raux, (qbeg + qend) >> 1);
            raux->min_intv_limit = smems->a[k].hitcount + 1;
            if (smems->a[k].hitcount >= 1 && smems->a[k].hitcount < 10) {
                raux->cache_pivot_end = smems->a[k].end;
                raux->cache_pivot     = smems->a[k].start;
                raux->cache_refpos    = smems->a[k].cache_refpos;
                Learned_getSMEMsOnePosOneThread(iaux, raux, smems, hits, hasN, true);
            } else {
                Learned_getSMEMsOnePosOneThread(iaux, raux, smems, hits, hasN, false);
            }
            raux->min_intv_limit = original_min_intv;
            set_forward_pivot(raux, next_pivot_saved);
        }
#else
        int before = smems->n;
        Learned_getSMEMsOnePosOneThread_step1(iaux, raux, smems, hits, hasN);
        int after = smems->n;
        for (int k = before; k < after; ++k) {
            int next_pivot_saved  = raux->pivot;
            int original_min_intv = raux->min_intv_limit;
            int qbeg = smems->a[k].start;
            int qend = smems->a[k].end;
            if ((qend - qbeg) < split_len || smems->a[k].hitcount > split_width) {
                set_forward_pivot(raux, next_pivot_saved);
                continue;
            }
            set_forward_pivot(raux, (qbeg + qend) >> 1);
            raux->min_intv_limit = smems->a[k].hitcount + 1;
            Learned_getSMEMsOnePosOneThread(iaux, raux, smems, hits, hasN, false);
            raux->min_intv_limit = original_min_intv;
            set_forward_pivot(raux, next_pivot_saved);
        }
#endif
    }
}

/* ===========================================================================
 * Learned_getSMEMsAllPos_inter_read_batch  (-8 only)
 *
 * Pivot-level inter-read batching: process INTER_READ_BATCH reads in lock-step.
 *
 * Each "tick":
 *   Phase A — for every active read, compute Tokenization key for current
 *             pivot, run P-RMI (learned_index_lookup), issue SA prefetch.
 *             All DRAM latencies for SA start in parallel.
 *   Phase B — for every active read, execute the full step1 + step2 seeding
 *             (SA binary-search, SMEM expansion).  By the time we reach each
 *             read the SA cache line is (mostly) warm.
 *
 * A read is "done" when its pivot reaches l_seq.
 * The outer caller (mem_kernel1_core_Learned) feeds reads in groups of
 * INTER_READ_BATCH; leftover reads fall back to the single-read simd8 path.
 * ===========================================================================*/
#define INTER_READ_BATCH 4

/* helper: compute key + P-RMI + prefetch SA for raux at current pivot */
static inline void pivot_prmi_sa_prefetch(Learned_index_aux_t* iaux,
                                          Learned_read_aux_t*  raux,
                                          bool hasN)
{
    if (raux->pivot >= raux->l_seq) return;
    if (raux->unpacked_queue_buf[raux->pivot] >= 4) return;

    uint32_t ambiguous_pos;
    uint64_t key = Tokenization(raux, /*right_forward=*/true, &ambiguous_pos, hasN);
    uint32_t read_valid_len = ambiguous_pos - raux->pivot;
    if ((int)read_valid_len < raux->min_seed_len) return;

    size_t enc_err;
    uint64_t est_pos = learned_index_lookup(key, &enc_err);
    _mm_prefetch((const char*)(iaux->sa_pos + est_pos * SASIZE), _MM_HINT_T0);
}

void Learned_getSMEMsAllPos_inter_read_batch(
        Learned_index_aux_t**  iaux_arr,
        Learned_read_aux_t**   raux_arr,
        mem_tlv**              smems_arr,
        u64v**                 hits_arr,
        bool*                  hasN_arr,
        int                    n_reads,
        int split_len, int split_width,
        int max_mem_intv,
        int max_mem_intv_min_seed_len)
{
    /* initialise all pivots */
    for (int i = 0; i < n_reads; i++)
        set_forward_pivot(raux_arr[i], 0);

    bool any_active = true;
    while (any_active) {
        any_active = false;

        /* ── Phase A: batch P-RMI + SA prefetch for all active reads ── */
        for (int i = 0; i < n_reads; i++) {
            Learned_read_aux_t* raux = raux_arr[i];
            if (raux->pivot >= raux->l_seq) continue;
            any_active = true;

            /* also prefetch L2_PARAMETERS for next pivot (intra-read) */
            {
                int next_p = raux->pivot + raux->min_seed_len;
                while (next_p < raux->l_seq - raux->min_seed_len + 1 &&
                       raux->unpacked_queue_buf[next_p] >= 4)
                    next_p++;
                if (next_p < raux->l_seq - raux->min_seed_len + 1 &&
                    raux->unpacked_queue_buf[next_p] < 4) {
                    uint64_t nkey = next_pivot_key(raux, next_p);
                    size_t mi_pf  = nkey >> bit_shift;
                    _mm_prefetch((const char*)(L2_PARAMETERS + mi_pf * 24), _MM_HINT_T0);
                }
            }
            /* inter-read SA prefetch */
            pivot_prmi_sa_prefetch(iaux_arr[i], raux, hasN_arr[i]);
        }

        if (!any_active) break;

        /* ── Phase B: full seeding for each active read ── */
        for (int i = 0; i < n_reads; i++) {
            Learned_read_aux_t* raux = raux_arr[i];
            if (raux->pivot >= raux->l_seq) continue;

            Learned_index_aux_t* iaux  = iaux_arr[i];
            mem_tlv*             smems = smems_arr[i];
            u64v*                hits  = hits_arr[i];
            bool                 hasN  = hasN_arr[i];

#if MEM_TRADEOFF_CACHED
            int before = smems->n;
            Learned_getSMEMsOnePosOneThread_step1(iaux, raux, smems, hits, hasN);
            int after = smems->n;
            for (int k = before; k < after; ++k) {
                int next_pivot_saved  = raux->pivot;
                int original_min_intv = raux->min_intv_limit;
                int qbeg = smems->a[k].start;
                int qend = smems->a[k].end;
                if ((qend - qbeg) < split_len || smems->a[k].hitcount > split_width) {
                    set_forward_pivot(raux, next_pivot_saved);
                    continue;
                }
                set_forward_pivot(raux, (qbeg + qend) >> 1);
                raux->min_intv_limit = smems->a[k].hitcount + 1;
                if (smems->a[k].hitcount >= 1 && smems->a[k].hitcount < 10) {
                    raux->cache_pivot_end = smems->a[k].end;
                    raux->cache_pivot     = smems->a[k].start;
                    raux->cache_refpos    = smems->a[k].cache_refpos;
                    Learned_getSMEMsOnePosOneThread(iaux, raux, smems, hits, hasN, true);
                } else {
                    Learned_getSMEMsOnePosOneThread(iaux, raux, smems, hits, hasN, false);
                }
                raux->min_intv_limit = original_min_intv;
                set_forward_pivot(raux, next_pivot_saved);
            }
#else
            int before = smems->n;
            Learned_getSMEMsOnePosOneThread_step1(iaux, raux, smems, hits, hasN);
            int after = smems->n;
            for (int k = before; k < after; ++k) {
                int next_pivot_saved  = raux->pivot;
                int original_min_intv = raux->min_intv_limit;
                int qbeg = smems->a[k].start;
                int qend = smems->a[k].end;
                if ((qend - qbeg) < split_len || smems->a[k].hitcount > split_width) {
                    set_forward_pivot(raux, next_pivot_saved);
                    continue;
                }
                set_forward_pivot(raux, (qbeg + qend) >> 1);
                raux->min_intv_limit = smems->a[k].hitcount + 1;
                Learned_getSMEMsOnePosOneThread(iaux, raux, smems, hits, hasN, false);
                raux->min_intv_limit = original_min_intv;
                set_forward_pivot(raux, next_pivot_saved);
            }
#endif
        }
    }

    /* ── Post-seeding: optional bwtSeedStrategy for each read ── */
    if (max_mem_intv > 0) {
        for (int i = 0; i < n_reads; i++) {
            Learned_read_aux_t*  raux  = raux_arr[i];
            Learned_index_aux_t* iaux  = iaux_arr[i];
            mem_tlv*             smems = smems_arr[i];
            u64v*                hits  = hits_arr[i];
            bool                 hasN  = hasN_arr[i];

            int saved_min_intv     = raux->min_intv_limit;
            int saved_min_seed_len = raux->min_seed_len;

            raux->min_intv_limit = max_mem_intv;
            raux->min_seed_len   = max_mem_intv_min_seed_len;

            if (raux->max_l_seq != 0) {
                Learned_bwtSeedStrategyAllPosOneThread_mem_tradeoff(iaux, raux, smems, hits, hasN);
            } else {
                Learned_bwtSeedStrategyAllPosOneThread(iaux, raux, smems, hits, hasN);
            }

            raux->min_intv_limit = saved_min_intv;
            raux->min_seed_len   = saved_min_seed_len;
        }
    }
}


void Learned_bwtSeedStrategyAllPosOneThread(Learned_index_aux_t* iaux, Learned_read_aux_t* raux, mem_tlv* smems, u64v* hits, bool hasN){
		set_forward_pivot(raux, 0);
		// loop until raux->pivot reaches end of read
		int min_intv_value = raux->min_intv_limit;
		uint8_t* ref_string = iaux->ref_string;
		uint8_t* sa_pos = iaux->sa_pos;
		uint64_t sa_num = iaux->bns->l_pac*2;
		bool exact_match_flag;
		while (raux->pivot < raux->l_seq - raux->min_seed_len + 1){
			uint64_t key ;
			uint64_t enc_err ;
			uint64_t curr_err, err;
			uint64_t iter_pos ;
			uint32_t last_match_len ,match_len;
			uint32_t ambiguous_pos;
			uint32_t read_valid_len;
			bool right_forward;
			exact_match_flag=false;
			mem_tl mem;
	#if Count_mem_ref
			uint32_t count_search_bs=0;
			uint32_t count_search_linear=0;
			uint32_t count_search_min_intv=0;
			uint32_t count_smem=0;
	#endif
			// Check pivot point whether ambiguous base appears
			if (raux->unpacked_queue_buf[raux->pivot] >= 4){
				set_forward_pivot(raux, raux->pivot+1 );
				continue;
			}
			right_forward = true;


			key = Tokenization(raux, right_forward, &ambiguous_pos, hasN);
			read_valid_len = ambiguous_pos - raux->pivot;
			if (read_valid_len < raux->min_seed_len){
				set_forward_pivot(raux,    raux->pivot + read_valid_len );
				continue;
			}
			iter_pos = learned_index_lookup(key ,&enc_err);
			curr_err= (enc_err>>32) & 0x3fffffff;
			err=(enc_err<<32)>>32 & 0x7fffffff;
			// search for curr_err position first and do binary search on remaining bound
    #if CURR_SEARCH_METHOD == 2
			uint64_t exp_search_move=MEM_TRADEOFF_USECACHE_EXP_SEARCH_START;
			// do exponential search
			uint64_t upper_b,lower_b,n,middle, half; // variables for binary search
			iter_pos = iter_pos > MEM_TRADEOFF_USECACHE_EXP_SEARCH_START? iter_pos:MEM_TRADEOFF_USECACHE_EXP_SEARCH_START;
			iter_pos = iter_pos < sa_num-1 - MEM_TRADEOFF_USECACHE_EXP_SEARCH_START? iter_pos:sa_num - 1-MEM_TRADEOFF_USECACHE_EXP_SEARCH_START;
	#if Count_mem_ref
			// count_search_exp++;
			count_search_bs++;
	#endif	
			if (!compare_read_and_ref_binary(iaux->pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)){
				if (exact_match_flag){
					lower_b=iter_pos;
					n=1;
				}else{
					iter_pos -= exp_search_move;
		#if PREFETCH
					_mm_prefetch(sa_pos+ ((iter_pos - exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
		#endif
		#if Count_mem_ref
					count_search_bs++;
		#endif
					while (!compare_read_and_ref_binary(iaux->pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
						if (iter_pos ==0){
							// smaller than iter_pos 0 -> search_start_pos should be 0
							break;
						}
						exp_search_move *=MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW;
						if (iter_pos > exp_search_move){
							
							iter_pos -= exp_search_move;
		#if PREFETCH
							_mm_prefetch(sa_pos+ ((iter_pos - exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
		#endif								
						}else{
							exp_search_move=iter_pos;
							iter_pos = 0;
						}
		#if Count_mem_ref
						count_search_bs++;
		#endif
						
					}
					if (exact_match_flag){
						lower_b=iter_pos;
						n=1;
					}else{
						upper_b = iter_pos + exp_search_move>= sa_num-1? sa_num-2: iter_pos + exp_search_move;
						lower_b = iter_pos >0? iter_pos:1;
						n=  upper_b-lower_b + 1;
					}
					
				}
			}
			else{
				if (exact_match_flag){
					lower_b=iter_pos;
					n=1;
				}else{
					//  estimated position have smaller key than read, increase iter_pos and compare
					iter_pos += exp_search_move;
		#if PREFETCH
					_mm_prefetch(sa_pos+ ((iter_pos + exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
		#endif				
		#if Count_mem_ref
					count_search_bs++;
		#endif
					while (compare_read_and_ref_binary(iaux->pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
						if (iter_pos == sa_num - 1){
							break;
						}
						exp_search_move *=MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW;
						if (sa_num - 1 > iter_pos+exp_search_move){
							iter_pos += exp_search_move;
		#if PREFETCH
							_mm_prefetch(sa_pos+ ((iter_pos + exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
		#endif						
						}else{
							exp_search_move=sa_num - iter_pos -1;
							iter_pos = sa_num - 1;
						}
		#if Count_mem_ref
						// count_search_exp++;
						count_search_bs++;
		#endif
					}

					if (exact_match_flag){
						lower_b=iter_pos;
						n=1;
					}else{
						upper_b = iter_pos >= sa_num -1? sa_num-2: iter_pos ;
						lower_b = iter_pos > exp_search_move ? iter_pos- exp_search_move:1;
						n=  upper_b-lower_b + 1;
					}
				}
			}
			middle=iter_pos;
	#elif CURR_SEARCH_METHOD ==1
			uint64_t upper_b = iter_pos+err >= sa_num-1? sa_num-2: iter_pos+err;
			uint64_t lower_b = iter_pos > curr_err? iter_pos-curr_err:1;
			lower_b = lower_b >= upper_b ? upper_b - 1: lower_b;
			uint64_t n = upper_b-lower_b + 1; // `end` is inclusive.
			uint64_t middle=iter_pos;

			// Search for first occurrence of key.w
	#else	// binary search method
			err = std::max(curr_err,err);
			uint64_t upper_b = iter_pos+err >= sa_num-1? sa_num-2: iter_pos+err;
			uint64_t lower_b = iter_pos > err? iter_pos-err:1;
			lower_b = lower_b >= upper_b ? upper_b - 1: lower_b;
			uint64_t n = upper_b-lower_b + 1; // `end` is inclusive.
			uint64_t middle=iter_pos;
	#endif
			// Function adapted from https://github.com/gvinciguerra/rmi_pgm/blob/357acf668c22f927660d6ed11a15408f722ea348/main.cpp#L29.
			// Authored by Giorgio Vinciguerra.
			while (uint64_t half = (n>>1)) {
				middle = lower_b + half;
	#if Count_mem_ref
				count_search_bs++;
	#endif
	#if PREFETCH
				_mm_prefetch(sa_pos+ (middle+half/2)*SASIZE, _MM_HINT_T0);
				_mm_prefetch(sa_pos+((lower_b+half/2)*SASIZE), _MM_HINT_T0);
	#endif
				lower_b = compare_read_and_ref_binary(iaux->pac, sa_pos, middle, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)? middle: lower_b;
			
				if(exact_match_flag){
					break;
				}
				n -= half;
			}
			// no need to do additional linear search
			if(middle!=lower_b){
				// lookup key was smaller than middle, should check lower_b
				last_match_len = match_len;
				iter_pos = lower_b;
	#if Count_mem_ref
				count_search_linear++;
	#endif			
				while (!compare_read_and_ref_binary(iaux->pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
					if (iter_pos ==0){
						break;
					}
					iter_pos --;
	#if Count_mem_ref
					count_search_linear++;
	#endif
					last_match_len = match_len;
				}
				//  stops when ref is smaller than read, iter_pos should point to exact matching or bigger ref pos
				if (last_match_len > match_len){
					iter_pos++;
					match_len = last_match_len;
				}
			}
			else{
				// lookup key was bigger than middle, should check lower_b+1
				last_match_len = match_len;
				//  estimated position have smaller key than read, increase iter_pos and compare
				iter_pos =lower_b+1;
				iter_pos = iter_pos >= sa_num ? sa_num-1 : iter_pos;
	#if Count_mem_ref
				count_search_linear++;
	#endif		
				while (compare_read_and_ref_binary(iaux->pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
					if (iter_pos == sa_num - 1){
						//
						break;
					}
					iter_pos ++;
	#if Count_mem_ref
					count_search_linear++;
	#endif				
					last_match_len = match_len;
				}
				//  stops when ref is bigger than read or exact matches with read,
				if (last_match_len > match_len){
					iter_pos--;
					match_len = last_match_len;
				}
			} //end of else
			uint64_t search_start_pos = iter_pos;
			uint64_t up=1, low=1;
			uint32_t up_match=match_len, low_match=match_len, match_num=1, min_seed_len,last_match_num=0;
			uint64_t last_iter_pos=iter_pos ;
			min_seed_len = raux->min_seed_len;
			if (match_len < min_seed_len){
				// bwtstrategy search until min_seed_len even if no candidate exists
				set_forward_pivot(raux,    raux->pivot + min_seed_len );
				continue;
			}
			bool low_search_flag, up_search_flag;
			while(1){
					while (1){
						low_search_flag = low_match >= match_len && low <= search_start_pos;
						up_search_flag = up_match >= match_len &&  sa_num-1 >= up + search_start_pos;
						if (low_search_flag){
	#if Count_mem_ref
							count_search_min_intv++;
	#endif					
							compare_read_and_ref_binary(iaux->pac, sa_pos, (search_start_pos-low), raux, sa_num,match_len, &low_match,&exact_match_flag);
							low++;
	
						}
						if(up_search_flag){
	#if Count_mem_ref
							count_search_min_intv++;
	#endif						
							compare_read_and_ref_binary(iaux->pac, sa_pos, (search_start_pos+up), raux, sa_num,match_len, &up_match,&exact_match_flag);
							up++;
	
						}
						if (up+low-3 >= min_intv_value || (!low_search_flag && !up_search_flag)){
							break;
						}
					}
					if (low_match == match_len && low > search_start_pos){
						low+=1;
						low_match = 0;
					}
					if (up_match == match_len && ((up + search_start_pos) > sa_num-1)){
						up +=1;	
						up_match = 0;
					}
					match_num = up + low -3;
					if( match_num >= min_intv_value){
						match_num = last_match_num;
						if (last_match_num == 0){
							match_num = up + low -3;
						}
						iter_pos = last_iter_pos;
						match_len = match_len+1;
						break;
					}
					if ( std::max(up_match,low_match) < min_seed_len){
						match_len= min_seed_len;
						// last_match_num = match_num;
						iter_pos=search_start_pos-low+2;
						//match_num doesn't need modification
						break;
					}
					last_match_num = match_num;
					match_len = up_match > low_match? up_match: low_match;
					iter_pos=search_start_pos-low+2;
					last_iter_pos = iter_pos;
			}//end while 1
			
			if ( match_num < min_intv_value  ){
				match_len = std::max( match_len , min_seed_len);
				mem.start = raux->pivot;
				mem.end = raux->pivot + match_len;
				mem.hitbeg = hits->n; // begin index in hits vector
				mem.hitcount = match_num; // number of hits in reference
				for (uint64_t i=0; i< match_num; i++){
					uint64_t pos_val = *(uint32_t*)(sa_pos + (iter_pos+i)*SASIZE );
					pos_val = pos_val <<8 | sa_pos[(iter_pos+i)*SASIZE + 4];
					kv_push(uint64_t, *hits, pos_val );
				}
				kv_push(mem_tl, *smems, mem);
			}
	#if Count_mem_ref
		fprintf(stdout,"[BWTSTRATEGY_func]Max match ref len:%d Count_bs:%d Count_linear:%d Count_minintv:%d input minintv:%d\n",match_len, count_search_bs, count_search_linear, count_search_min_intv, min_intv_value);
	#endif
			set_forward_pivot(raux,    raux->pivot + match_len );
		}
}
void Learned_bwtSeedStrategyAllPosOneThread_mem_tradeoff(Learned_index_aux_t* iaux, Learned_read_aux_t* raux, mem_tlv* smems, u64v* hits, bool hasN){
		set_forward_pivot(raux, 0);
		// loop until raux->pivot reaches end of read
		// bwaseedstrategy is finding shortest exact match len, satisfing min_seed_len and max_intv_ar
		int min_intv_value = raux->min_intv_limit;
		uint8_t* ref_string = iaux->ref_string;
		uint8_t* sa_pos = iaux->sa_pos;
		uint64_t sa_num = iaux->bns->l_pac*2;
		bool exact_match_flag;
		while (raux->pivot < raux->l_seq - raux->min_seed_len + 1){
			uint64_t key ;
			uint64_t iter_pos ;
			uint32_t last_match_len ,match_len;
			uint32_t ambiguous_pos;
			uint32_t read_valid_len;
			bool right_forward;
			exact_match_flag=false;
			mem_tl mem;
	#if Count_mem_ref
			uint32_t count_search_bs=0;
			uint32_t count_search_linear=0;
			uint32_t count_search_min_intv=0;
			uint32_t count_smem=0;
	#endif
			// Check pivot point whether ambiguous base appears
			if (raux->unpacked_queue_buf[raux->pivot] >= 4){
				set_forward_pivot(raux, raux->pivot+1 );
				continue;
			}
			right_forward = true;
			key = Tokenization(raux, right_forward, &ambiguous_pos, hasN);
			read_valid_len = ambiguous_pos - raux->pivot;
			if (read_valid_len < raux->min_seed_len){
					set_forward_pivot(raux,    raux->pivot + read_valid_len );
					continue;
			}
			if (raux ->max_l_seq != raux->l_seq){
				assert(0); // not implemented yet
				iter_pos =  *(uint32_t*)(iaux->ref2sa+(raux->max_refpos+raux->pivot)*5);
				iter_pos = iter_pos <<8 | iaux->ref2sa[(raux->max_refpos+raux->pivot)*5+4];
	#if Count_mem_ref
				count_search_linear++;
	#endif		
				if (!compare_read_and_ref_binary(iaux->pac,  sa_pos,iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)){
	
					last_match_len = match_len;
					iter_pos --;
	#if Count_mem_ref
					count_search_linear++;
	#endif		
					while (!compare_read_and_ref_binary(iaux->pac,  sa_pos,iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
						if (iter_pos ==0){
							break;
						}
						iter_pos --;
	#if Count_mem_ref
						count_search_linear++;
	#endif		
						last_match_len = match_len;
					}
					//  stops when ref is smaller than read, iter_pos should point to exact matching or bigger ref pos
					if (last_match_len > match_len){
						iter_pos++;
						// std::cout <<"iter pos up "<< match_len <<"\n";
						match_len = last_match_len;
					}
				}
				else{
					last_match_len = match_len;
					//  estimated position have smaller key than read, increase iter_pos and compare
					iter_pos ++;
	#if Count_mem_ref
					count_search_linear++;
	#endif		
					while (compare_read_and_ref_binary(iaux->pac, sa_pos,iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
						if (iter_pos == sa_num - 1){
							break;
						}
						iter_pos ++;
	#if Count_mem_ref
						count_search_linear++;
	#endif		
						last_match_len = match_len;
					}
					//  stops when ref is bigger than read or exact matches with read,
					if (last_match_len > match_len){
						iter_pos--;
						match_len = last_match_len;
					}
				}
			}
			else{ // exact match, no need to search
				iter_pos =  *(uint32_t*)(iaux->ref2sa+(raux->max_refpos+raux->pivot)*5);
				iter_pos = iter_pos <<8 | iaux->ref2sa[(raux->max_refpos+raux->pivot)*5+4];
				match_len = read_valid_len;
			}
			uint64_t search_start_pos = iter_pos;
			uint64_t up=1, low=1;
			uint32_t up_match=match_len, low_match=match_len, match_num=1, min_seed_len,last_match_num=0;
			uint64_t last_iter_pos=iter_pos ;
			min_seed_len = raux->min_seed_len;
			if (match_len < min_seed_len){
				// bwtstrategy search until min_seed_len even if no candidate exists
				set_forward_pivot(raux,  raux->pivot + min_seed_len );
				continue;
			}
			bool low_search_flag, up_search_flag;
			while(1){
					while (1){
						low_search_flag = low_match >= match_len && low <= search_start_pos;
						up_search_flag = up_match >= match_len &&  sa_num-1 >= up + search_start_pos;
						if (low_search_flag){
	#if Count_mem_ref
							count_search_min_intv++;
	#endif					
							compare_read_and_ref_binary(iaux->pac,  sa_pos, (search_start_pos-low), raux, sa_num,match_len, &low_match,&exact_match_flag);
							low++;
	
						}
						if(up_search_flag){
	#if Count_mem_ref
							count_search_min_intv++;
	#endif						
							compare_read_and_ref_binary(iaux->pac, sa_pos, (search_start_pos+up), raux, sa_num,match_len, &up_match,&exact_match_flag);
							up++;
	
						}
						if (up+low-3 >= min_intv_value || (!low_search_flag && !up_search_flag)){
							break;
						}
					}
					if (low_match == match_len && low > search_start_pos){
						low+=1;
						low_match = 0;
					}
					if (up_match == match_len && ((up + search_start_pos) > sa_num-1)){
						up +=1;	
						up_match = 0;
					}
					match_num = up + low -3;
					if( match_num >= min_intv_value){
						match_num = last_match_num;
						if (last_match_num == 0){
							match_num = up + low -3;
						}
						iter_pos = last_iter_pos;
						match_len = match_len+1;
						break;
					}
					if ( std::max(up_match,low_match) < min_seed_len){
						match_len= min_seed_len;
						// last_match_num = match_num;
						iter_pos=search_start_pos-low+2;
						//match_num doesn't need modification
						break;
					}
					last_match_num = match_num;
					match_len = up_match > low_match? up_match: low_match;
					iter_pos=search_start_pos-low+2;
					last_iter_pos = iter_pos;
			}//end while 1

			if ( match_num < min_intv_value  ){
				match_len = std::max( match_len , min_seed_len);
				// memset_s(&mem, sizeof(mem_tl), 0);
				mem.start = raux->pivot;
				mem.end = raux->pivot + match_len;
				mem.hitbeg = hits->n; // begin index in hits vector
				mem.hitcount = match_num; // number of hits in reference
				for (uint64_t i=0; i< match_num; i++){
					uint64_t pos_val = *(uint32_t*)(sa_pos + (iter_pos+i)*SASIZE );
					pos_val = pos_val <<8 | sa_pos[(iter_pos+i)*SASIZE + 4];
					kv_push(uint64_t, *hits, pos_val );
				}
				kv_push(mem_tl, *smems, mem);
			}
			
	#if Count_mem_ref
			fprintf(stdout,"[BWTSTRATEGY_memtradeoff_func]Max match ref len:%d Count_bs:%d Count_linear:%d Count_minintv:%d input minintv:%d\n",match_len, count_search_bs, count_search_linear, count_search_min_intv, min_intv_value);
	#endif
			set_forward_pivot(raux,    raux->pivot + match_len );
		}
}


void Learned_getSMEMsOnePosOneThread_no_smem(Learned_index_aux_t* iaux, Learned_read_aux_t* raux, mem_tlv* smems, u64v* hits, bool hasN, bool use_cached=false) {
	/*
		Perform extension in zigzag style
	*/

	uint64_t key;
    uint64_t err ;
    uint64_t position ;
	uint64_t ss_position ;
	uint32_t exact_match_len ;
	uint32_t ss_exact_match_len ;
	uint32_t next_pivot ;
	int64_t suffix_array_num = iaux->bns->l_pac*2;
	uint32_t ambiguous_pos;
	// Right extension from pivot, save next raux->pivot
	bool right_forward= true;
#if MEM_TRADEOFF
	bool no_search ; // when short read has full-match in reference, reuse position without search
#endif
	// Check pivot point whether ambiguous base appears
	if (raux->unpacked_queue_buf[raux->pivot] >= 4){
		if (raux->l_seq - raux->pivot < raux->min_seed_len ){
			set_forward_pivot(raux, raux->l_seq );
			// set_forward_pivot(raux, raux->pivot+1 );
		}
		else{
			set_forward_pivot(raux, raux->pivot+1 );
		}
		return;
	}
	if (raux->pivot !=0 && raux->unpacked_queue_buf[raux->pivot-1] < 4){
		// - 2. Infer Learned index and get prediction
		next_pivot = raux->l_seq;
		int search_pivot = raux->pivot;
		while (search_pivot < next_pivot){
			if (raux->unpacked_queue_buf[search_pivot] >= 4){
				if (raux->l_seq - search_pivot < raux->min_seed_len ){
					set_forward_pivot(raux, raux->l_seq );
					search_pivot = raux->l_seq;
					// set_forward_pivot(raux, raux->pivot+1 );
				}
				else{
					search_pivot += 1; 
					set_forward_pivot(raux, raux->pivot+1 );
				}
				continue;
			}
			// Left extension from pivot, raux->pivot should be updated at every extension direction change
#if MEM_TRADEOFF
			right_forward = false;
			// Make uint64_t Key starting from pivot point
			key = Tokenization(raux, right_forward, &ambiguous_pos, hasN);
			if (raux ->max_l_seq == raux->l_seq){
				ss_position =  *(uint32_t*)(iaux->ref2sa+(suffix_array_num - raux->max_refpos - raux->pivot-1)*5);
				ss_position = ss_position <<8 | iaux->ref2sa[(suffix_array_num - raux->max_refpos - raux->pivot-1)*5+4];
	#if PREFETCH
				_mm_prefetch(iaux->sa_pos+ (ss_position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
				no_search= true;

				ss_position = mem_search_tradeoff(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
									 raux, ss_position, right_forward, &ss_exact_match_len,&ambiguous_pos,no_search);
			}
			else{
				if (use_cached && (raux->pivot< raux->cache_pivot_end) && ( raux->cache_pivot +MEM_TRADEOFF_USECACHE_THRESHOLD <= raux->pivot) ){
	#if DEBUG_MODE
					assert(raux->pivot> raux->cache_pivot);
	#endif
					ss_position =  *(uint32_t*)(iaux->ref2sa+(suffix_array_num + raux->cache_pivot - raux->cache_refpos - raux->pivot-1)*5);
					ss_position = ss_position <<8 | iaux->ref2sa[(suffix_array_num + raux->cache_pivot - raux->cache_refpos - raux->pivot-1)*5+4];
	#if PREFETCH
					_mm_prefetch(iaux->sa_pos+ (ss_position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
					no_search=false;
					ss_position = mem_search_tradeoff(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
									 raux, ss_position, right_forward, &ss_exact_match_len,&ambiguous_pos,no_search);
				}else{
					ss_position = learned_index_lookup(key ,&err);
	#if PREFETCH
					_mm_prefetch(iaux->sa_pos+ (ss_position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
					ss_position = mem_search(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
									 raux, ss_position, err, right_forward, &ss_exact_match_len,&ambiguous_pos);
				}
			}
			set_forward_pivot(raux,  raux->pivot - ss_exact_match_len+1 );
			if(next_pivot - raux->pivot < raux->min_seed_len ){
				break;
			}
			right_forward = true;
			key = Tokenization(raux, right_forward, &ambiguous_pos, hasN);
			if (raux ->max_l_seq == raux->l_seq){
	
				ss_position =  *(uint32_t*)(iaux->ref2sa+(raux->max_refpos + raux->pivot)*5);
				ss_position = ss_position <<8 | iaux->ref2sa[(raux->max_refpos + raux->pivot)*5+4];
	#if PREFETCH
				_mm_prefetch(iaux->sa_pos+ (ss_position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
				no_search= true;
				// ss_position = right_smem_search_tradeoff(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
				// 					 raux, ss_position, &ss_exact_match_len, smems, hits, &ambiguous_pos,no_search);	
				ss_position = mem_search_tradeoff(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
									 raux, ss_position, right_forward, &ss_exact_match_len,&ambiguous_pos,no_search);
				

			}else{
				if (use_cached && (raux->cache_pivot_end>= (raux->pivot+MEM_TRADEOFF_USECACHE_THRESHOLD)) && (raux->pivot>= raux->cache_pivot) ){
	#if DEBUG_MODE
					assert(raux->pivot>= raux->cache_pivot);
	#endif
					ss_position =  *(uint32_t*)(iaux->ref2sa+(raux->cache_refpos + raux->pivot - raux->cache_pivot)*5);
					ss_position = ss_position <<8 | iaux->ref2sa[(raux->cache_refpos + raux->pivot - raux->cache_pivot)*5+4];
	#if PREFETCH
					_mm_prefetch(iaux->sa_pos+ (ss_position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
					no_search=false;
					// ss_position = right_smem_search_tradeoff(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
					// 					raux, ss_position, &ss_exact_match_len, smems, hits, &ambiguous_pos,no_search);
					ss_position = mem_search_tradeoff(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
									 raux, ss_position, right_forward, &ss_exact_match_len,&ambiguous_pos,no_search);
				}else{
					ss_position = learned_index_lookup(key ,&err);
	#if PREFETCH
					_mm_prefetch(iaux->sa_pos+ (ss_position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
					// ss_position = right_smem_search(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
					// 					raux, ss_position, err, &ss_exact_match_len, smems, hits, &ambiguous_pos);
					
					ss_position = mem_search_tradeoff(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
									 raux, ss_position, right_forward, &ss_exact_match_len,&ambiguous_pos,no_search);
	
				}
			}
#else
			right_forward = false;
			key = Tokenization(raux, right_forward, &ambiguous_pos, hasN);
			ss_position = learned_index_lookup(key ,&err);
	#if PREFETCH
			_mm_prefetch(iaux->sa_pos+ (ss_position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
			
			ss_position = mem_search(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
							raux, ss_position, err, right_forward, &ss_exact_match_len,&ambiguous_pos);
			set_forward_pivot(raux,  raux->pivot - ss_exact_match_len+1 );

			if(next_pivot - raux->pivot < raux->min_seed_len ){
				break;
			}

			right_forward = true;

			key = Tokenization(raux, right_forward, &ambiguous_pos, hasN);
			ss_position = learned_index_lookup(key ,&err);
	#if PREFETCH
			_mm_prefetch(iaux->sa_pos+ (ss_position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
			// ss_position = right_smem_search(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
			// 						 raux, ss_position, err, &ss_exact_match_len, smems, hits, &ambiguous_pos);
			
			ss_position = mem_search(iaux->ref_string, iaux->sa_pos,iaux->pac, suffix_array_num, 
									 raux, ss_position, err, right_forward, &ss_exact_match_len,&ambiguous_pos);
			
#endif

#if DEBUG_MODE
			assert( (raux->pivot+ss_exact_match_len) > search_pivot);
#endif
			search_pivot = raux->pivot + ss_exact_match_len ;
			
			set_forward_pivot(raux,   search_pivot );
		}
	}
	else{
		key = Tokenization(raux, right_forward, &ambiguous_pos, hasN);
		// - 2. Infer Learned index and get prediction		
#if MEM_TRADEOFF
		position = learned_index_lookup(key ,&err);
	#if PREFETCH
		_mm_prefetch(iaux->sa_pos+ (position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif	
		// position = right_smem_search(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
		// 							raux, position, err, &exact_match_len,smems,hits,&ambiguous_pos);

		position = mem_search(iaux->ref_string, iaux->sa_pos,iaux->pac, suffix_array_num, 
									 raux, position, err, right_forward, &exact_match_len,&ambiguous_pos);
	#if MEM_TRADEOFF_CACHED && PREFETCH
		if (exact_match_len>= raux->min_seed_len){

			uint64_t pos_val = *(uint32_t*)(iaux->sa_pos + position*SASIZE );
			pos_val = pos_val <<8 | iaux->sa_pos[position*SASIZE + 4];
			_mm_prefetch( iaux->ref2sa+ pos_val*5, _MM_HINT_T1 );
			_mm_prefetch( iaux->ref2sa+ (suffix_array_num - pos_val - exact_match_len-1)*5, _MM_HINT_T1 );
		}
	#endif
		if (exact_match_len == raux->l_seq){
			raux->max_l_seq = exact_match_len;
			uint64_t pos_val = *(uint32_t*)(iaux->sa_pos + position*SASIZE );
			pos_val = pos_val <<8 | iaux->sa_pos[position*SASIZE + 4];
			raux->max_refpos = pos_val;
			raux->max_pivot = 0;
	#if PREFETCH
			_mm_prefetch( iaux->ref2sa+ raux->max_refpos*5, _MM_HINT_T1 );
			_mm_prefetch( iaux->ref2sa+ (suffix_array_num - raux->max_refpos - raux->l_seq - 1)*5, _MM_HINT_T1 );
	#endif
		}
#else
		position = learned_index_lookup(key ,&err);
	#if PREFETCH
		_mm_prefetch(iaux->sa_pos+ (position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
		// position = right_smem_search(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
		// 							 raux, position, err, &exact_match_len,smems,hits,&ambiguous_pos);
		
		position = mem_search(iaux->ref_string, iaux->sa_pos,iaux->pac, suffix_array_num, 
									 raux, position, err, right_forward, &exact_match_len,&ambiguous_pos);
#endif	
		next_pivot = raux->pivot + exact_match_len;
	}
	set_forward_pivot(raux,    next_pivot );
}


void Learned_getSMEMsOnePosOneThread_step1(Learned_index_aux_t* iaux, Learned_read_aux_t* raux, mem_tlv* smems, u64v* hits, bool hasN, bool use_cached=false) {
	/*
		Perform extension in zigzag style
	*/
	uint64_t key;
    uint64_t err ;
    uint64_t position ;
	uint64_t ss_position ;
	uint32_t exact_match_len ;
	uint32_t ss_exact_match_len ;
	uint32_t next_pivot ;
	int64_t suffix_array_num = iaux->bns->l_pac*2;
	uint32_t ambiguous_pos;
	// Right extension from pivot, save next raux->pivot
	bool right_forward= true;
#if MEM_TRADEOFF
	bool no_search ; // when short read has full-match in reference, reuse position without search
#endif
	// Check pivot point whether ambiguous base appears
	if (raux->unpacked_queue_buf[raux->pivot] >= 4){
		if (raux->l_seq - raux->pivot < raux->min_seed_len ){
			set_forward_pivot(raux, raux->l_seq );
			// set_forward_pivot(raux, raux->pivot+1 );
		}
		else{
			set_forward_pivot(raux, raux->pivot+1 );
		}
		return;
	}
	if (raux->pivot !=0 && raux->unpacked_queue_buf[raux->pivot-1] < 4){
		// - 2. Infer Learned index and get prediction
		next_pivot = raux->l_seq;
		int search_pivot = raux->pivot;
		while (search_pivot < next_pivot){
			if (raux->unpacked_queue_buf[search_pivot] >= 4){
				if (raux->l_seq - search_pivot < raux->min_seed_len ){
					set_forward_pivot(raux, raux->l_seq );
					search_pivot = raux->l_seq;
					// set_forward_pivot(raux, raux->pivot+1 );
				}
				else{
					search_pivot += 1; 
					set_forward_pivot(raux, raux->pivot+1 );
				}
				continue;
			}
			// Left extension from pivot, raux->pivot should be updated at every extension direction change
#if MEM_TRADEOFF
			right_forward = false;
			// Make uint64_t Key starting from pivot point
			key = Tokenization(raux, right_forward, &ambiguous_pos, hasN);
			if (raux ->max_l_seq == raux->l_seq){
				ss_position =  *(uint32_t*)(iaux->ref2sa+(suffix_array_num - raux->max_refpos - raux->pivot-1)*5);
				ss_position = ss_position <<8 | iaux->ref2sa[(suffix_array_num - raux->max_refpos - raux->pivot-1)*5+4];
	#if PREFETCH
				_mm_prefetch(iaux->sa_pos+ (ss_position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
				no_search= true;
				ss_position = mem_search_tradeoff(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
									 raux, ss_position, right_forward, &ss_exact_match_len,&ambiguous_pos,no_search);
			}
			else{
				if (use_cached && (raux->pivot< raux->cache_pivot_end) && ( raux->cache_pivot +MEM_TRADEOFF_USECACHE_THRESHOLD <= raux->pivot) ){
	#if DEBUG_MODE
					assert(raux->pivot> raux->cache_pivot);
	#endif
					ss_position =  *(uint32_t*)(iaux->ref2sa+(suffix_array_num + raux->cache_pivot - raux->cache_refpos - raux->pivot-1)*5);
					ss_position = ss_position <<8 | iaux->ref2sa[(suffix_array_num + raux->cache_pivot - raux->cache_refpos - raux->pivot-1)*5+4];
	#if PREFETCH
					_mm_prefetch(iaux->sa_pos+ (ss_position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
					no_search=false;
					ss_position = mem_search_tradeoff(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
									 raux, ss_position, right_forward, &ss_exact_match_len,&ambiguous_pos,no_search);
				}else{
					ss_position = learned_index_lookup(key ,&err);
	#if PREFETCH
					_mm_prefetch(iaux->sa_pos+ (ss_position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
					ss_position = mem_search(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
									 raux, ss_position, err, right_forward, &ss_exact_match_len,&ambiguous_pos);
				}
			}
			set_forward_pivot(raux,  raux->pivot - ss_exact_match_len+1 );
			if(next_pivot - raux->pivot < raux->min_seed_len ){
				break;
			}
			right_forward = true;
			key = Tokenization(raux, right_forward, &ambiguous_pos, hasN);
			if (raux ->max_l_seq == raux->l_seq){
	
				ss_position =  *(uint32_t*)(iaux->ref2sa+(raux->max_refpos + raux->pivot)*5);
				ss_position = ss_position <<8 | iaux->ref2sa[(raux->max_refpos + raux->pivot)*5+4];
	#if PREFETCH
				_mm_prefetch(iaux->sa_pos+ (ss_position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
				no_search= true;
				ss_position = right_smem_search_tradeoff(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
									 raux, ss_position, &ss_exact_match_len, smems, hits, &ambiguous_pos,no_search);	
				

			}else{
				if (use_cached && (raux->cache_pivot_end>= (raux->pivot+MEM_TRADEOFF_USECACHE_THRESHOLD)) && (raux->pivot>= raux->cache_pivot) ){
	#if DEBUG_MODE
					assert(raux->pivot>= raux->cache_pivot);
	#endif
					ss_position =  *(uint32_t*)(iaux->ref2sa+(raux->cache_refpos + raux->pivot - raux->cache_pivot)*5);
					ss_position = ss_position <<8 | iaux->ref2sa[(raux->cache_refpos + raux->pivot - raux->cache_pivot)*5+4];
	#if PREFETCH
					_mm_prefetch(iaux->sa_pos+ (ss_position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
					no_search=false;
					ss_position = right_smem_search_tradeoff(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
										raux, ss_position, &ss_exact_match_len, smems, hits, &ambiguous_pos,no_search);
				}else{
					ss_position = learned_index_lookup(key ,&err);
	#if PREFETCH
					_mm_prefetch(iaux->sa_pos+ (ss_position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
					ss_position = right_smem_search(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
										raux, ss_position, err, &ss_exact_match_len, smems, hits, &ambiguous_pos);
	
				}
			}
#else
			right_forward = false;
			key = Tokenization(raux, right_forward, &ambiguous_pos, hasN);
			ss_position = learned_index_lookup(key ,&err);
	#if PREFETCH
			_mm_prefetch(iaux->sa_pos+ (ss_position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
			ss_position = mem_search(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
							raux, ss_position, err, right_forward, &ss_exact_match_len,&ambiguous_pos);
			set_forward_pivot(raux,  raux->pivot - ss_exact_match_len+1 );

			if(next_pivot - raux->pivot < raux->min_seed_len ){
				break;
			}

			right_forward = true;

			key = Tokenization(raux, right_forward, &ambiguous_pos, hasN);
			ss_position = learned_index_lookup(key ,&err);
	#if PREFETCH
			_mm_prefetch(iaux->sa_pos+ (ss_position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
			ss_position = right_smem_search(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
									 raux, ss_position, err, &ss_exact_match_len, smems, hits, &ambiguous_pos);
			
			
#endif

#if DEBUG_MODE
			assert( (raux->pivot+ss_exact_match_len) > search_pivot);
#endif
			search_pivot = raux->pivot + ss_exact_match_len ;
			
			set_forward_pivot(raux,   search_pivot );
		}
	}
	else{
		key = Tokenization(raux, right_forward, &ambiguous_pos, hasN);
		// - 2. Infer Learned index and get prediction		
#if MEM_TRADEOFF
		position = learned_index_lookup(key ,&err);
	#if PREFETCH
		_mm_prefetch(iaux->sa_pos+ (position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif	
		position = right_smem_search(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
									raux, position, err, &exact_match_len,smems,hits,&ambiguous_pos);
	#if MEM_TRADEOFF_CACHED && PREFETCH
		if (exact_match_len>= raux->min_seed_len){

			uint64_t pos_val = *(uint32_t*)(iaux->sa_pos + position*SASIZE );
			pos_val = pos_val <<8 | iaux->sa_pos[position*SASIZE + 4];
			_mm_prefetch( iaux->ref2sa+ pos_val*5, _MM_HINT_T1 );
			_mm_prefetch( iaux->ref2sa+ (suffix_array_num - pos_val - exact_match_len-1)*5, _MM_HINT_T1 );
		}
	#endif
		if (exact_match_len == raux->l_seq){
			raux->max_l_seq = exact_match_len;
			uint64_t pos_val = *(uint32_t*)(iaux->sa_pos + position*SASIZE );
			pos_val = pos_val <<8 | iaux->sa_pos[position*SASIZE + 4];
			raux->max_refpos = pos_val;
			raux->max_pivot = 0;
	#if PREFETCH
			_mm_prefetch( iaux->ref2sa+ raux->max_refpos*5, _MM_HINT_T1 );
			_mm_prefetch( iaux->ref2sa+ (suffix_array_num - raux->max_refpos - raux->l_seq - 1)*5, _MM_HINT_T1 );
	#endif
		}
#else
		position = learned_index_lookup(key ,&err);
	#if PREFETCH
		_mm_prefetch(iaux->sa_pos+ (position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
		position = right_smem_search(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
									 raux, position, err, &exact_match_len,smems,hits,&ambiguous_pos);
		// position = mem_search(iaux->ref_string, iaux->sa_pos,iaux->pac, suffix_array_num, 
		// 							 raux, position, err, right_forward, &exact_match_len,&ambiguous_pos);
#endif	
		next_pivot = raux->pivot + exact_match_len;
	}
	set_forward_pivot(raux,    next_pivot );
}


void Learned_getSMEMsOnePosOneThread(Learned_index_aux_t* iaux, Learned_read_aux_t* raux, mem_tlv* smems, u64v* hits, bool hasN, bool use_cached=false) {
	/*
		Perform extension in zigzag style
	*/
	uint64_t key;
    uint64_t err ;
    uint64_t position ;
	uint64_t ss_position ;
	uint32_t exact_match_len ;
	uint32_t ss_exact_match_len ;
	uint32_t next_pivot ;
	int64_t suffix_array_num = iaux->bns->l_pac*2;
	uint32_t ambiguous_pos;
	// Right extension from pivot, save next raux->pivot
	
	bool right_forward= true;
#if MEM_TRADEOFF
	bool no_search ;
#endif
	// Check pivot point whether ambiguous base appears
	if (raux->unpacked_queue_buf[raux->pivot] >= 4){
		if (raux->l_seq - raux->pivot < raux->min_seed_len ){
			// if the length is shorter than min_seed_len no need to search
			set_forward_pivot(raux, raux->l_seq );
			// set_forward_pivot(raux, raux->pivot+1 );
		}
		else{
			set_forward_pivot(raux, raux->pivot+1 );
		}
		return;
	}
	// Make uint64_t Key starting from pivot point
	key = Tokenization(raux, right_forward, &ambiguous_pos, hasN);
	if (raux->pivot !=0 && raux->unpacked_queue_buf[raux->pivot-1] < 4){
		// - 2. Infer Learned index and get prediction
#if MEM_TRADEOFF
		if (raux ->max_l_seq == raux->l_seq){
			position =  *(uint32_t*)(iaux->ref2sa+(raux->max_refpos+raux->pivot)*5);
			position = position <<8 | iaux->ref2sa[(raux->max_refpos+raux->pivot)*5+4];
	#if PREFETCH
			_mm_prefetch(iaux->sa_pos+ (position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
			no_search= true;
			position = mem_search_tradeoff(iaux->ref_string, iaux->sa_pos,iaux->pac, suffix_array_num, 
									 raux, position, right_forward, &exact_match_len,&ambiguous_pos, no_search);	
		}else{
			if (use_cached &&  (raux->cache_pivot_end>= (raux->pivot+MEM_TRADEOFF_USECACHE_THRESHOLD))  && (raux->pivot>= raux->cache_pivot)){
				position =  *(uint32_t*)(iaux->ref2sa+(raux->cache_refpos + raux->pivot - raux->cache_pivot)*5);
				position = position <<8 | iaux->ref2sa[(raux->cache_refpos + raux->pivot - raux->cache_pivot)*5+4];
				no_search=false;
				position = mem_search_tradeoff(iaux->ref_string, iaux->sa_pos,iaux->pac, suffix_array_num, 
									 raux, position,  right_forward, &exact_match_len,&ambiguous_pos, no_search);	
			}else{
				position = learned_index_lookup(key ,&err);
	#if PREFETCH
				_mm_prefetch(iaux->sa_pos+ (position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
				position = mem_search(iaux->ref_string, iaux->sa_pos,iaux->pac, suffix_array_num, 
									 raux, position, err, right_forward, &exact_match_len,&ambiguous_pos);	
			}
			
		}
#else
		position = learned_index_lookup(key ,&err);
	#if PREFETCH
		_mm_prefetch(iaux->sa_pos+ (position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
		position = mem_search(iaux->ref_string, iaux->sa_pos,iaux->pac, suffix_array_num, 
									 raux, position, err, right_forward, &exact_match_len,&ambiguous_pos);
#endif								
		next_pivot = raux->pivot + exact_match_len;
		int search_pivot = raux->pivot;
		while (search_pivot < next_pivot){
			// Left extension from pivot, raux->pivot should be updated at every extension direction change
#if MEM_TRADEOFF
			right_forward = false;
			key = Tokenization(raux, right_forward, &ambiguous_pos, hasN);
			if (raux ->max_l_seq == raux->l_seq){
				ss_position =  *(uint32_t*)(iaux->ref2sa+(suffix_array_num - raux->max_refpos - raux->pivot-1)*5);
				ss_position = ss_position <<8 | iaux->ref2sa[(suffix_array_num - raux->max_refpos - raux->pivot-1)*5+4];
	#if PREFETCH
				_mm_prefetch(iaux->sa_pos+ (ss_position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
				no_search= true;
				ss_position = mem_search_tradeoff(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
									 raux, ss_position, right_forward, &ss_exact_match_len,&ambiguous_pos,no_search);
			}
			else{
				if (use_cached && (raux->pivot< raux->cache_pivot_end) && ( raux->cache_pivot +MEM_TRADEOFF_USECACHE_THRESHOLD <= raux->pivot) ){
	#if DEBUG_MODE
					assert(raux->pivot> raux->cache_pivot);
	#endif
					ss_position =  *(uint32_t*)(iaux->ref2sa+(suffix_array_num + raux->cache_pivot - raux->cache_refpos - raux->pivot-1)*5);
					ss_position = ss_position <<8 | iaux->ref2sa[(suffix_array_num + raux->cache_pivot - raux->cache_refpos - raux->pivot-1)*5+4];
	#if PREFETCH
					_mm_prefetch(iaux->sa_pos+ (ss_position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
					no_search=false;
					ss_position = mem_search_tradeoff(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
									 raux, ss_position, right_forward, &ss_exact_match_len,&ambiguous_pos,no_search);
				}else{
					ss_position = learned_index_lookup(key ,&err);
	#if PREFETCH
					_mm_prefetch(iaux->sa_pos+ (ss_position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
					ss_position = mem_search(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
									 raux, ss_position, err, right_forward, &ss_exact_match_len,&ambiguous_pos);
				}
			}
			set_forward_pivot(raux,  raux->pivot - ss_exact_match_len+1 );
			if(next_pivot - raux->pivot < raux->min_seed_len ){
				break;
			}
			right_forward = true;
			key = Tokenization(raux, right_forward, &ambiguous_pos, hasN);
			if (raux ->max_l_seq == raux->l_seq){
	
				ss_position =  *(uint32_t*)(iaux->ref2sa+(raux->max_refpos + raux->pivot)*5);
				ss_position = ss_position <<8 | iaux->ref2sa[(raux->max_refpos + raux->pivot)*5+4];
	#if PREFETCH
				_mm_prefetch(iaux->sa_pos+ (ss_position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
				no_search= true;
				ss_position = right_smem_search_tradeoff(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
									 raux, ss_position, &ss_exact_match_len, smems, hits, &ambiguous_pos,no_search);	
				

			}else{
				if (use_cached && (raux->cache_pivot_end>= (raux->pivot+MEM_TRADEOFF_USECACHE_THRESHOLD)) && (raux->pivot>= raux->cache_pivot) ){
	#if DEBUG_MODE
					assert(raux->pivot>= raux->cache_pivot);
	#endif
					ss_position =  *(uint32_t*)(iaux->ref2sa+(raux->cache_refpos + raux->pivot - raux->cache_pivot)*5);
					ss_position = ss_position <<8 | iaux->ref2sa[(raux->cache_refpos + raux->pivot - raux->cache_pivot)*5+4];
	#if PREFETCH
					_mm_prefetch(iaux->sa_pos+ (ss_position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
					no_search=false;
					ss_position = right_smem_search_tradeoff(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
										raux, ss_position, &ss_exact_match_len, smems, hits, &ambiguous_pos,no_search);
				}else{
					ss_position = learned_index_lookup(key ,&err);
	#if PREFETCH
					_mm_prefetch(iaux->sa_pos+ (ss_position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
					ss_position = right_smem_search(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
										raux, ss_position, err, &ss_exact_match_len, smems, hits, &ambiguous_pos);
	// #if MEM_TRADEOFF_CACHED && PREFETCH
	// 				if (ss_exact_match_len>= raux->min_seed_len){
	// 					_mm_prefetch( iaux->ref2sa+ iaux->sa_pos[ss_position<<1], _MM_HINT_T1 );
	// 					_mm_prefetch( iaux->ref2sa+ suffix_array_num - iaux->sa_pos[ss_position<<1] - ss_exact_match_len-1, _MM_HINT_T1 );
	// 				}
	// #endif
				}
			}
#else
			right_forward = false;
			key = Tokenization(raux, right_forward, &ambiguous_pos, hasN);
			ss_position = learned_index_lookup(key ,&err);
	#if PREFETCH
			_mm_prefetch(iaux->sa_pos+ (ss_position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
			ss_position = mem_search(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
							raux, ss_position, err, right_forward, &ss_exact_match_len,&ambiguous_pos);
			set_forward_pivot(raux,  raux->pivot - ss_exact_match_len+1 );

			if(next_pivot - raux->pivot < raux->min_seed_len ){
				break;
			}

			right_forward = true;

			key = Tokenization(raux, right_forward, &ambiguous_pos, hasN);
			ss_position = learned_index_lookup(key ,&err);
	#if PREFETCH
			_mm_prefetch(iaux->sa_pos+ (ss_position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
			ss_position = right_smem_search(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
									 raux, ss_position, err, &ss_exact_match_len, smems, hits, &ambiguous_pos);
#endif

#if DEBUG_MODE
			assert( (raux->pivot+ss_exact_match_len) > search_pivot);
#endif
			search_pivot = raux->pivot + ss_exact_match_len ;
			
			set_forward_pivot(raux,   search_pivot );
		}
	}
	else{
		// - 2. Infer Learned index and get prediction		
#if MEM_TRADEOFF
		position = learned_index_lookup(key ,&err);
	#if PREFETCH
		_mm_prefetch(iaux->sa_pos+ (position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif	
		position = right_smem_search(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
									raux, position, err, &exact_match_len,smems,hits,&ambiguous_pos);
	#if MEM_TRADEOFF_CACHED && PREFETCH
		if (exact_match_len>= raux->min_seed_len){

			uint64_t pos_val = *(uint32_t*)(iaux->sa_pos + position*SASIZE );
			pos_val = pos_val <<8 | iaux->sa_pos[position*SASIZE + 4];
			_mm_prefetch( iaux->ref2sa+ pos_val*5, _MM_HINT_T1 );
			_mm_prefetch( iaux->ref2sa+ (suffix_array_num - pos_val - exact_match_len-1)*5, _MM_HINT_T1 );
		}
	#endif
		if (exact_match_len == raux->l_seq){
			raux->max_l_seq = exact_match_len;
			uint64_t pos_val = *(uint32_t*)(iaux->sa_pos + position*SASIZE );
			pos_val = pos_val <<8 | iaux->sa_pos[position*SASIZE + 4];
			raux->max_refpos = pos_val;
			raux->max_pivot = 0;
	#if PREFETCH
			_mm_prefetch( iaux->ref2sa+ raux->max_refpos*5, _MM_HINT_T1 );
			_mm_prefetch( iaux->ref2sa+ (suffix_array_num - raux->max_refpos - raux->l_seq - 1)*5, _MM_HINT_T1 );
	#endif
		}
#else
		position = learned_index_lookup(key ,&err);
	#if PREFETCH
		_mm_prefetch(iaux->sa_pos+ (position*SASIZE)-SASIZE, _MM_HINT_T0);
	#endif
		position = right_smem_search(iaux->ref_string, iaux->sa_pos, iaux->pac, suffix_array_num, 
									 raux, position, err, &exact_match_len,smems,hits,&ambiguous_pos);
#endif	
		next_pivot = raux->pivot + exact_match_len;
	}
	set_forward_pivot(raux,    next_pivot );
}




uint64_t right_smem_search(const uint8_t* ref_string,const uint8_t* sa_pos,const uint8_t* pac, uint64_t sa_num,
                   Learned_read_aux_t* raux,  const uint64_t est_pos,
                    uint64_t enc_err,  uint32_t* exact_match_len, mem_tlv* smems, u64v* hits, uint32_t* ambiguous_pos) {
    /*
		Find maximal exact match and save position in iter_pos
		if min_intv_value is bigger than 1, search for upper and lower bound where it satisfies min_intv_value
		Save seeds in smems if satisfies min_seed_len and min_intv_value
	*/
	int min_intv_value = raux->min_intv_limit ;
	uint64_t iter_pos = est_pos;
	uint64_t read_valid_len = *ambiguous_pos - raux->pivot;
	bool exact_match_flag=false;
	uint32_t match_len;
	uint32_t last_match_len;
	uint64_t curr_err= (enc_err>>32) & 0x3fffffff;
	uint64_t err=(enc_err<<32)>>32 & 0x7fffffff;
	// if (curr_err == err){
	// 	curr_err = err - 1;
	// }
#if Count_mem_ref
	uint32_t count_search_bs=0;
	uint32_t count_search_linear=0;
	uint32_t count_search_min_intv=0;
	uint32_t count_smem=0;
#endif
	
    #if CURR_SEARCH_METHOD == 2
		uint64_t exp_search_move=MEM_TRADEOFF_USECACHE_EXP_SEARCH_START;
		// do exponential search
		uint64_t upper_b,lower_b,n,middle, half;
		iter_pos = iter_pos > MEM_TRADEOFF_USECACHE_EXP_SEARCH_START? iter_pos:MEM_TRADEOFF_USECACHE_EXP_SEARCH_START;
		iter_pos = iter_pos < sa_num-1 - MEM_TRADEOFF_USECACHE_EXP_SEARCH_START? iter_pos:sa_num - 1-MEM_TRADEOFF_USECACHE_EXP_SEARCH_START;
		
#if Count_mem_ref
		// count_search_exp++;
		count_search_bs++;
#endif	
		// std::cout <<"[smemtradeoff] Exp search start\n";
		// std::cout <<"iter_pos:"<<iter_pos<<"\n";
		if (!compare_read_and_ref_binary(pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)){
			if (exact_match_flag){
				lower_b=iter_pos;
				n=1;
			}else{
				iter_pos -= exp_search_move;
	#if PREFETCH
				_mm_prefetch(sa_pos+ ((iter_pos - exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
	#endif
	#if Count_mem_ref
				// count_search_exp++;
				count_search_bs++;
	#endif
				while (!compare_read_and_ref_binary(pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
					if (iter_pos ==0){
						// smaller than iter_pos 0 -> search_start_pos should be 0
						break;
					}
					exp_search_move *=MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW;
					if (iter_pos > exp_search_move){
						
						iter_pos -= exp_search_move;
	#if PREFETCH
						_mm_prefetch(sa_pos+ ((iter_pos - exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
	#endif								
					}else{
						exp_search_move=iter_pos;
						iter_pos = 0;
					}
	#if Count_mem_ref
					// count_search_exp++;
					count_search_bs++;
	#endif
					
				}
				if (exact_match_flag){
					lower_b=iter_pos;
					n=1;
				}else{
					upper_b = iter_pos + exp_search_move>= sa_num-1? sa_num-2: iter_pos + exp_search_move;
					lower_b = iter_pos >0? iter_pos:1;
					n=  upper_b-lower_b + 1;
				}
				
			}
		}
		else{
			if (exact_match_flag){
				lower_b=iter_pos;
				n=1;
			}else{
				//  estimated position have smaller key than read, increase iter_pos and compare
				iter_pos += exp_search_move;
	#if PREFETCH
				_mm_prefetch(sa_pos+ ((iter_pos + exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
	#endif				
	#if Count_mem_ref
				// count_search_exp++;
				count_search_bs++;
	#endif
				while (compare_read_and_ref_binary(pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
					if (iter_pos == sa_num - 1){
						break;
					}
					exp_search_move *=MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW;
					if (sa_num - 1 > iter_pos+exp_search_move){
						iter_pos += exp_search_move;
	#if PREFETCH
						_mm_prefetch(sa_pos+ ((iter_pos + exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
	#endif						
					}else{
						exp_search_move=sa_num - iter_pos -1;
						iter_pos = sa_num - 1;
					}
	#if Count_mem_ref
					// count_search_exp++;
					count_search_bs++;
	#endif
				}

				if (exact_match_flag){
					lower_b=iter_pos;
					n=1;
				}else{
					upper_b = iter_pos >= sa_num -1? sa_num-2: iter_pos ;
					lower_b = iter_pos > exp_search_move ? iter_pos- exp_search_move:1;
					n=  upper_b-lower_b + 1;
				}
			}
		}
		middle=iter_pos;
	#elif CURR_SEARCH_METHOD ==1
	uint64_t upper_b = iter_pos+err >= sa_num-1? sa_num-2: iter_pos+err;
	uint64_t lower_b = iter_pos > curr_err? iter_pos-curr_err:1;
	lower_b = lower_b >= upper_b ? upper_b - 1: lower_b;
	// std::cout << "[rightsmem]iterpos:"<<iter_pos<<"Upper b:" << upper_b <<" Lower b:" << lower_b <<"\n";
	uint64_t n = upper_b-lower_b + 1; // `end` is inclusive.
	uint64_t middle=iter_pos;
	#else	//old binary search method
	err = std::max(curr_err,err);
	uint64_t upper_b = iter_pos+err >= sa_num-1? sa_num-2: iter_pos+err;
	uint64_t lower_b = iter_pos > err? iter_pos-err:1;
	uint64_t n = upper_b-lower_b + 1; // `end` is inclusive.
	uint64_t middle=iter_pos;
	#endif
	// Function adapted from https://github.com/gvinciguerra/rmi_pgm/blob/357acf668c22f927660d6ed11a15408f722ea348/main.cpp#L29.
	// Authored by Giorgio Vinciguerra.

	while (uint64_t half = (n>>1)) {
		middle = lower_b + half;
	#if Count_mem_ref
			count_search_bs++;
	#endif
	#if PREFETCH
		_mm_prefetch(sa_pos+(middle+half/2)*SASIZE, _MM_HINT_T0);
		_mm_prefetch(sa_pos+((lower_b+half/2)*SASIZE), _MM_HINT_T0);
	#endif
		lower_b = compare_read_and_ref_binary(pac, sa_pos, middle , raux, sa_num,read_valid_len, &match_len,&exact_match_flag)? middle: lower_b;
	
			
		if(exact_match_flag){
			break;
		}
		n -= half;
	}
// #if PREFETCH
// 		_mm_prefetch(sa_pos+((lower_b-16)<<1), _MM_HINT_T0);		
// #endif
	if (exact_match_flag){
		iter_pos = lower_b;

	}else{
		// no need to do additional linear search
		if(middle!=lower_b){
			// lookup key was smaller than middle, should check lower_b
			last_match_len = match_len;
			iter_pos = lower_b;
		
	#if Count_mem_ref
			count_search_linear++;
	#endif
			while (!compare_read_and_ref_binary(pac, sa_pos, iter_pos , raux, sa_num, read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
				if (iter_pos ==0){
					break;
				} 
				#if PREFETCH
				if (g_use_simd8_mode) {
					_mm_prefetch(sa_pos + (iter_pos - 1) * SASIZE, _MM_HINT_T0);
				}
				#endif
				iter_pos --;
				
	#if Count_mem_ref
				count_search_linear++;
	#endif
				

				last_match_len = match_len;
			}
			//  stops when ref is smaller than read, iter_pos should point to exact matching or bigger ref pos
			if (last_match_len > match_len){
				iter_pos++;
				match_len = last_match_len;
			}
		}
		else{
			// lookup key was bigger than middle, should check lower_b+1
			last_match_len = match_len;
			//  estimated position have smaller key than read, increase iter_pos and compare
			iter_pos =lower_b+1;
			iter_pos = iter_pos >= sa_num ? sa_num-1 : iter_pos;
	#if Count_mem_ref
			count_search_linear++;
	#endif

			while (compare_read_and_ref_binary(pac, sa_pos,iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
				if (iter_pos == sa_num - 1){
					break;
				}
				#if PREFETCH
				if (g_use_simd8_mode) {
					_mm_prefetch(sa_pos + (iter_pos + 1) * SASIZE, _MM_HINT_T0);
				}
				#endif
				iter_pos ++;
	#if Count_mem_ref
				count_search_linear++;
	#endif
				

				last_match_len = match_len;
			}
			//  stops when ref is bigger than read or exact matches with read,
			if (last_match_len > match_len){
				iter_pos--;
				match_len = last_match_len;
			}
		}
	}
	//  iter_pos points to best exact matching position, match_len have the number of exact match
	last_match_len = match_len;
	uint64_t search_start_pos = iter_pos;
	uint32_t up_match=match_len, low_match=match_len, match_num=1;
	// when need to find smems, should count number of matches
		

#if EXPONENTIAL_SMEMSEARCH
		uint64_t up=0, low=0;
		while(1){
			while ( up+low < 15){
					if ( match_len < raux->min_seed_len && (up+low+ (up==0)+(low==0)-1 ) >= min_intv_value){
						#if Count_mem_ref
						fprintf(stdout,"[smem_search func - right e-lin]\tMax match ref len:%d Count Total: %d Count_bs:%d Count_linear:%d Count_minintv:%d input minintv:%d\n",match_len,count_search_bs+count_search_linear+count_search_min_intv, count_search_bs, count_search_linear, count_search_min_intv, min_intv_value);
						#endif
						*exact_match_len = match_len;
						return iter_pos;
					}
					if (low_match >= match_len && low+1 <= search_start_pos){
						#if Count_mem_ref
						count_search_min_intv++;
						#endif
						#if PREFETCH
						if (g_use_simd8_mode) {
							_mm_prefetch(sa_pos + (search_start_pos - low - 1) * SASIZE, _MM_HINT_T0);
						}
						#endif
						low++;
						compare_read_and_ref_binary(pac, sa_pos, (search_start_pos-low), raux, sa_num,match_len, &low_match,&exact_match_flag);

						// if (low_match < match_len){
						// 	low--;
						// }

					}
					else if(up_match >= match_len &&  sa_num-2 >= up + search_start_pos){
						#if Count_mem_ref
						count_search_min_intv++;
						#endif			
						#if PREFETCH
						if (g_use_simd8_mode) {
							_mm_prefetch(sa_pos + (search_start_pos + up + 1) * SASIZE, _MM_HINT_T0);
						}
						#endif
						up++;
						compare_read_and_ref_binary(pac, sa_pos, (search_start_pos+up), raux, sa_num,match_len, &up_match,&exact_match_flag);
						
						// if (up_match < match_len){
						// 	up--;
						// }
					}
					else{
						break;
					}

					if ( match_len < raux->min_seed_len && (up+low+(up==0)-1) >= min_intv_value){
						#if Count_mem_ref
						fprintf(stdout,"[smem_search func - right e-lin]\tMax match ref len:%d Count Total: %d Count_bs:%d Count_linear:%d Count_minintv:%d input minintv:%d\n",match_len,count_search_bs+count_search_linear+count_search_min_intv, count_search_bs, count_search_linear, count_search_min_intv, min_intv_value);
						#endif
						*exact_match_len = match_len;
						return iter_pos;
					}
			}
			

			#if CURR_SEARCH_METHOD != 2
			uint64_t exp_search_move = EXPONENTIAL_EXP_START;
			#else
			exp_search_move = EXPONENTIAL_EXP_START;
			#endif
			
			if (low_match >= match_len && low < search_start_pos){
				//exponential search
				upper_b = search_start_pos-low;
				exp_search_move = std::min(search_start_pos-low, exp_search_move);
				lower_b = search_start_pos-low-exp_search_move;
				n=  upper_b-lower_b + 1;
				// printf("[Low-exp] Up_b:%lld Low_b:%lld\n",upper_b, lower_b);
				while (low_match >= match_len){
					#if Count_mem_ref
					count_search_min_intv++;
					#endif
					compare_read_and_ref_binary(pac, sa_pos, lower_b, raux, sa_num,match_len, &low_match,&exact_match_flag);
					if (low_match < match_len ){
						// found different position
						break;
					}
					if (lower_b == 0){
						break;
					}
					exp_search_move <<= EXPONENTIAL_EXP_POW;
					exp_search_move = std::min(exp_search_move,lower_b);
					lower_b -= exp_search_move;
					
					upper_b = lower_b + exp_search_move;
					n=  upper_b-lower_b + 1;
				}
				// printf("[Low]Upper_b: %lld Lower_b:%lld n:%lld\n",upper_b,lower_b,n);
				//binary search 
				while (uint64_t half = (n>>1)) {
					if (match_len < raux->min_seed_len &&  up+search_start_pos-upper_b >= min_intv_value){
						#if Count_mem_ref
						fprintf(stdout,"[smem_search func - right e-exp]\tMax match ref len:%d Count Total: %d Count_bs:%d Count_linear:%d Count_minintv:%d input minintv:%d\n",match_len,count_search_bs+count_search_linear+count_search_min_intv, count_search_bs, count_search_linear, count_search_min_intv, min_intv_value);
						#endif
						*exact_match_len = match_len;
						return iter_pos;
					}
					middle = upper_b - half;
					// middle = lower_b + half;
					#if Count_mem_ref
						count_search_min_intv++;
					#endif
					#if PREFETCH
					_mm_prefetch(sa_pos+(middle-half/2)*SASIZE, _MM_HINT_T0);
					_mm_prefetch(sa_pos+((upper_b-half/2)*SASIZE), _MM_HINT_T0);
					#endif
					
					compare_read_and_ref_binary(pac, sa_pos, middle , raux, sa_num,match_len, &low_match,&exact_match_flag);
					// printf("[BIN]n:%lld middle: %lld lower_b: %lld match_len: %lld low_match: %lld\n",n, middle, lower_b, match_len, low_match);
					if (low_match >= match_len){
						upper_b = middle;
						if (upper_b !=0 && n<3){
							//update low_match to the next biggest exact match length
							#if Count_mem_ref
								count_search_min_intv++;
							#endif
							compare_read_and_ref_binary(pac, sa_pos, (upper_b-1) , raux, sa_num,match_len, &low_match,&exact_match_flag);
						}
					}
					n -= half;
				}
				low = search_start_pos+1 - upper_b ;
				if (match_len < raux->min_seed_len &&  up+low-1 >= min_intv_value){
					#if Count_mem_ref
						fprintf(stdout,"[smem_search func - right e-exp]\tMax match ref len:%d Count Total: %d Count_bs:%d Count_linear:%d Count_minintv:%d input minintv:%d\n",match_len,count_search_bs+count_search_linear+count_search_min_intv, count_search_bs, count_search_linear, count_search_min_intv, min_intv_value);
					#endif
					*exact_match_len = match_len;
					return iter_pos;
				}
			}

			if(up_match >= match_len && sa_num-1 > up + search_start_pos){
				// find Up
				exp_search_move = EXPONENTIAL_EXP_START;
				//exponential search
				lower_b = search_start_pos+up;
				exp_search_move = std::min(sa_num-1-lower_b, exp_search_move);
				upper_b = lower_b+exp_search_move;
				n=  upper_b-lower_b + 1;
				// printf("[Up-exp] Up_b:%lld Low_b:%lld\n",upper_b, lower_b);
				while (up_match >= match_len){
					#if Count_mem_ref
					count_search_min_intv++;
					#endif
					compare_read_and_ref_binary(pac, sa_pos, upper_b, raux, sa_num,match_len, &up_match,&exact_match_flag);
					if (up_match < match_len){
						// found different position
						break;
					}
					if (upper_b == sa_num-1){
						break;
					}
					exp_search_move <<= EXPONENTIAL_EXP_POW;
					exp_search_move = std::min(exp_search_move,sa_num-1-upper_b);
					upper_b += exp_search_move;
					lower_b = upper_b-exp_search_move;
					n=  upper_b-lower_b + 1;
				}
				// printf("[Up]Upper_b: %lld Lower_b:%lld n:%lld\n",upper_b,lower_b,n);
				//binary search 
				while (uint64_t half = (n>>1)) {
					if (match_len < raux->min_seed_len &&  lower_b-search_start_pos+low >= min_intv_value){
						#if Count_mem_ref
							fprintf(stdout,"[smem_search func - right e-exp]\tMax match ref len:%d Count Total: %d Count_bs:%d Count_linear:%d Count_minintv:%d input minintv:%d\n",match_len,count_search_bs+count_search_linear+count_search_min_intv, count_search_bs, count_search_linear, count_search_min_intv, min_intv_value);
						#endif
						*exact_match_len = match_len;
						return iter_pos;
					}
					middle = lower_b + half;
					#if Count_mem_ref
						count_search_min_intv++;
					#endif
					#if PREFETCH
					_mm_prefetch(sa_pos+(middle+half/2)*SASIZE, _MM_HINT_T0);
					_mm_prefetch(sa_pos+((lower_b+half/2)*SASIZE), _MM_HINT_T0);
					#endif
					compare_read_and_ref_binary(pac, sa_pos, middle, raux, sa_num,match_len, &up_match,&exact_match_flag);
					if (up_match >= match_len){
						lower_b = middle;
						if(lower_b != sa_num-1 && n<3){
							#if Count_mem_ref
								count_search_min_intv++;
							#endif
							compare_read_and_ref_binary(pac, sa_pos, (lower_b+1), raux, sa_num,match_len, &up_match,&exact_match_flag);
						}
					}
					
					n -= half;
				}
				up = lower_b+1 - search_start_pos;

				if (match_len < raux->min_seed_len &&  up+low-1 >= min_intv_value){
					#if Count_mem_ref
						fprintf(stdout,"[smem_search func - right e-exp]\tMax match ref len:%d Count Total: %d Count_bs:%d Count_linear:%d Count_minintv:%d input minintv:%d\n",match_len,count_search_bs+count_search_linear+count_search_min_intv, count_search_bs, count_search_linear, count_search_min_intv, min_intv_value);
					#endif
					*exact_match_len = match_len;
					return iter_pos;
				}

			}

			if (low_match >= match_len && low >= search_start_pos){
				low = search_start_pos+1;
				low_match = 0;
			}
			if (up_match >= match_len && ((up + search_start_pos) >= sa_num-1)){
				up = sa_num-search_start_pos;	
				up_match = 0;
			}
			
			match_num = up+low-1;
			iter_pos=search_start_pos-low+1;
			if (  match_num >= min_intv_value){
					break;
			}
			match_len = up_match > low_match? up_match: low_match;
		}//end while 1
#else
		uint64_t up=1, low=1;
		while(1){
			while (1){
				if (match_len < raux->min_seed_len &&  (up+low-3+(up==1)+(low==1)) >= min_intv_value){
	#if Count_mem_ref
					fprintf(stdout,"[smem_search func - right]\tMax match ref len:%d Count Total: %d Count_bs:%d Count_linear:%d Count_minintv:%d input minintv:%d\n",match_len,count_search_bs+count_search_linear+count_search_min_intv, count_search_bs, count_search_linear, count_search_min_intv, min_intv_value);
	#endif
					*exact_match_len = match_len;
					return iter_pos;
				}

				if (low_match >= match_len && low <= search_start_pos){
	#if Count_mem_ref
					count_search_min_intv++;
	#endif
					compare_read_and_ref_binary(pac, sa_pos, (search_start_pos-low), raux, sa_num,match_len, &low_match,&exact_match_flag);


					low++;
					
				}
				else if(up_match >= match_len &&  sa_num-1 >= up + search_start_pos){
	#if Count_mem_ref
					count_search_min_intv++;
	#endif					
					compare_read_and_ref_binary(pac, sa_pos, (search_start_pos+up), raux, sa_num,match_len, &up_match,&exact_match_flag);

					up++;
					
				}
				else{
					break;
				}
				if (match_len < raux->min_seed_len &&  (up+low-3+(up==1)) >= min_intv_value){
	#if Count_mem_ref
					fprintf(stdout,"[smem_search func - right]\tMax match ref len:%d Count Total: %d Count_bs:%d Count_linear:%d Count_minintv:%d input minintv:%d\n",match_len,count_search_bs+count_search_linear+count_search_min_intv, count_search_bs, count_search_linear, count_search_min_intv, min_intv_value);
	#endif
					*exact_match_len = match_len;
					return iter_pos;
				}
			}
			if (low_match == match_len && low > search_start_pos){
				// low+=1;
				low = search_start_pos+2;
				low_match = 0;
			}
			if (up_match == match_len && ((up + search_start_pos) > sa_num-1)){
				up = sa_num+1-search_start_pos;	
				up_match = 0;
			}
			match_num = up+low-3;//up+low+1-4;
			iter_pos=search_start_pos-low+2;
			if (  match_num >= min_intv_value){
					break;
			}
			match_len = up_match > low_match? up_match: low_match;
		}//end while 1
		// printf("[ORi]up: %lld low: %lld\n",up, low);
		// printf("[ORi]match_num: %lld match_len: %lld up_match: %lld low_match: %lld\n",match_num, match_len, up_match, low_match);
#endif

		*exact_match_len = match_len;
		// add found smems to smems and hits array
		if (match_len>= raux->min_seed_len){

			mem_tl mem;
			// memset_s(&mem, sizeof(mem_tl), 0);
			mem.start = raux->pivot;
			mem.end = raux->pivot + match_len;
#if MEM_TRADEOFF_CACHED
			mem.cache_refpos = *(uint32_t*)(sa_pos + iter_pos*SASIZE );
			mem.cache_refpos = mem.cache_refpos <<8 | sa_pos[iter_pos*SASIZE + 4];
#endif
			mem.hitbeg = hits->n; // begin index in hits vector
			mem.hitcount = match_num; // number of hits in reference
			for (uint64_t i=0; i< match_num; i++){
				uint64_t pos_val = *(uint32_t*)(sa_pos + (iter_pos+i)*SASIZE );
				pos_val = pos_val <<8 | sa_pos[(iter_pos+i)*SASIZE + 4];
				kv_push(uint64_t, *hits, pos_val );
			}
			kv_push(mem_tl, *smems, mem);
		}
#if Count_mem_ref
	// printf("Up:%lld Low:%lld \n", up,low);
	fprintf(stdout,"[smem_search func - right]\tMax match ref len:%d Count Total: %d Count_bs:%d Count_linear:%d Count_minintv:%d input minintv:%d\n",match_len,count_search_bs+count_search_linear+count_search_min_intv, count_search_bs, count_search_linear, count_search_min_intv, min_intv_value);
	// assert(0);
#endif
		return iter_pos;
}


uint64_t mem_search(const uint8_t* ref_string,const uint8_t* sa_pos,const uint8_t* pac, uint64_t sa_num,
                  Learned_read_aux_t* raux,  const uint64_t est_pos,
                    uint64_t enc_err, bool right_forward, uint32_t* exact_match_len, uint32_t* ambiguous_pos) {
    /*
		Find maximal exact match and save position in iter_pos
		if min_intv_value is bigger than 1, search for upper and lower bound where it satisfies min_intv_value
		Save seeds in smems if satisfies min_seed_len and min_intv_value
	*/

	int min_intv_value = raux->min_intv_limit ;
	uint64_t iter_pos = est_pos;
	uint32_t read_valid_len;
	bool exact_match_flag=false;
	uint32_t match_len;
	uint32_t last_match_len;
	uint64_t curr_err= (enc_err>>32) & 0x3fffffff;
	uint64_t err=(enc_err<<32)>>32 & 0x7fffffff;
	// if (curr_err == err){
	// 	curr_err = err - 1;
	// }
#if Count_mem_ref
	uint32_t count_search_bs=0;
	uint32_t count_search_linear=0;
	uint32_t count_search_min_intv=0;
	uint32_t count_smem=0;
#endif

	if (right_forward){
		read_valid_len = *ambiguous_pos - raux->pivot;

#if CURR_SEARCH_METHOD == 2
		uint64_t exp_search_move=MEM_TRADEOFF_USECACHE_EXP_SEARCH_START;
		// do exponential search
		uint64_t upper_b,lower_b,n,middle, half;
		iter_pos = iter_pos > MEM_TRADEOFF_USECACHE_EXP_SEARCH_START? iter_pos:MEM_TRADEOFF_USECACHE_EXP_SEARCH_START;
		iter_pos = iter_pos < sa_num-1 - MEM_TRADEOFF_USECACHE_EXP_SEARCH_START? iter_pos:sa_num - 1-MEM_TRADEOFF_USECACHE_EXP_SEARCH_START;
		
#if Count_mem_ref
		// count_search_exp++;
		count_search_bs++;
#endif	
		// std::cout <<"[smemtradeoff] Exp search start\n";
		// std::cout <<"iter_pos:"<<iter_pos<<"\n";
		if (!compare_read_and_ref_binary(pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)){
			if (exact_match_flag){
				lower_b=iter_pos;
				n=1;
			}else{
				iter_pos -= exp_search_move;
	#if PREFETCH
				_mm_prefetch(sa_pos+ ((iter_pos - exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
	#endif
	#if Count_mem_ref
				// count_search_exp++;
				count_search_bs++;
	#endif
				while (!compare_read_and_ref_binary(pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
					if (iter_pos ==0){
						// smaller than iter_pos 0 -> search_start_pos should be 0
						break;
					}
					exp_search_move *=MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW;
					if (iter_pos > exp_search_move){
						
						iter_pos -= exp_search_move;
	#if PREFETCH
						_mm_prefetch(sa_pos+ ((iter_pos - exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
	#endif								
					}else{
						exp_search_move=iter_pos;
						iter_pos = 0;
					}
	#if Count_mem_ref
					// count_search_exp++;
					count_search_bs++;
	#endif
					
				}
				if (exact_match_flag){
					lower_b=iter_pos;
					n=1;
				}else{
					upper_b = iter_pos + exp_search_move>= sa_num-1? sa_num-2: iter_pos + exp_search_move;
					lower_b = iter_pos >0? iter_pos:1;
					n=  upper_b-lower_b + 1;
				}
				
			}
		}
		else{
			if (exact_match_flag){
				lower_b=iter_pos;
				n=1;
			}else{
				//  estimated position have smaller key than read, increase iter_pos and compare
				iter_pos += exp_search_move;
	#if PREFETCH
				_mm_prefetch(sa_pos+ ((iter_pos + exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
	#endif				
	#if Count_mem_ref
				// count_search_exp++;
				count_search_bs++;
	#endif
				while (compare_read_and_ref_binary(pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
					if (iter_pos == sa_num - 1){
						break;
					}
					exp_search_move *=MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW;
					if (sa_num - 1 > iter_pos+exp_search_move){
						iter_pos += exp_search_move;
	#if PREFETCH
						_mm_prefetch(sa_pos+ ((iter_pos + exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
	#endif						
					}else{
						exp_search_move=sa_num - iter_pos -1;
						iter_pos = sa_num - 1;
					}
	#if Count_mem_ref
					// count_search_exp++;
					count_search_bs++;
	#endif
				}

				if (exact_match_flag){
					lower_b=iter_pos;
					n=1;
				}else{
					upper_b = iter_pos >= sa_num -1? sa_num-2: iter_pos ;
					lower_b = iter_pos > exp_search_move ? iter_pos- exp_search_move:1;
					n=  upper_b-lower_b + 1;
				}
			}
		}
		middle=iter_pos;
#elif CURR_SEARCH_METHOD ==1
		uint64_t upper_b = iter_pos+err >= sa_num-1? sa_num-2: iter_pos+err;
		uint64_t lower_b = iter_pos > curr_err? iter_pos-curr_err:1;
		lower_b = lower_b >= upper_b ? upper_b - 1: lower_b;
		// std::cout << "[rightsmem]iterpos:"<<iter_pos<<"Upper b:" << upper_b <<" Lower b:" << lower_b <<"\n";
		uint64_t n = upper_b-lower_b + 1; // `end` is inclusive.
		uint64_t middle=iter_pos;

		// Search for first occurrence of key.w
#else	//old binary search method
		err = std::max(curr_err,err);
		uint64_t upper_b = iter_pos+err >= sa_num-1? sa_num-2: iter_pos+err;
		uint64_t lower_b = iter_pos > err? iter_pos-err:1;
		uint64_t n = upper_b-lower_b + 1; // `end` is inclusive.
		uint64_t middle=iter_pos;
#endif
		// Function adapted from https://github.com/gvinciguerra/rmi_pgm/blob/357acf668c22f927660d6ed11a15408f722ea348/main.cpp#L29.
		// Authored by Giorgio Vinciguerra.
		while (uint64_t half = (n>>1)) {
			middle = lower_b + half;
	#if Count_mem_ref
			count_search_bs++;
	#endif
	#if PREFETCH
			_mm_prefetch(sa_pos+ ((middle+half/2)*SASIZE), _MM_HINT_T0);
			_mm_prefetch(sa_pos+((lower_b+half/2)*SASIZE), _MM_HINT_T0);
	#endif
			lower_b = compare_read_and_ref_binary(pac, sa_pos, middle, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)? middle: lower_b;
	
			if(exact_match_flag){
				break;
			}
			n -= half;
		}
	// #if PREFETCH
	// 	_mm_prefetch(sa_pos+((lower_b-16)<<1), _MM_HINT_T0);		
	// #endif
		if (exact_match_flag){
			iter_pos = lower_b;
		}else{
			// no need to do additional linear search
			if(middle!=lower_b){
				// lookup key was smaller than middle, should check lower_b
				last_match_len = match_len;
				iter_pos = lower_b;
	#if Count_mem_ref
				count_search_linear++;
	#endif
	
				while (!compare_read_and_ref_binary(pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
					if (iter_pos ==0){
						break;
					}
					#if PREFETCH
					if (g_use_simd8_mode) {
						_mm_prefetch(sa_pos + (iter_pos - 1) * SASIZE, _MM_HINT_T0);
					}
					#endif
					iter_pos --;
	
	#if Count_mem_ref
					count_search_linear++;
	#endif
					last_match_len = match_len;
				}
				//  stops when ref is smaller than read, iter_pos should point to exact matching or bigger ref pos
				if (last_match_len > match_len){
					iter_pos++;
					match_len = last_match_len;
				}
			}
			else{
				// lookup key was bigger than middle, should check lower_b+1
				last_match_len = match_len;
				//  estimated position have smaller key than read, increase iter_pos and compare
				iter_pos = lower_b+1;
				iter_pos = iter_pos >= sa_num ? sa_num-1 : iter_pos;

	#if Count_mem_ref
				count_search_linear++;
	#endif
		
				while (compare_read_and_ref_binary(pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
					if (iter_pos == sa_num - 1){
						break;
					}
					#if PREFETCH
					if (g_use_simd8_mode) {
						_mm_prefetch(sa_pos + (iter_pos + 1) * SASIZE, _MM_HINT_T0);
					}
					#endif
					iter_pos ++;
	#if Count_mem_ref
					count_search_linear++;
	#endif
					last_match_len = match_len;
				}
				//  stops when ref is bigger than read or exact matches with read,
				if (last_match_len > match_len){
					iter_pos--;
					match_len = last_match_len;
				}
			}
		}
	#if PREFETCH
		_mm_prefetch(sa_pos+((iter_pos-16)*SASIZE), _MM_HINT_T0);		
	#endif
		//  iter_pos points to best exact matching position, match_len have the number of exact match
		uint64_t search_start_pos = iter_pos;
		uint32_t up_match=match_len, low_match=match_len, match_num=1;
		// when need to find smems, should count number of matches
		if (1 != min_intv_value  ){
			uint64_t up=1, low=1;
			bool low_search_flag, up_search_flag;
			while(1){
				// while (low_match >= match_len || up_match >= match_len){
				while (1){
					low_search_flag = low_match >= match_len && low <= search_start_pos;
					up_search_flag = up_match >= match_len &&  sa_num-1 >= up + search_start_pos;
					if (low_search_flag){
#if Count_mem_ref
						count_search_min_intv++;
#endif
						compare_read_and_ref_binary(pac, sa_pos, (search_start_pos-low), raux, sa_num,match_len, &low_match,&exact_match_flag);
						#if PREFETCH
						if (g_use_simd8_mode) {
							_mm_prefetch(sa_pos + (search_start_pos - low - 1) * SASIZE, _MM_HINT_T0);
						}
						#endif
						low++;
					}
					if(up_search_flag){
#if Count_mem_ref
						count_search_min_intv++;
#endif	
						compare_read_and_ref_binary(pac, sa_pos, (search_start_pos+up), raux, sa_num,match_len, &up_match,&exact_match_flag);
						#if PREFETCH
						if (g_use_simd8_mode) {
							_mm_prefetch(sa_pos + (search_start_pos + up + 1) * SASIZE, _MM_HINT_T0);
						}
						#endif
						up++;
					}
					if (up+low-3 >= min_intv_value || (!low_search_flag && !up_search_flag)){
						break;
					}
				}
				if (low_match == match_len && low > search_start_pos){
					low+=1;
					low_match = 0;
				}
				if (up_match == match_len && ((up + search_start_pos) > sa_num-1)){
					up +=1;	
					up_match = 0;
				}
				match_num = up+low-3;
				if (  match_num >= min_intv_value){
						break;
				}
				match_len = up_match > low_match? up_match: low_match;
			}//end while 1
		}
		*exact_match_len = match_len;
#if Count_mem_ref
		fprintf(stdout,"[mem_search func - right]\tMax match ref len:%d Count Total: %d Count_bs:%d Count_linear:%d Count_minintv:%d input minintv:%d\n",match_len, count_search_bs+count_search_linear+count_search_min_intv,count_search_bs, count_search_linear, count_search_min_intv, min_intv_value);
#endif
		return iter_pos;
	}
	else{
		/************************************
		// left extension
		*************************************/
		read_valid_len = *ambiguous_pos - raux->l_pivot;
#if CURR_SEARCH_METHOD == 2
		uint64_t exp_search_move=MEM_TRADEOFF_USECACHE_EXP_SEARCH_START;
		// do exponential search
		uint64_t upper_b,lower_b,n,middle, half;
		iter_pos = iter_pos > MEM_TRADEOFF_USECACHE_EXP_SEARCH_START? iter_pos:MEM_TRADEOFF_USECACHE_EXP_SEARCH_START;
		iter_pos = iter_pos < sa_num-1 - MEM_TRADEOFF_USECACHE_EXP_SEARCH_START? iter_pos:sa_num - 1-MEM_TRADEOFF_USECACHE_EXP_SEARCH_START;
		
#if Count_mem_ref
		// count_search_exp++;
		count_search_bs++;
#endif	
		// std::cout <<"[smemtradeoff] Exp search start\n";
		// std::cout <<"iter_pos:"<<iter_pos<<"\n";
		if (!compare_read_and_ref_binary_left(pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)){
			if (exact_match_flag){
				lower_b=iter_pos;
				n=1;
			}else{
				iter_pos -= exp_search_move;
	#if PREFETCH
				_mm_prefetch(sa_pos+ ((iter_pos - exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
	#endif
	#if Count_mem_ref
				// count_search_exp++;
				count_search_bs++;
	#endif
				while (!compare_read_and_ref_binary_left(pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
					if (iter_pos ==0){
						// smaller than iter_pos 0 -> search_start_pos should be 0
						break;
					}
					exp_search_move *=MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW;
					if (iter_pos > exp_search_move){
						
						iter_pos -= exp_search_move;
	#if PREFETCH
						_mm_prefetch(sa_pos+ ((iter_pos - exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
	#endif								
					}else{
						exp_search_move=iter_pos;
						iter_pos = 0;
					}
	#if Count_mem_ref
					// count_search_exp++;
					count_search_bs++;
	#endif
					
				}
				if (exact_match_flag){
					lower_b=iter_pos;
					n=1;
				}else{
					upper_b = iter_pos + exp_search_move>= sa_num-1? sa_num-2: iter_pos + exp_search_move;
					lower_b = iter_pos >0? iter_pos:1;
					n=  upper_b-lower_b + 1;
				}
				
			}
		}
		else{
			if (exact_match_flag){
				lower_b=iter_pos;
				n=1;
			}else{
				//  estimated position have smaller key than read, increase iter_pos and compare
				iter_pos += exp_search_move;
	#if PREFETCH
				_mm_prefetch(sa_pos+ ((iter_pos + exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
	#endif				
	#if Count_mem_ref
				// count_search_exp++;
				count_search_bs++;
	#endif
				while (compare_read_and_ref_binary_left(pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
					if (iter_pos == sa_num - 1){
						break;
					}
					exp_search_move *=MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW;
					if (sa_num - 1 > iter_pos+exp_search_move){
						iter_pos += exp_search_move;
	#if PREFETCH
						_mm_prefetch(sa_pos+ ((iter_pos + exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
	#endif						
					}else{
						exp_search_move=sa_num - iter_pos -1;
						iter_pos = sa_num - 1;
					}
	#if Count_mem_ref
					// count_search_exp++;
					count_search_bs++;
	#endif
				}

				if (exact_match_flag){
					lower_b=iter_pos;
					n=1;
				}else{
					upper_b = iter_pos >= sa_num -1? sa_num-2: iter_pos ;
					lower_b = iter_pos > exp_search_move ? iter_pos- exp_search_move:1;
					n=  upper_b-lower_b + 1;
				}
			}
		}
		middle=iter_pos;
#elif CURR_SEARCH_METHOD ==1
		uint64_t upper_b = iter_pos+err >= sa_num-1? sa_num-2: iter_pos+err;
		uint64_t lower_b = iter_pos > curr_err? iter_pos-curr_err:1;
		lower_b = lower_b >= upper_b ? upper_b - 1: lower_b;
		uint64_t n = upper_b-lower_b + 1; // `end` is inclusive.
		uint64_t middle=iter_pos;
#else	//old binary search method
		err = std::max(curr_err,err);
		uint64_t upper_b = iter_pos+err >= sa_num-1? sa_num-2: iter_pos+err;
		uint64_t lower_b = iter_pos > err? iter_pos-err:1;
		lower_b = lower_b >= upper_b ? upper_b - 1: lower_b;
		uint64_t n = upper_b-lower_b + 1; // `end` is inclusive.
		uint64_t middle=iter_pos;
#endif
		
		// uint32_t count = 0;
		// Function adapted from https://github.com/gvinciguerra/rmi_pgm/blob/357acf668c22f927660d6ed11a15408f722ea348/main.cpp#L29.
		// Authored by Giorgio Vinciguerra.
		while (uint64_t half = (n>>1)) {
			middle = lower_b + half;
			last_match_len = match_len;
	#if Count_mem_ref
			count_search_bs++;
	#endif
	#if PREFETCH
			_mm_prefetch(sa_pos+ (middle+half/2)*SASIZE, _MM_HINT_T0);
			_mm_prefetch(sa_pos+(lower_b+half/2)*SASIZE, _MM_HINT_T0);
			
	#endif
			lower_b = compare_read_and_ref_binary_left(pac, sa_pos, middle, raux, sa_num ,read_valid_len, &match_len,&exact_match_flag)? middle: lower_b;
	
			if(exact_match_flag){
				break;
			}
			n -= half;
		}
// #if PREFETCH
// 		_mm_prefetch(sa_pos+((lower_b-16)<<1), _MM_HINT_T0);		
// #endif
		if (exact_match_flag){
			iter_pos = lower_b;
		}else{
			// no need to do additional linear search
			if(middle!=lower_b){
				// lookup key was smaller than middle, should check lower_b
				last_match_len = match_len;
				iter_pos = lower_b;
	#if Count_mem_ref
				count_search_linear++;
	#endif			
				while (!compare_read_and_ref_binary_left(pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
					if (iter_pos ==0){
						break;
					}
					#if PREFETCH
					if (g_use_simd8_mode) {
						_mm_prefetch(sa_pos + (iter_pos - 1) * SASIZE, _MM_HINT_T0);
					}
					#endif
					iter_pos --;
	
	#if Count_mem_ref
					count_search_linear++;
	#endif				
					
					last_match_len = match_len;
				}
				//  stops when ref is smaller than read, iter_pos should point to exact matching or bigger ref pos
				if (last_match_len > match_len){
					iter_pos++;
					match_len = last_match_len;
				}
			}
			else{
				// lookup key was bigger than middle, should check lower_b+1
				last_match_len = match_len;
				//  estimated position have smaller key than read, increase iter_pos and compare
				iter_pos =lower_b+1;
				iter_pos = iter_pos >= sa_num ? sa_num-1 : iter_pos;
	#if Count_mem_ref
				count_search_linear++;
	#endif		
				while (compare_read_and_ref_binary_left(pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
					if (iter_pos == sa_num - 1){
						break;
					}
					#if PREFETCH
					if (g_use_simd8_mode) {
						_mm_prefetch(sa_pos + (iter_pos + 1) * SASIZE, _MM_HINT_T0);
					}
					#endif
					iter_pos ++;
		
	#if Count_mem_ref
					count_search_linear++;
	#endif				
					
					last_match_len = match_len;
				}
				//  stops when ref is bigger than read or exact matches with read,
				if (last_match_len > match_len){
					iter_pos--;
					match_len = last_match_len;
				}
			}
		}
		//  iter_pos points to best exact matching position, match_len have the number of exact match
		uint64_t search_start_pos = iter_pos;
		uint32_t up_match=match_len, low_match=match_len, match_num=1;
		if (1 != min_intv_value  ){
			uint64_t up=1, low=1;
			bool low_search_flag, up_search_flag;
			while(1){
				while (1){
					low_search_flag = low_match >= match_len && low <= search_start_pos;
					up_search_flag = up_match >= match_len &&  sa_num-1 >= up + search_start_pos;
					if (low_search_flag){
#if Count_mem_ref
						count_search_min_intv++;
#endif
						compare_read_and_ref_binary_left(pac, sa_pos, (search_start_pos-low), raux, sa_num,match_len, &low_match,&exact_match_flag);
						#if PREFETCH
						if (g_use_simd8_mode) {
							_mm_prefetch(sa_pos + (search_start_pos - low - 1) * SASIZE, _MM_HINT_T0);
						}
						#endif
						low++;
					}
					if(up_search_flag){
#if Count_mem_ref
						count_search_min_intv++;
#endif
						compare_read_and_ref_binary_left(pac, sa_pos, (search_start_pos+up) , raux, sa_num,match_len, &up_match,&exact_match_flag);
						#if PREFETCH
						if (g_use_simd8_mode) {
							_mm_prefetch(sa_pos + (search_start_pos + up + 1) * SASIZE, _MM_HINT_T0);
						}
						#endif
						up++;
					}
					if (up+low-3 >= min_intv_value || (!low_search_flag && !up_search_flag)){
						break;
					}
				}
				if (low_match == match_len && low > search_start_pos){
					low+=1;
					low_match = 0;
				}
				if (up_match == match_len && ((up + search_start_pos) > sa_num-1)){
					up +=1;	
					up_match = 0;
				}
				match_num = up+low-3;
				if (  match_num >= min_intv_value){
						break;
				}
				match_len = up_match > low_match? up_match: low_match;
			}//end while 1
		} // if min_intv != 1
#if Count_mem_ref
		fprintf(stdout,"[mem_search func - left]\tMax match ref len:%d Count Total: %d Count_bs:%d Count_linear:%d Count_minintv:%d input minintv:%d\n",match_len, count_search_bs+count_search_linear+count_search_min_intv,count_search_bs, count_search_linear, count_search_min_intv, min_intv_value);
#endif
		*exact_match_len = match_len;
		return iter_pos;

	}
}

	
//#if MEM_TRADEOFF
uint64_t mem_search_tradeoff(const uint8_t* ref_string,const uint8_t* sa_pos,const uint8_t* pac, uint64_t sa_num,
                  Learned_read_aux_t* raux,  const uint64_t est_pos,
                    bool right_forward, uint32_t* exact_match_len, uint32_t* ambiguous_pos, bool no_search) {
    /*
		Find maximal exact match and save position in iter_pos
		if min_intv_value is bigger than 1, search for upper and lower bound where it satisfies min_intv_value
		Save seeds in smems if satisfies min_seed_len and min_intv_value
	*/
	
	int min_intv_value = raux->min_intv_limit ;

	uint64_t iter_pos = est_pos;
	if (!no_search){
		iter_pos = iter_pos > MEM_TRADEOFF_USECACHE_EXP_SEARCH_START? iter_pos:MEM_TRADEOFF_USECACHE_EXP_SEARCH_START;
		iter_pos = iter_pos < sa_num-1 - MEM_TRADEOFF_USECACHE_EXP_SEARCH_START? iter_pos:sa_num - 1-MEM_TRADEOFF_USECACHE_EXP_SEARCH_START;
	}
	uint32_t read_valid_len;
	bool exact_match_flag=false;
	uint32_t match_len;
	uint32_t last_match_len;
	uint64_t exp_search_move=MEM_TRADEOFF_USECACHE_EXP_SEARCH_START;
	
	#if Count_mem_ref
	uint32_t count_search_exp=0;
	uint32_t count_search_bs=0;
	uint32_t count_search_linear=0;
	uint32_t count_search_min_intv=0;
	uint32_t count_smem=0;
	#endif


	if (right_forward){
		read_valid_len = *ambiguous_pos - raux->pivot;
		if(no_search){
			match_len=read_valid_len;
		}else{
			// do exponential search
			uint64_t upper_b,lower_b,n,middle,half;
	#if Count_mem_ref
			count_search_exp++;
	#endif	
			if (!compare_read_and_ref_binary(pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)){
				if (exact_match_flag){
					lower_b=iter_pos;
					n=1;
				}else{
					iter_pos -= exp_search_move;
		#if PREFETCH
					_mm_prefetch(sa_pos+ ((iter_pos - exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
		#endif
		#if Count_mem_ref
					count_search_exp++;
		#endif
					while (!compare_read_and_ref_binary(pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
						if (iter_pos ==0){
							// smaller than iter_pos 0 -> search_start_pos should be 0
							break;
						}
						exp_search_move *=MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW;
						if (iter_pos > exp_search_move){
							iter_pos -= exp_search_move;
		#if PREFETCH
							_mm_prefetch(sa_pos+ ((iter_pos - exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
		#endif
						}else{
							exp_search_move=iter_pos;
							iter_pos = 0;
						}
		#if Count_mem_ref
						count_search_exp++;
		#endif
						
					}
					if (exact_match_flag){
						lower_b=iter_pos;
						n=1;
					}else{
						upper_b = iter_pos + exp_search_move>= sa_num-1? sa_num-2: iter_pos + exp_search_move;
						lower_b = iter_pos >0? iter_pos:1;
						n=  upper_b-lower_b + 1;
					}
				}
			}
			else{
				if (exact_match_flag){
					lower_b=iter_pos;
					n=1;
				}else{
					//  estimated position have smaller key than read, increase iter_pos and compare
					iter_pos += exp_search_move;
		#if PREFETCH
					_mm_prefetch(sa_pos+ ((iter_pos + exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
		#endif
		#if Count_mem_ref
					count_search_exp++;
		#endif
					while (compare_read_and_ref_binary(pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
						if (iter_pos == sa_num - 1){
							break;
						}
						exp_search_move *=MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW;
						if (sa_num - 1 > iter_pos+exp_search_move){
							iter_pos += exp_search_move;
		#if PREFETCH
							_mm_prefetch(sa_pos+ ((iter_pos + exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
		#endif		
						}else{
							exp_search_move=sa_num - iter_pos -1;
							iter_pos = sa_num - 1;
						}
		#if Count_mem_ref
						count_search_exp++;
		#endif
					}

					if (exact_match_flag){
						lower_b=iter_pos;
						n=1;
					}else{
						upper_b = iter_pos >= sa_num -1? sa_num-2: iter_pos ;
						lower_b = iter_pos > exp_search_move ? iter_pos- exp_search_move:1;
						n=  upper_b-lower_b + 1;
					}
				}

				
			}
			//done exponential search
			// std::cout <<"[memtradeoff-right] Binary search start\n";
			//do binary search
			middle=iter_pos;

			while ( half = (n>>1)) {
				
				middle = lower_b + half;
	#if Count_mem_ref
				count_search_bs++;
	#endif
	#if PREFETCH
				_mm_prefetch(sa_pos+ (middle+half/2)*SASIZE, _MM_HINT_T0);
				_mm_prefetch(sa_pos+(lower_b+half/2)*SASIZE, _MM_HINT_T0);
	#endif
				lower_b = compare_read_and_ref_binary(pac, sa_pos, middle , raux, sa_num,read_valid_len, &match_len,&exact_match_flag)? middle: lower_b;
				if(exact_match_flag){
					break;
				}
				n -= half;
			}
			// std::cout <<"[memtradeoff-right] Linear search start\n";
			if (exact_match_flag){
				iter_pos = lower_b;

			}else{
				// no need to do additional linear search
				if(middle!=lower_b){
					// lookup key was smaller than middle, should check lower_b
					last_match_len = match_len;
					iter_pos = lower_b;
				
			#if Count_mem_ref
					count_search_linear++;
			#endif
					while (!compare_read_and_ref_binary(pac, sa_pos, iter_pos , raux, sa_num, read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
						if (iter_pos ==0){
							break;
						} 
						#if PREFETCH
						if (g_use_simd8_mode) {
							_mm_prefetch(sa_pos + (iter_pos - 1) * SASIZE, _MM_HINT_T0);
						}
						#endif
						iter_pos --;
						
			#if Count_mem_ref
						count_search_linear++;
			#endif
						last_match_len = match_len;
					}
					//  stops when ref is smaller than read, iter_pos should point to exact matching or bigger ref pos
					if (last_match_len > match_len){
						iter_pos++;
						match_len = last_match_len;
					}
				}
				else{
					// lookup key was bigger than middle, should check lower_b+1
					last_match_len = match_len;
					//  estimated position have smaller key than read, increase iter_pos and compare
					iter_pos =lower_b+1;
					iter_pos = iter_pos >= sa_num ? sa_num-1 : iter_pos;
			#if Count_mem_ref
					count_search_linear++;
			#endif

					while (compare_read_and_ref_binary(pac, sa_pos,iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
						if (iter_pos == sa_num - 1){
							break;
						}
						#if PREFETCH
						if (g_use_simd8_mode) {
							_mm_prefetch(sa_pos + (iter_pos + 1) * SASIZE, _MM_HINT_T0);
						}
						#endif
						iter_pos ++;
			#if Count_mem_ref
						count_search_linear++;
			#endif
						

						last_match_len = match_len;
					}
					//  stops when ref is bigger than read or exact matches with read,
					if (last_match_len > match_len){
						iter_pos--;
						match_len = last_match_len;
					}
				}
			}
		}
		//  iter_pos points to best exact matching position, match_len have the number of exact match
		uint64_t search_start_pos = iter_pos;
		uint32_t up_match=match_len, low_match=match_len, match_num=1;
		// when need to find smems, should count number of matches
		if (1 != min_intv_value  ){
			uint64_t up=1, low=1;
			bool low_search_flag, up_search_flag;
			while(1){
				while (1){
					low_search_flag = low_match >= match_len && low <= search_start_pos;
					up_search_flag = up_match >= match_len &&  sa_num-1 >= up + search_start_pos;
					if (low_search_flag){
#if Count_mem_ref
	count_search_min_intv++;
#endif
						compare_read_and_ref_binary(pac, sa_pos, (search_start_pos-low), raux, sa_num,match_len, &low_match,&exact_match_flag);

						#if PREFETCH
						if (g_use_simd8_mode) {
							_mm_prefetch(sa_pos + (search_start_pos - low - 1) * SASIZE, _MM_HINT_T0);
						}
						#endif
						low++;
					}
					if(up_search_flag){
#if Count_mem_ref
						count_search_min_intv++;
#endif	
						compare_read_and_ref_binary(pac, sa_pos, (search_start_pos+up), raux, sa_num,match_len, &up_match,&exact_match_flag);

						#if PREFETCH
						if (g_use_simd8_mode) {
							_mm_prefetch(sa_pos + (search_start_pos + up + 1) * SASIZE, _MM_HINT_T0);
						}
						#endif
						up++;
					}
					if (up+low-3 >= min_intv_value || (!low_search_flag && !up_search_flag)){
						break;
					}
				}
				if (low_match == match_len && low > search_start_pos){
					low+=1;
					low_match = 0;
				}
				if (up_match == match_len && ((up + search_start_pos) > sa_num-1)){
					up +=1;	
					up_match = 0;
				}
				match_num = up+low-3;
				if (  match_num >= min_intv_value){
						break;
				}
				match_len = up_match > low_match? up_match: low_match;
			}//end while 1
		}
		*exact_match_len = match_len;
#if Count_mem_ref
		fprintf(stdout,"[mem_search_linear func - right]\tMax match ref len:%d Count Total: %d Count_exp:%d Count_bs:%d Count_linear:%d Count_minintv:%d input minintv:%d\n",match_len, count_search_bs+count_search_linear+count_search_min_intv+count_search_exp,count_search_exp,count_search_bs, count_search_linear, count_search_min_intv, min_intv_value);
#endif
		return iter_pos;
	}
	else{
		/************************************
		// left extension
		*************************************/

		read_valid_len = *ambiguous_pos - raux->l_pivot;
		if(no_search){
			match_len=read_valid_len;
		}else{
			uint64_t upper_b,lower_b,n,middle,half;
#if Count_mem_ref
			count_search_exp++;
#endif
			// std::cout <<"[memtradeoff-left] Exp search start\n";

			if (!compare_read_and_ref_binary_left(pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)){
				
				if (exact_match_flag){
					lower_b=iter_pos;
					n=1;
				}else{
					iter_pos -= exp_search_move;
#if PREFETCH
					_mm_prefetch(sa_pos+ ((iter_pos - exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
#endif
#if Count_mem_ref
					count_search_exp++;
#endif
					while (!compare_read_and_ref_binary_left(pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
						if (iter_pos ==0){
							// smaller than iter_pos 0 -> search_start_pos should be 0
							break;
						}
						exp_search_move *=MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW;
						if (iter_pos > exp_search_move){
							iter_pos -= exp_search_move;
#if PREFETCH
							_mm_prefetch(sa_pos+ ((iter_pos - exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
#endif							
						}else{
							exp_search_move=iter_pos;
							iter_pos = 0;
						}
#if Count_mem_ref
					count_search_exp++;
#endif
					}
					if (exact_match_flag){
						lower_b=iter_pos;
						n=1;
					}else{
						upper_b = iter_pos + exp_search_move>= sa_num-1? sa_num-2: iter_pos + exp_search_move;
						lower_b = iter_pos >0? iter_pos:1;
						n=  upper_b-lower_b + 1;
					}
				}
			}
			else{
				if (exact_match_flag){
					lower_b=iter_pos;
					n=1;
				}else{
					iter_pos += exp_search_move;
#if PREFETCH
					_mm_prefetch(sa_pos+ ((iter_pos + exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
#endif

#if Count_mem_ref
					count_search_exp++;
#endif
					while (compare_read_and_ref_binary_left(pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
						if (iter_pos == sa_num - 1){
							//
							break;
						}
						exp_search_move *=MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW;
						if (sa_num - 1 > iter_pos+exp_search_move){
							iter_pos += exp_search_move;
#if PREFETCH
							_mm_prefetch(sa_pos+ ((iter_pos + exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
#endif
						}else{
							exp_search_move=sa_num - iter_pos -1;
							iter_pos = sa_num - 1;
						}
#if Count_mem_ref
						count_search_exp++;
#endif
					}
					if (exact_match_flag){
						lower_b=iter_pos;
						n=1;
					}else{
						upper_b = iter_pos >= sa_num -1? sa_num-2: iter_pos ;
						lower_b = iter_pos > exp_search_move ? iter_pos- exp_search_move:1;
						n=  upper_b-lower_b + 1;
					}
				}
			}
			//done exponential search
			// std::cout <<"[memtradeoff-left] Binary search start\n";
			//do binary search
			middle=iter_pos;

			while ( half = (n>>1)) {
				
				middle = lower_b + half;
	#if Count_mem_ref
				count_search_bs++;
	#endif
	#if PREFETCH
				_mm_prefetch(sa_pos+ (middle+half/2)*SASIZE, _MM_HINT_T0);
				_mm_prefetch(sa_pos+(lower_b+half/2)*SASIZE, _MM_HINT_T0);
	#endif
				lower_b = compare_read_and_ref_binary_left(pac, sa_pos, middle , raux, sa_num,read_valid_len, &match_len,&exact_match_flag)? middle: lower_b;
				if(exact_match_flag){
					break;
				}
				n -= half;
			}
			// std::cout <<"[memtradeoff-left] Linear search start\n";
			if (exact_match_flag){
				iter_pos = lower_b;

			}else{
				// no need to do additional linear search
				if(middle!=lower_b){
					// lookup key was smaller than middle, should check lower_b
					last_match_len = match_len;
					iter_pos = lower_b;
				
			#if Count_mem_ref
					count_search_linear++;
			#endif
					while (!compare_read_and_ref_binary_left(pac, sa_pos, iter_pos , raux, sa_num, read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
						if (iter_pos ==0){
							break;
						} 
						#if PREFETCH
						if (g_use_simd8_mode) {
							_mm_prefetch(sa_pos + (iter_pos - 1) * SASIZE, _MM_HINT_T0);
						}
						#endif
						iter_pos --;
						
			#if Count_mem_ref
						count_search_linear++;
			#endif
						last_match_len = match_len;
					}
					//  stops when ref is smaller than read, iter_pos should point to exact matching or bigger ref pos
					if (last_match_len > match_len){
						iter_pos++;
						match_len = last_match_len;
					}
				}
				else{
					// lookup key was bigger than middle, should check lower_b+1
					last_match_len = match_len;
					//  estimated position have smaller key than read, increase iter_pos and compare
					iter_pos =lower_b+1;
					iter_pos = iter_pos >= sa_num ? sa_num-1 : iter_pos;
			#if Count_mem_ref
					count_search_linear++;
			#endif

					while (compare_read_and_ref_binary_left(pac, sa_pos,iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
						if (iter_pos == sa_num - 1){
							break;
						}
						#if PREFETCH
						if (g_use_simd8_mode) {
							_mm_prefetch(sa_pos + (iter_pos + 1) * SASIZE, _MM_HINT_T0);
						}
						#endif
						iter_pos ++;
			#if Count_mem_ref
						count_search_linear++;
			#endif
						

						last_match_len = match_len;
					}
					//  stops when ref is bigger than read or exact matches with read,
					if (last_match_len > match_len){
						iter_pos--;
						match_len = last_match_len;
					}
				}
			}

		}
		//  iter_pos points to best exact matching position, match_len have the number of exact match
		uint64_t search_start_pos = iter_pos;
		uint32_t up_match=match_len, low_match=match_len, match_num=1;
		if (1 != min_intv_value  ){
			uint64_t up=1, low=1;
			bool low_search_flag, up_search_flag;
			while(1){
				while (1){
					low_search_flag = low_match >= match_len && low <= search_start_pos;
					up_search_flag = up_match >= match_len &&  sa_num-1 >= up + search_start_pos;
					if (low_search_flag){
#if Count_mem_ref
						count_search_min_intv++;
#endif
						compare_read_and_ref_binary_left(pac, sa_pos, (search_start_pos-low), raux, sa_num,match_len, &low_match,&exact_match_flag);
						#if PREFETCH
						if (g_use_simd8_mode) {
							_mm_prefetch(sa_pos + (search_start_pos - low - 1) * SASIZE, _MM_HINT_T0);
						}
						#endif
						low++;


					}
					
					if(up_search_flag){
#if Count_mem_ref
						count_search_min_intv++;
#endif
						compare_read_and_ref_binary_left(pac, sa_pos, (search_start_pos+up), raux, sa_num,match_len, &up_match,&exact_match_flag);
						#if PREFETCH
						if (g_use_simd8_mode) {
							_mm_prefetch(sa_pos + (search_start_pos + up + 1) * SASIZE, _MM_HINT_T0);
						}
						#endif
						up++;

					}
					if (up+low-3 >= min_intv_value || (!low_search_flag && !up_search_flag)){
						break;
					}
				}
				if (low_match == match_len && low > search_start_pos){
					low+=1;
					low_match = 0;
				}
				if (up_match == match_len && ((up + search_start_pos) > sa_num-1)){
					up +=1;	
					up_match = 0;
				}
				match_num = up+low-3;
				if (  match_num >= min_intv_value){
						break;
				}
				match_len = up_match > low_match? up_match: low_match;
			}//end while 1
		}
#if Count_mem_ref
	fprintf(stdout,"[mem_search_linear func - left]\tMax match ref len:%d Count Total: %d Count_exp:%d Count_bs:%d Count_linear:%d Count_minintv:%d input minintv:%d\n",match_len, count_search_bs+count_search_linear+count_search_min_intv+count_search_exp,count_search_exp,count_search_bs, count_search_linear, count_search_min_intv, min_intv_value);
#endif
		*exact_match_len = match_len;
		return iter_pos;
	} // end of left extension
}

uint64_t right_smem_search_tradeoff(const uint8_t* ref_string,const uint8_t* sa_pos,const uint8_t* pac, uint64_t sa_num,
                   Learned_read_aux_t* raux,  const uint64_t est_pos,
                     uint32_t* exact_match_len, mem_tlv* smems, u64v* hits, uint32_t* ambiguous_pos, bool no_search) {
    /*
		Find maximal exact match and save position in iter_pos
		if min_intv_value is bigger than 1, search for upper and lower bound where it satisfies min_intv_value
		Save seeds in smems if satisfies min_seed_len and min_intv_value
	*/
	int min_intv_value = raux->min_intv_limit ;
	uint64_t iter_pos = est_pos;
	if (!no_search){
		iter_pos = iter_pos > MEM_TRADEOFF_USECACHE_EXP_SEARCH_START? iter_pos:MEM_TRADEOFF_USECACHE_EXP_SEARCH_START;
		iter_pos = iter_pos < sa_num -1- MEM_TRADEOFF_USECACHE_EXP_SEARCH_START? iter_pos:sa_num -1- MEM_TRADEOFF_USECACHE_EXP_SEARCH_START;
	}
	uint64_t read_valid_len= *ambiguous_pos - raux->pivot;
	bool exact_match_flag=false;
	uint32_t match_len;
	uint32_t last_match_len;
	uint64_t exp_search_move=MEM_TRADEOFF_USECACHE_EXP_SEARCH_START;
	uint64_t upper_b,lower_b,n,middle, half;
#if Count_mem_ref
	uint32_t count_search_exp=0;
	uint32_t count_search_bs=0;
	uint32_t count_search_linear=0;
	uint32_t count_search_min_intv=0;
	uint32_t count_smem=0;
	
#endif

	if (no_search){
		match_len = read_valid_len;
	}else{
		// do exponential search
		
		
#if Count_mem_ref
		count_search_exp++;
#endif	
		// std::cout <<"[smemtradeoff] Exp search start\n";
		// std::cout <<"iter_pos:"<<iter_pos<<"\n";
		if (!compare_read_and_ref_binary(pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)){
			if (exact_match_flag){
				lower_b=iter_pos;
				n=1;
			}else{
				iter_pos -= exp_search_move;
	#if PREFETCH
				_mm_prefetch(sa_pos+ ((iter_pos - exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
	#endif
	#if Count_mem_ref
				count_search_exp++;
	#endif
				while (!compare_read_and_ref_binary(pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
					if (iter_pos ==0){
						// smaller than iter_pos 0 -> search_start_pos should be 0
						break;
					}
					exp_search_move *=MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW;
					if (iter_pos > exp_search_move){
						
						iter_pos -= exp_search_move;
	#if PREFETCH
						_mm_prefetch(sa_pos+ ((iter_pos - exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
	#endif								
					}else{
						exp_search_move=iter_pos;
						iter_pos = 0;
					}
	#if Count_mem_ref
					count_search_exp++;
	#endif
					
				}
				if (exact_match_flag){
					lower_b=iter_pos;
					n=1;
				}else{
					upper_b = iter_pos + exp_search_move>= sa_num-1? sa_num-2: iter_pos + exp_search_move;
					lower_b = iter_pos >0? iter_pos:1;
					n=  upper_b-lower_b + 1;
				}
				
			}
		}
		else{
			if (exact_match_flag){
				lower_b=iter_pos;
				n=1;
			}else{
				//  estimated position have smaller key than read, increase iter_pos and compare
				iter_pos += exp_search_move;
	#if PREFETCH
				_mm_prefetch(sa_pos+ ((iter_pos + exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
	#endif				
	#if Count_mem_ref
				count_search_exp++;
	#endif
				while (compare_read_and_ref_binary(pac, sa_pos, iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
					if (iter_pos == sa_num - 1){
						break;
					}
					exp_search_move *=MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW;
					if (sa_num - 1 > iter_pos+exp_search_move){
						iter_pos += exp_search_move;
	#if PREFETCH
						_mm_prefetch(sa_pos+ ((iter_pos + exp_search_move*MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW)*SASIZE), _MM_HINT_T0);
	#endif						
					}else{
						exp_search_move=sa_num - iter_pos -1;
						iter_pos = sa_num - 1;
					}
	#if Count_mem_ref
					count_search_exp++;
	#endif
				}

				if (exact_match_flag){
					lower_b=iter_pos;
					n=1;
				}else{
					upper_b = iter_pos >= sa_num -1? sa_num-2: iter_pos ;
					lower_b = iter_pos > exp_search_move ? iter_pos- exp_search_move:1;
					n=  upper_b-lower_b + 1;
				}
			}

			
		}
		//done exponential search
		//do binary search
		middle=iter_pos;
		while ( half = (n>>1)) {
			
			middle = lower_b + half;
#if Count_mem_ref
			count_search_bs++;
#endif
#if PREFETCH
			_mm_prefetch(sa_pos+ (middle+half/2)*SASIZE, _MM_HINT_T0);
			_mm_prefetch(sa_pos+(lower_b+half/2)*SASIZE, _MM_HINT_T0);
#endif
			lower_b = compare_read_and_ref_binary(pac, sa_pos, middle , raux, sa_num,read_valid_len, &match_len,&exact_match_flag)? middle: lower_b;
			if(exact_match_flag){
				break;
			}
			n -= half;
		}
		if (exact_match_flag){
			iter_pos = lower_b;

		}else{
			// no need to do additional linear search
			if(middle!=lower_b){
				// lookup key was smaller than middle, should check lower_b
				last_match_len = match_len;
				iter_pos = lower_b;
			
		#if Count_mem_ref
				count_search_linear++;
		#endif
				while (!compare_read_and_ref_binary(pac, sa_pos, iter_pos , raux, sa_num, read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
					if (iter_pos ==0){
						break;
					} 
					#if PREFETCH
					if (g_use_simd8_mode) {
						_mm_prefetch(sa_pos + (iter_pos - 1) * SASIZE, _MM_HINT_T0);
					}
					#endif
					iter_pos --;
					
		#if Count_mem_ref
					count_search_linear++;
		#endif
					last_match_len = match_len;
				}
				// stops when ref is smaller than read, iter_pos should point to exact matching or bigger ref pos
				if (last_match_len > match_len){
					iter_pos++;
					match_len = last_match_len;
				}
			}
			else{
				// lookup key was bigger than middle, should check lower_b+1
				last_match_len = match_len;
				// estimated position have smaller key than read, increase iter_pos and compare
				iter_pos =lower_b+1;
				iter_pos = iter_pos >= sa_num ? sa_num-1 : iter_pos;
		#if Count_mem_ref
				count_search_linear++;
		#endif

				while (compare_read_and_ref_binary(pac, sa_pos,iter_pos, raux, sa_num,read_valid_len, &match_len,&exact_match_flag)&& !exact_match_flag ){
					if (iter_pos == sa_num - 1){
						break;
					}
					#if PREFETCH
					if (g_use_simd8_mode) {
						_mm_prefetch(sa_pos + (iter_pos + 1) * SASIZE, _MM_HINT_T0);
					}
					#endif
					iter_pos ++;
		#if Count_mem_ref
					count_search_linear++;
		#endif
					

					last_match_len = match_len;
				}
				// stops when ref is bigger than read or exact matches with read,
				if (last_match_len > match_len){
					iter_pos--;
					match_len = last_match_len;
				}
			}
		}

	}
	// iter_pos points to best exact matching position, match_len have the number of exact match
	uint64_t search_start_pos = iter_pos;
	uint32_t up_match=match_len, low_match=match_len, match_num=1;
	// when need to find smems, should count number of matches
#if EXPONENTIAL_SMEMSEARCH
	uint64_t up=0, low=0;
	while(1){
		while ( up+low < 15){
				if ( match_len < raux->min_seed_len && (up+low+ (up==0)+(low==0)-1 ) >= min_intv_value){
					#if Count_mem_ref
					fprintf(stdout,"[smem_search func - right e-lin]\tMax match ref len:%d Count Total: %d Count_bs:%d Count_linear:%d Count_minintv:%d input minintv:%d\n",match_len,count_search_bs+count_search_linear+count_search_min_intv, count_search_bs, count_search_linear, count_search_min_intv, min_intv_value);
					#endif
					*exact_match_len = match_len;
					return iter_pos;
				}
				if (low_match >= match_len && low+1 <= search_start_pos){
					#if Count_mem_ref
					count_search_min_intv++;
					#endif
					#if PREFETCH
					if (g_use_simd8_mode) {
						_mm_prefetch(sa_pos + (search_start_pos - low - 1) * SASIZE, _MM_HINT_T0);
					}
					#endif
					low++;
					compare_read_and_ref_binary(pac, sa_pos, (search_start_pos-low), raux, sa_num,match_len, &low_match,&exact_match_flag);

				}
				else if(up_match >= match_len &&  sa_num-2 >= up + search_start_pos){
					#if Count_mem_ref
					count_search_min_intv++;
					#endif			
					#if PREFETCH
					if (g_use_simd8_mode) {
						_mm_prefetch(sa_pos + (search_start_pos + up + 1) * SASIZE, _MM_HINT_T0);
					}
					#endif
					up++;
					compare_read_and_ref_binary(pac, sa_pos, (search_start_pos+up), raux, sa_num,match_len, &up_match,&exact_match_flag);
				}
				else{
					break;
				}

				if ( match_len < raux->min_seed_len && (up+low+(up==0)-1) >= min_intv_value){
					#if Count_mem_ref
					fprintf(stdout,"[smem_search func - right e-lin]\tMax match ref len:%d Count Total: %d Count_bs:%d Count_linear:%d Count_minintv:%d input minintv:%d\n",match_len,count_search_bs+count_search_linear+count_search_min_intv, count_search_bs, count_search_linear, count_search_min_intv, min_intv_value);
					#endif
					*exact_match_len = match_len;
					return iter_pos;
				}
		}
		

		#if CURR_SEARCH_METHOD != 2
		uint64_t exp_search_move = EXPONENTIAL_EXP_START;
		#else
		exp_search_move = EXPONENTIAL_EXP_START;
		#endif
		
		if (low_match >= match_len && low < search_start_pos){
			//exponential search
			upper_b = search_start_pos-low;
			exp_search_move = std::min(search_start_pos-low, exp_search_move);
			lower_b = search_start_pos-low-exp_search_move;
			n=  upper_b-lower_b + 1;
			while (low_match >= match_len){
				#if Count_mem_ref
				count_search_min_intv++;
				#endif
				compare_read_and_ref_binary(pac, sa_pos, lower_b, raux, sa_num,match_len, &low_match,&exact_match_flag);
				if (low_match < match_len ){
					// found different position
					break;
				}
				if (lower_b == 0){
					break;
				}
				exp_search_move <<= EXPONENTIAL_EXP_POW;
				exp_search_move = std::min(exp_search_move,lower_b);
				lower_b -= exp_search_move;
				
				upper_b = lower_b + exp_search_move;
				n=  upper_b-lower_b + 1;
			}
			//binary search 
			while (uint64_t half = (n>>1)) {
				if (match_len < raux->min_seed_len &&  up+search_start_pos-upper_b >= min_intv_value){
					#if Count_mem_ref
					fprintf(stdout,"[smem_search func - right e-exp]\tMax match ref len:%d Count Total: %d Count_bs:%d Count_linear:%d Count_minintv:%d input minintv:%d\n",match_len,count_search_bs+count_search_linear+count_search_min_intv, count_search_bs, count_search_linear, count_search_min_intv, min_intv_value);
					#endif
					*exact_match_len = match_len;
					return iter_pos;
				}
				middle = upper_b - half;
				// middle = lower_b + half;
				#if Count_mem_ref
					count_search_min_intv++;
				#endif
				#if PREFETCH
				_mm_prefetch(sa_pos+(middle-half/2)*SASIZE, _MM_HINT_T0);
				_mm_prefetch(sa_pos+((upper_b-half/2)*SASIZE), _MM_HINT_T0);
				#endif
				
				compare_read_and_ref_binary(pac, sa_pos, middle , raux, sa_num,match_len, &low_match,&exact_match_flag);
				if (low_match >= match_len){
					upper_b = middle;
					if (upper_b !=0 && n<3){
						//update low_match to the next biggest exact match length
						#if Count_mem_ref
							count_search_min_intv++;
						#endif
						compare_read_and_ref_binary(pac, sa_pos, (upper_b-1) , raux, sa_num,match_len, &low_match,&exact_match_flag);
					}
				}
				n -= half;
			}
			low = search_start_pos+1 - upper_b ;
			if (match_len < raux->min_seed_len &&  up+low-1 >= min_intv_value){
				#if Count_mem_ref
					fprintf(stdout,"[smem_search func - right e-exp]\tMax match ref len:%d Count Total: %d Count_bs:%d Count_linear:%d Count_minintv:%d input minintv:%d\n",match_len,count_search_bs+count_search_linear+count_search_min_intv, count_search_bs, count_search_linear, count_search_min_intv, min_intv_value);
				#endif
				*exact_match_len = match_len;
				return iter_pos;
			}
		}

		if(up_match >= match_len && sa_num-1 > up + search_start_pos){
			// find Up
			exp_search_move = EXPONENTIAL_EXP_START;
			//exponential search
			lower_b = search_start_pos+up;
			exp_search_move = std::min(sa_num-1-lower_b, exp_search_move);
			upper_b = lower_b+exp_search_move;
			n=  upper_b-lower_b + 1;
			while (up_match >= match_len){
				#if Count_mem_ref
				count_search_min_intv++;
				#endif
				compare_read_and_ref_binary(pac, sa_pos, upper_b, raux, sa_num,match_len, &up_match,&exact_match_flag);
				if (up_match < match_len){
					break;
				}
				if (upper_b == sa_num-1){
					break;
				}
				exp_search_move <<= EXPONENTIAL_EXP_POW;
				exp_search_move = std::min(exp_search_move,sa_num-1-upper_b);
				upper_b += exp_search_move;
				lower_b = upper_b-exp_search_move;
				n=  upper_b-lower_b + 1;
			}
			//binary search 
			while (uint64_t half = (n>>1)) {
				if (match_len < raux->min_seed_len &&  lower_b-search_start_pos+low >= min_intv_value){
					#if Count_mem_ref
						fprintf(stdout,"[smem_search func - right e-exp]\tMax match ref len:%d Count Total: %d Count_bs:%d Count_linear:%d Count_minintv:%d input minintv:%d\n",match_len,count_search_bs+count_search_linear+count_search_min_intv, count_search_bs, count_search_linear, count_search_min_intv, min_intv_value);
					#endif
					*exact_match_len = match_len;
					return iter_pos;
				}
				middle = lower_b + half;
				#if Count_mem_ref
					count_search_min_intv++;
				#endif
				#if PREFETCH
				_mm_prefetch(sa_pos+(middle+half/2)*SASIZE, _MM_HINT_T0);
				_mm_prefetch(sa_pos+((lower_b+half/2)*SASIZE), _MM_HINT_T0);
				#endif
				compare_read_and_ref_binary(pac, sa_pos, middle, raux, sa_num,match_len, &up_match,&exact_match_flag);
				if (up_match >= match_len){
					lower_b = middle;
					if(lower_b != sa_num-1 && n<3){
						#if Count_mem_ref
							count_search_min_intv++;
						#endif
						compare_read_and_ref_binary(pac, sa_pos, (lower_b+1), raux, sa_num,match_len, &up_match,&exact_match_flag);
					}
				}
				n -= half;
			}
			up = lower_b+1 - search_start_pos;

			if (match_len < raux->min_seed_len &&  up+low-1 >= min_intv_value){
				#if Count_mem_ref
					fprintf(stdout,"[smem_search func - right e-exp]\tMax match ref len:%d Count Total: %d Count_bs:%d Count_linear:%d Count_minintv:%d input minintv:%d\n",match_len,count_search_bs+count_search_linear+count_search_min_intv, count_search_bs, count_search_linear, count_search_min_intv, min_intv_value);
				#endif
				*exact_match_len = match_len;
				return iter_pos;
			}

		}

		if (low_match >= match_len && low >= search_start_pos){
			low = search_start_pos+1;
			low_match = 0;
		}
		if (up_match >= match_len && ((up + search_start_pos) >= sa_num-1)){
			up = sa_num-search_start_pos;	
			up_match = 0;
		}
		
		match_num = up+low-1;
		iter_pos=search_start_pos-low+1;
		if (  match_num >= min_intv_value){
				break;
		}
		match_len = up_match > low_match? up_match: low_match;
	}//end while 1
#else
	uint64_t up=1, low=1;

	while(1){
		while (1){
			if (match_len < raux->min_seed_len &&  (up+low-3+(up==1)+(low==1)) >= min_intv_value){
#if Count_mem_ref
				fprintf(stdout,"[smem_search func - right]\tMax match ref len:%d Count Total: %d Count_bs:%d Count_linear:%d Count_minintv:%d input minintv:%d\n",match_len,count_search_bs+count_search_linear+count_search_min_intv, count_search_bs, count_search_linear, count_search_min_intv, min_intv_value);
#endif
				*exact_match_len = match_len;
				return iter_pos;
			}

			if (low_match >= match_len && low <= search_start_pos){
#if Count_mem_ref
				count_search_min_intv++;
#endif
				compare_read_and_ref_binary(pac, sa_pos, (search_start_pos-low), raux, sa_num,match_len, &low_match,&exact_match_flag);


				low++;
				
			}
			else if(up_match >= match_len &&  sa_num-1 >= up + search_start_pos){
#if Count_mem_ref
				count_search_min_intv++;
#endif					
				compare_read_and_ref_binary(pac, sa_pos, (search_start_pos+up), raux, sa_num,match_len, &up_match,&exact_match_flag);

				up++;
				
			}
			else{
				break;
			}
			if (match_len < raux->min_seed_len &&  (up+low-3+(up==1)) >= min_intv_value){
#if Count_mem_ref
				fprintf(stdout,"[smem_search func - right]\tMax match ref len:%d Count Total: %d Count_bs:%d Count_linear:%d Count_minintv:%d input minintv:%d\n",match_len,count_search_bs+count_search_linear+count_search_min_intv, count_search_bs, count_search_linear, count_search_min_intv, min_intv_value);
#endif
				*exact_match_len = match_len;
				return iter_pos;
			}
		}
		if (low_match == match_len && low > search_start_pos){
			low = search_start_pos+2;
			low_match = 0;
		}
		if (up_match == match_len && ((up + search_start_pos) > sa_num-1)){
			up = sa_num+1-search_start_pos;	
			up_match = 0;
		}
		match_num = up+low-3;
		iter_pos=search_start_pos-low+2;
		if (  match_num >= min_intv_value){
				break;
		}
		match_len = up_match > low_match? up_match: low_match;
	}//end while 1
#endif
	*exact_match_len = match_len;
	// add found smems to smems and hits array
	if (match_len>= raux->min_seed_len){
		mem_tl mem;
		mem.start = raux->pivot;
		mem.end = raux->pivot + match_len;
		mem.hitbeg = hits->n; // begin index in hits vector
		mem.hitcount = match_num; // number of hits in reference
		for (uint64_t i=0; i< match_num; i++){
			uint64_t pos_val = *(uint32_t*)(sa_pos + (iter_pos+i)*SASIZE );
			pos_val = pos_val <<8 | sa_pos[(iter_pos+i)*SASIZE + 4];
			kv_push(uint64_t, *hits, pos_val );
		}
		kv_push(mem_tl, *smems, mem);
	}
#if Count_mem_ref
	fprintf(stdout,"[smem_search_linear func - right]\tMax match ref len:%d Count Total: %d Count_exp:%d Count_bs:%d Count_linear:%d Count_minintv:%d input minintv:%d\n",match_len,count_search_bs+count_search_linear+count_search_min_intv+count_search_exp,count_search_exp, count_search_bs, count_search_linear, count_search_min_intv, min_intv_value);
#endif
	return iter_pos;
}
