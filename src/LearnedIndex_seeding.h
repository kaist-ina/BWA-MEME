#ifndef LEARNEDSEEDING_HPP
#define LEARNEDSEEDING_HPP

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <vector>
#include "kstring.h"
#include "ksw.h"
#include "kvec.h"
#include "ksort.h"
#include "utils.h"
#include "bwt.h"
#include "bntseq.h"
#include "bwa.h"
#include "macro.h"
#include "profiling.h"

#include "memcpy_bwamem.h"
#if (__AVX512BW__ || __AVX2__)
#include <immintrin.h>
#else
#include <smmintrin.h>  // for SSE4.1
#define __mmask8 uint8_t
#define __mmask16 uint16_t
#endif
#include "ertseeding.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "safe_mem_lib.h"
#include "safe_str_lib.h"
#include <snprintf_s.h>
#ifdef __cplusplus
}
#endif

#include <cstddef>
#include <cstdint>

// Configs for learned index inference 

/*
	1: Without 64bit key and ISA
	2: Without ISA
	3: BWA-MEME full
*/
#define MODE 3

#if MODE==1
#define LOADSUFFIX 0
#define MEM_TRADEOFF 0
#elif MODE == 2
#define LOADSUFFIX 1
#define MEM_TRADEOFF 0
#elif MODE == 3
#define LOADSUFFIX 1
#define MEM_TRADEOFF 1
#endif



#define READ_FROM_FILE 1  // set 0 to build in runtime, should not be turned on when using compact index

// Change to test last mile search method or print query length and number of last mile search
#define CURR_SEARCH_METHOD 1 // 2: exponential 1: P-RMI 0: binary
#define Count_mem_ref 0




// No need to change below
#define PREFETCH 1
#define EXPONENTIAL_SMEMSEARCH 1
#define EXPONENTIAL_EXP_START 5
#define EXPONENTIAL_EXP_POW 1
#define MEM_TRADEOFF_CACHED 1
#define MEM_TRADEOFF_USECACHE_THRESHOLD 18 // 
#define MEM_TRADEOFF_USECACHE_EXP_SEARCH_START 2
#define MEM_TRADEOFF_USECACHE_EXP_SEARCH_POW 5 // 
#define REMOVE_DUP_SEED 0 // Done change this
#define DEBUG_MODE 0
// #define COMPACT_INDEX 1 // 0 for build in runtime with non compact
#define PAD_1 1
#if LOADSUFFIX
	#define SASIZE 13
#else
	#define SASIZE 5
#endif



// /**
//  * State for each maximal-exact-match (MEM)
//  */
typedef struct {
	// uint8_t forward;    // RMEM or LMEM. We need this to normalize hit positions
	int start;          // MEM start position in read
	int end;            // MEM end position in read. [start, end)
	// int rc_start;       // MEM start position in reverse complemented (RC) read (used for backward search)
	// int rc_end;         // MEM end position in reverse complemented (RC) read (used for backward search)
	int hitbeg;         // Index into hit array
	int hitcount;       // Count of hits


	// int cache_pivot; // cached match start position in right extension  same as start
	// int cache_l_seq; // length of cached match end - start is len
	uint64_t cache_refpos; // length of cached match

	// int end_correction; // Amount by which MEM has extended beyond backward search start position in read
} mem_tl;

static const unsigned char BitReverseTable256[256] = 
{
#   define R2(n)     n,     n + 1*64,     n + 2*64,     n + 3*64
#   define R4(n) R2(n), R2(n + 1*16), R2(n + 2*16), R2(n + 3*16)
#   define R6(n) R4(n), R4(n + 1*4 ), R4(n + 2*4 ), R4(n + 3*4 )
    R6(0), R6(1), R6(2), R6(3)
};

typedef kvec_t(mem_tl) mem_tlv;


typedef struct {
	uint8_t* sa_pos;
	uint8_t* ref2sa;         // ref2sa
	const bwt_t* bwt;           // FM-index
	const bntseq_t* bns;        // Input reads sequences
	const uint8_t* pac;         // Reference genome (2-bit encoded)
  uint8_t* ref_string;
} Learned_index_aux_t;

/**
 * 'Read' auxiliary data structures
 */
typedef struct {
	int min_seed_len;               // Minimum length of seed
	int l_seq;                      // Read length
	int num_hits;                   // Number of hits for each node in the ERT
    int pivot; // right, forward pivot
    int l_pivot; // left, backward pivot
	int max_pivot; // longest match start position in right extension 
	int max_l_seq; // length of longest match
	uint64_t max_refpos; // length of longest match
	int cache_pivot; // cached match start position in right extension  
	int cache_pivot_end; // length of cached match 
	uint64_t cache_refpos; // length of cached match
	int min_intv_limit;                      // Number of hits after which extension must be stopped
	char* read_name;                // Read name
	uint8_t* unpacked_queue_buf;    // Read sequence (2-bit encoded)
	uint8_t* unpacked_rc_queue_buf; // Reverse complemented read (2-bit encoded)
	uint8_t* unpacked_queue_binary_buf_shift1;    // Read sequence (2-bit encoded)
	uint8_t* unpacked_queue_binary_buf_shift2;    // Read sequence (2-bit encoded)
	uint8_t* unpacked_queue_binary_buf_shift3;    // Read sequence (2-bit encoded)
	uint8_t* unpacked_queue_binary_buf_shift4;    // Read sequence (2-bit encoded)
	uint8_t* unpacked_rc_queue_binary_buf_shift1;    // Read sequence (2-bit encoded)
	uint8_t* unpacked_rc_queue_binary_buf_shift2;    // Read sequence (2-bit encoded)
	uint8_t* unpacked_rc_queue_binary_buf_shift3;    // Read sequence (2-bit encoded)
	uint8_t* unpacked_rc_queue_binary_buf_shift4;    // Read sequence (2-bit encoded)
} Learned_read_aux_t;
#define _get_pac_bigendian(pac, l) ( BitReverseTable256[(pac)[(l)>>2]]>>((~(l)&3)<<1)&3)

inline uint64_t get_key_of_ref(uint8_t* pac , uint64_t pos, uint64_t len){
    uint64_t key=0,r__=32;
    // assert(len!=0);
    for (; r__>= len; r__--){
        
        key = key << 2;
        key |= 3;
    }
    for (; r__>=0; r__--){
        // fprintf(stderr,"%llu\n",key);
        key = key << 2;
        key |= _get_pac_bigendian(pac,pos+r__);
        if (r__==0){
            break;
        }
    }
    return key;
};

uint64_t right_smem_search(const uint8_t* ref_string,const uint8_t* sa_pos,const uint8_t* pac, uint64_t sa_num,
                   Learned_read_aux_t* raux,  const uint64_t est_pos,
                    uint64_t err, uint32_t* exact_match_len, mem_tlv* smems, u64v* hits, uint32_t* ambiguous_pos);
uint64_t mem_search(const uint8_t* ref_string,const uint8_t* sa_pos,const uint8_t* pac, uint64_t sa_num,
                   Learned_read_aux_t* raux, const uint64_t est_pos,
                   uint64_t err, bool right_forward, uint32_t* exact_match_len, uint32_t* ambiguous_pos);

uint64_t mem_search_tradeoff(const uint8_t* ref_string,const uint8_t* sa_pos,const uint8_t* pac, uint64_t sa_num,
                   Learned_read_aux_t* raux, const uint64_t est_pos,
                   bool right_forward, uint32_t* exact_match_len, uint32_t* ambiguous_pos, bool no_search);				   

uint64_t right_smem_search_tradeoff(const uint8_t* ref_string,const uint8_t* sa_pos,const uint8_t* pac, uint64_t sa_num,
                   Learned_read_aux_t* raux,  const uint64_t est_pos,
                    uint32_t* exact_match_len, mem_tlv* smems, u64v* hits, uint32_t* ambiguous_pos, bool no_search);				   

void Learned_bwtSeedStrategyAllPosOneThread(Learned_index_aux_t* iaux, Learned_read_aux_t* raux, mem_tlv* smems, u64v* hits, bool hasN);
void Learned_bwtSeedStrategyAllPosOneThread_mem_tradeoff(Learned_index_aux_t* iaux, Learned_read_aux_t* raux, mem_tlv* smems, u64v* hits, bool hasN);
void Learned_getSMEMsAllPosOneThread_step1only(Learned_index_aux_t* iaux, Learned_read_aux_t* raux, mem_tlv* smems, u64v* hits, bool hasN, int split_len, int split_width);

void Learned_getSMEMsAllPosOneThread(Learned_index_aux_t* iaux, Learned_read_aux_t* raux, mem_tlv* smems, u64v* hits, bool hasN, int split_len, int split_width);
void Learned_getSMEMsOnePosOneThread(Learned_index_aux_t* iaux, Learned_read_aux_t* raux, mem_tlv* smems, u64v* hits, bool hasN, bool use_cached=false);
void Learned_getSMEMsOnePosOneThread_step1(Learned_index_aux_t* iaux, Learned_read_aux_t* raux, mem_tlv* smems, u64v* hits, bool hasN, bool use_cached=false);
extern uint8_t sa_raux_buf[4][4];
// extern uint32_t sa_raux_buf[16];

inline uint64_t Tokenization( Learned_read_aux_t* raux, bool right_forward, uint32_t* ambiguous_pos, bool hasN);

bool check_exact_match(Learned_index_aux_t* iaux, Learned_read_aux_t* raux, mem_tlv* smems, u64v* hits, bool hasN, bool use_cached=false);

// namespace learned_index {
bool learned_index_load(char const* dataPath, char const* dataPath2,char const* dataPath3, double suffix_array_num);
void learned_index_cleanup();
inline uint64_t learned_index_lookup(uint64_t key, size_t* err);
// disable this in pwl,linear model
extern double L0_PARAMETER0;
extern double L0_PARAMETER1;
//
extern double SA_NUM;
extern char* L1_PARAMETERS;
// for partial 3 layer models
extern char* L2_PARAMETERS;


// }


#endif
