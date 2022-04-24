#! /usr/bin/env bash

if  ! command -v bwa-meme-train-prmi &> /dev/null 
then
    # bwa-meme-train-prmi command not found, should build it
    # To build the command, RMI folder should exist
    if [ ! -d RMI ] 
    then
        echo "ERROR:"
        echo BWA-MEME/RMI directory does not exist or run this script in BWA-MEME folder
        exit 1
    fi

    # check Cargo exists
    if ! command -v cargo &> /dev/null
    then
        echo "Command: cargo could not be found, please install cargo from https://rustup.rs/"
        exit 1
    fi
    # build the binary 
    cd RMI && cargo build --release && cd ..
fi

if [ -z "$1" ]; then
    
    echo "Should give input 
ex) bash build_rmis_dna.sh ./human.fasta"
    exit 0
fi

if [ ! -f $1.suffixarray_uint64 ] 
then
    echo "ERROR:"
    echo "suffixarray_uint64 file does not exists in `dirname $1`
Correct the path or build the suffixarray index first with command: bwa-meme index"
    exit 1
fi

function build_rmi_set() {
    DATA_PATH=$1
    DATA_NAME=`basename $1`
    DIR_NAME=`dirname $1`
    
    shift 1
    if  ! command -v bwa-meme-train-prmi &> /dev/null 
    then
        # RMI/target/release/bwa-meme-train-prmi --data-path $DIR_NAME $DATA_PATH $DATA_NAME pwl,linear,linear_spline 1024
	    RMI/target/release/bwa-meme-train-prmi --data-path $DIR_NAME $DATA_PATH $DATA_NAME pwl,linear,linear_spline 268435456
    else
        bwa-meme-train-prmi --data-path $DIR_NAME $DATA_PATH $DATA_NAME pwl,linear,linear_spline 268435456
    fi
}

build_rmi_set $1.suffixarray_uint64
