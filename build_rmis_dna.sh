#! /usr/bin/env bash

if [ -z "$1" ]; then
    echo "Should give input 
ex) bash build_rmis_dna.sh ./human.fasta"
    exit
fi

mkdir -p rmi_data 


function build_rmi_set() {
    DATA_PATH=$1
    DATA_NAME=`basename $1`
    DIR_NAME=`dirname $1`
    HEADER_PATH=rmi/${DATA_NAME}_0.h
    JSON_PATH=rmi_specs/config.json
    

    shift 1
    if [ ! -f $HEADER_PATH ]; then
        echo "Building RMI set for $DATA_NAME"
	#RMI/target/release/rmi $DATA_PATH $DATA_NAME pwl,linear 67108864
	RMI/target/release/rmi $DATA_PATH $DATA_NAME pwl,linear,linear_spline 268435456

	
	#mv ${DATA_NAME}_* ${DIR_NAME}
	cp  rmi_data/${DATA_NAME}_* ${DIR_NAME}
	mv ${DATA_NAME}.cpp rmi_data/
	mv ${DATA_NAME}.h rmi_data/
    fi
}


cd RMI && cargo build --release && cd ..

#echo $1

   
build_rmi_set $1.suffixarray_uint64
