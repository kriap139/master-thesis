#!/bin/bash

max_lgb_jobs=10
n_folds=5
inner_n_folds=5
inner_shuffle=1
inner_random_state=9
default_repeats=3
large_datasets=('higgs' 'rcv1')

repeats=()
param_indexes=()
datasets=()
params_file="''"
search_space="all"

# Get the options
while getopts ":f:i:d:s:m:r:" option; do
   case $option in
      d) # datasets
         for a in $OPTARG; do datasets+=("$a") ; done
         last_dataset=${datasets[$(( ${#datasets[*]} - 1 ))]};;
      
      i) # param indexes
         for a in $OPTARG; do param_indexes+=("$a") ; done;;
      
      r) # Repeats
         for a in $OPTARG; do repeats+=("$a") ; done;;
      
      f) # params file
         params_file="$OPTARG";;
      
      s) # Restrict search space
         search_space="$OPTARG";;
      
      m) # Search Method
         method="$OPTARG";;

     \?) # Invalid option
         echo "Error: Invalid option"
         exit;;
   esac
done  

if [ -z "$method" ]
then
    echo "Search Method not specified!"
    exit 1
fi

if [ ${#repeats[@]} -eq 0 ]
then
    echo "repeats array is empty, using defaults"
    for dataset in "${datasets[@]}"
    do
        if [[ ${large_datasets[*]} =~ $dataset ]]
        then
            repeats+=("0")
        else
            repeats+=("$default_repeats")
        fi
    done
fi 

if [ "$search_space" == "all" ]
then
    echo "search space unretricted!"
    unset search_space
else
    echo "restricting search space to: [$search_space]"
fi

if [ ! -f "$params_file" ]
then
    echo "params file dosen't exists, Params indexes reset!"
    param_indexes=()
fi

echo "datasets=[${datasets[*]}], last_dataset=${last_dataset[*]}, repeats=[${repeats[*]}], params_file=$params_file, param_indexes=[${param_indexes[*]}]"

set_copy() {
    unset move
    copy_logs=1
}

set_move() {
     move=1
    unset copy_logs
}

run_search() {
    #echo "method: $method, dataset: $dataset, inner_shuffle: ${inner_shuffle:+--inner-shuffle}, params: ${params:+--params "$params"}"
    #echo "move: ${move:+--move-slurm-logs}, copy_logs: ${copy_logs:+--copy-new-slurm-log-lines}"
    #exit 0
    python3 ./src/search.py                   \
    ${move:+--move-slurm-logs}                  \
    ${copy_logs:+--copy-new-slurm-log-lines}    \
    --method "$method"                          \
    --max-lgb-jobs $max_lgb_jobs                \
    --n-repeats $n_repeats                      \
    --n-folds $n_folds                          \
    --inner-n-folds $inner_n_folds              \
    ${inner_shuffle:+--inner-shuffle}           \
    --inner-random-state $inner_random_state    \
    --dataset "$dataset"                        \
    ${params:+--params "$params"}               \
    ${search_space:+--search-space "$search_space"}              
}

run_search_with_params() {
    for i in "${!param_indexes[@]}"
    do
        params_idx=${param_indexes[$i]}
        params="$(python3 src/Util/params_file_to_cmd.py "$params_file" "$params_idx")"
        
        # last params index
        if [ $i -eq $((${#param_indexes[@]}-1)) ] && [[ "$dataset" == "$last_dataset" ]]
        then
            set_move
        # params index not the last inde, so copy logs
        else
            set_copy
        fi

        run_search
    done
}

run_search_no_params() {
    # last params index
    if [[ $dataset == "$last_dataset" ]]
    then
        set_move
    # params index not the last inde, so copy logs
    else
        set_copy
    fi

    run_search
}

for dataset_idx in "${!datasets[@]}"
do  
    dataset=${datasets[$dataset_idx]}
    n_repeats=${repeats[$dataset_idx]}
    
    if [ ${#param_indexes[@]} -eq 0 ]; 
    then
        echo "Running search no params"
        run_search_no_params
    else
        echo "Running search with params"
        run_search_with_params
    fi
done
