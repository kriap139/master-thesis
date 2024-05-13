#!/bin/bash

max_lgb_jobs=10
n_folds=5
inner_n_folds=5
inner_shuffle=1
inner_random_state=9

default_repeats=3
large_datasets=('higgs' 'rcv1')

repeats=()
search_spaces=()
param_indexes=()
datasets=()
params_file="''"

print_exit() {
    echo "$1"
    exit 1
}

check_exit() {
    if [ $? -ne 0 ]; then
        print_exit "$1"
    fi
}

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
         for a in $OPTARG; do search_spaces+=("$a") ; done;;
      
      m) # Search Method
         method="$OPTARG";;

     \?) # Invalid option
         echo "Error: Invalid option"
         exit;;
   esac
done  


if [ -z "$method" ]; then print_exit "Search Method not specified!"; fi

if [ ! -f "$params_file" ]
then
    echo "params file dosen't exists, Params indexes reset!"
    param_indexes=()
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
elif [ ${#repeats[@]} -eq 1 ] && [ ${#datasets[@]} -gt 1 ]; then
    n_repeats=${repeats[0]}
    repeats=() && for a in "${datasets[@]}"; do repeats+=("$n_repeats") ; done
    unset n_repeats
    echo "Expanding single repeats, to all datsets: ${repeats[*]}"
elif [ ${#repeats[@]} -gt 1 ] && [ ${#datasets[@]} -ne ${#repeats[@]} ]; then
    print_exit "Number of given repeats (${#repeats[@]}) is smaller than the number of datasets(${#datasets[@]})!"
fi 



if [ ${#search_spaces[@]} -eq 0 ] && [ ${#param_indexes[@]} -eq 0 ]; then
    for dataset in "${datasets[@]}"; do search_spaces+=("all") ; done
    echo "Search space not set, and no params indexes present. Unrestrincting search spaces: ${search_spaces[*]}"

elif [ ${#search_spaces[@]} -eq 1 ] && [ ${#param_indexes[@]} -eq 0 ] && [ ${#datasets[@]} -gt 1 ]; then
    search_space=${search_spaces[0]}
    search_spaces=()
    for a in "${datasets[@]}"; do search_spaces+=("$search_space") ; done
    unset search_space
    echo "Expanding single search_space, to all datsets: ${search_spaces[*]}"

elif [ ${#search_spaces[@]} -eq 0 ] && [ ${#param_indexes[@]} -gt 0 ]; then
    spaces="$(python3 src/params_file_infer_search_space.py "$params_file" "${param_indexes[@]}" )"
    check_exit "Failed to infer search spaces: $spaces"
    search_spaces=() && for a in $spaces; do search_spaces+=("$a"); done
    echo "infered search space from params file: ${search_spaces[*]}"

elif [ ${#param_indexes[@]} -eq 0 ] && [ ${#search_spaces[@]} -gt 0 ] && [ ${#search_spaces[@]} -ne ${#datasets[@]} ]; then
    print_exit "Number of given search spaces (${#search_spaces[@]}) is less than the number of datasets (${#datasets[@]})!"

elif [ ${#search_spaces[@]} -gt 0 ] && [ ${#param_indexes[@]} -gt 0 ]; then
    echo "It is not possible to specify search space and params at the same time, as the search space will be infered from the params!"
    echo "it is possible to add a search space key to params in the params file instead!. However if stuff like k space parameters-"
    echo "are already present in the params file, then these will take precedent!"
    exit 1
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

print_search_space_info() {
    if [ "$search_space" == "all" ]
    then
        echo "search space unretricted!"
    else
        echo "restricting search space to: $search_space"
    fi
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
        search_space=${search_spaces[$i]}
        params="$(python3 src/params_file_to_cmd.py "$params_file" "$params_idx")"
        print_search_space_info
        
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
    print_search_space_info

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
        search_space=${search_spaces[$dataset_idx]}
        echo "Running search no params"
        run_search_no_params
    else
        echo "Running search with params"
        run_search_with_params
    fi
done
