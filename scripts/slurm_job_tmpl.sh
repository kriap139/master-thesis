#!/bin/bash

max_lgb_jobs=10
n_repeats=3
n_folds=5
inner_n_folds=5
inner_shuffle=1
inner_random_state=9

method="$1"
datasets=()
params_file="$3"
param_indexes=()

for i in $2; do datasets+=("$i") ; done
for i in $4; do param_indexes+=("$i") ; done
last_dataset=${datasets[$(( ${#datasets[*]} - 1 ))]}
echo "datasets=[${datasets[*]}], last_dataset=${last_dataset[*]}, params_file=$params_file, param_indexes=[${param_indexes[*]}]"

if [ ! -f "$params_file" ]
then
    echo "params file dosen't exists, Params indexes reset!"
    param_indexes=()
fi

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
    python3 ./tools/search.py                   \
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
    ${params:+--params "$params"}               
}

run_search_with_params() {
    for i in "${!param_indexes[@]}"
    do
        index=${param_indexes[$i]}
        params="$(python3 tools/Util/params_file_to_cmd.py "$params_file" "$index")"
        
        # last params index
        if [ $index -eq $((${#param_indexes[@]}-1)) ] && [[ "$dataset" == "$last_dataset" ]]
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

for dataset in "${datasets[@]}"
do  
    if [ ${#param_indexes[@]} -eq 0 ]; 
    then
        echo "Running search no params"
        run_search_no_params
    else
        echo "Running search with params"
        run_search_with_params
    fi
done