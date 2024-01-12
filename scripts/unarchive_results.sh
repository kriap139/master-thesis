#!/bin/bash

archive="$1"
results_dir="$PWD/data/test_results"

if [ -d "$results_dir" ]; then
    rm -r "$results_dir"
fi

mkdir -p "$results_dir"
tar -xf "$archive" -C "$results_dir"

SAVEIFS=$IFS
IFS=$(echo -en "\n\b")

for tar in $(find "$results_dir" -name '*.tar.gz'); do tar -xf "$tar" -C "$results_dir"; done

IFS=$SAVEIFS

nested_dir="$(basename $archive)"
nested_dir="$results_dir/${nested_dir%%.*}"

echo "Nested_dir: $nested_dir"
if [ -d "$nested_dir" ]; then
    rm -r "$nested_dir"
else
    find "$results_dir" -name "*.tar.gz" -type f --delete
fi

