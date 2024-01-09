#!/bin/bash

archive="results_archive"
mkdir $archive

SAVEIFS=$IFS
IFS=$(echo -en "\n\b")

for tar in $(find "$PWD/data/test_results/" -name '*.tar.gz'); do mv "$tar" "$archive"; done

IFS=$SAVEIFS

tar -czvf "$archive.tar.gz" $archive
