#!/bin/bash

SAVEIFS=$IFS
IFS=$(echo -en "\n\b")

for tar in $(find "$PWD/data/test_results/" -name '*.tar.gz'); do mv "$tar" "$PWD/data/test_archive"; done

IFS=$SAVEIFS
