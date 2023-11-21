#!/bin/bash

cd $PWD/data/test_results

for dir in */; do tar -czvf "${dir%/}".tar.gz "$dir"; done

