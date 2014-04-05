#!/bin/sh

for file in $(ls $1/*.depth);
do
    name=$(echo $file | sed 's/depth$/png/')
    echo "convert $file to $name"
    ./bin-Release/convert_dataset $file 640 480 $name
done
