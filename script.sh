#!/bin/bash
#script.sh


sizes=(10000 50000 100000 500000 1000000)
seeds=42
outputfile=timeData.csv


echo "Algorithm, Size, Time" > $outputfile

for size in "${sizes[@]}";
do
	for algo in thrust singlethread multithread;
	do 
			time=$("./$algo" $size $seeds 2>&1 \
				| grep "Total time in seconds" \
				| awk -F ': ' '{print $2}')

			echo "$algo, $size, $time" >> $outputfile
	done
done
