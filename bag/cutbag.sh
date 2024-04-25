#!/bin/bash
path=`pwd`
echo "pwd: $path"

ori_bag="$path/2024-04-22-22-04-18.bag"
echo "bag: $ori_bag"

t0=`rosbag info -y -k start $ori_bag`
echo "file start time: $t0"

t_start=`echo "$t0 + 18.3" | bc -l`
echo "chosen start time: $t_start"

t_end=`echo "$t0 + 19.31" | bc -l`
echo "chosen end time: $t_end"

tar_bag="$path/04cut.bag"
echo "target bag: $tar_bag"

rosbag filter $ori_bag $tar_bag "t.to_sec() >= $t_start and t.to_sec() <= $t_end"
# dont use t.secs, its wierd
