#!/bin/bash
path=`pwd`
echo "pwd: $path"

ori_bag="$path/2024-04-22-22-08-46.bag"
echo "bag: $ori_bag"

t0=`rosbag info -y -k start $ori_bag`
echo "file start time: $t0"

t_start=`echo "$t0 + 47.39" | bc -l`
echo "chosen start time: $t_start"

t_end=`echo "$t0 + 48.68" | bc -l`
echo "chosen end time: $t_end"

tar_bag="$path/08cut4.bag"
echo "target bag: $tar_bag"

rosbag filter $ori_bag $tar_bag "t.to_sec() >= $t_start and t.to_sec() <= $t_end"
# dont use t.secs, its wierd





t_start=`echo "$t0 + 41.13" | bc -l`
echo "chosen start time: $t_start"

t_end=`echo "$t0 + 42.40" | bc -l`
echo "chosen end time: $t_end"

tar_bag="$path/08cut3.bag"
echo "target bag: $tar_bag"

rosbag filter $ori_bag $tar_bag "t.to_sec() >= $t_start and t.to_sec() <= $t_end"





t_start=`echo "$t0 + 34.82" | bc -l`
echo "chosen start time: $t_start"

t_end=`echo "$t0 + 36.08" | bc -l`
echo "chosen end time: $t_end"

tar_bag="$path/08cut2.bag"
echo "target bag: $tar_bag"

rosbag filter $ori_bag $tar_bag "t.to_sec() >= $t_start and t.to_sec() <= $t_end"





t_start=`echo "$t0 + 18.59" | bc -l`
echo "chosen start time: $t_start"

t_end=`echo "$t0 + 19.81" | bc -l`
echo "chosen end time: $t_end"

tar_bag="$path/08cut1.bag"
echo "target bag: $tar_bag"

rosbag filter $ori_bag $tar_bag "t.to_sec() >= $t_start and t.to_sec() <= $t_end"
