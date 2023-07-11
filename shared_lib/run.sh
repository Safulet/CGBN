#!/bin/bash

./main $1 > log &
echo "RUN: ./main $1"
cuda_pid=$(pidof ./main)
echo "PID: $cuda_pid"
taskset -cp 0 $cuda_pid
