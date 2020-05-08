#!/bin/bash

# Gives key processor information:
/proc/cpuinfo

# total cores using cpu
grep processor /proc/cpuinfo

# Which core is assigned to each socket
grep 'physical id' /proc/cpuinfo

# ibrun is better than mpirun because it accounts for the way cores are numbered.
# /proc & /sys are not real file systems. They are just interfaces to linux kernel data structures.

# Get memory info:
/proc/meminfo

# For GPUs:
nvidia_smi

# Finding cache information
/sys/devices/system/cpu

# L1 & L2 cache are per core. L3 is shared between all cores in the socket.

# SCSI: common interface for mounting peripherals such as hard drives or SSD.
less /proc/scsi/scsi

# Lists mounted file systems:
more /etc/mtab

# df provides info on file system usage
df -h

# Finding network info
/sbin/ip link

# Find os and kernel info:
uname -a

# Get the linux distribution
cat /etc/centos-release

# top & htop DO NOT give any information on the source code level. It is only on the overall system levels.
# top only gives info about the processes (Not threads)
# To toggle thread display type "H" while top is running.
# is ordered by CPU usage, type "M" to order by memory usage.

# For a specific user:
top -u $USER

# Update info every n seconds:
top -d n


