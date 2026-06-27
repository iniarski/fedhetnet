#! /bin/bash

target_dir="/mnt/vol1/pcaps/dataset_v2/npz/"
awid3_dir="/mnt/shared/awids/AWID3_Dataset/pcaps/"
# awid3_dir="/mnt/vol1/pcaps/AWID3"
pcaps_f_dir="/mnt/vol1/pcaps/pcaps_f/"
pcaps_inz_dir="/mnt/vol1/pcaps/pcaps_inz/"



echo AWID3;

$writer -i "${awid3_dir}1. Deauth.pcap" -z "${target_dir}awid3_deauth.npz" -r -l 1 -f "wlan subtype deauth || wlan subtype disassoc" -u -s 1088022 -e 1626254
$writer -i "${awid3_dir}2. Disass.pcap" -z "${target_dir}awid3_disass.npz" -r -l 1 -f "wlan subtype deauth || wlan subtype disassoc" -u -s 1404237 -e 2013346

# read_numpy /mnt/shared/awids/AWID3_Dataset_CSV/npz/botnet/
# read_numpy /mnt/shared/awids/AWID3_Dataset_CSV/npz/ssh/
# read_numpy /mnt/shared/awids/AWID3_Dataset_CSV/npz/krack/