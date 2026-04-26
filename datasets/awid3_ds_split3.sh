#! /bin/bash

target_dir="/mnt/vol1/pcaps/dataset_v2/npz/"
awid3_dir="/mnt/vol1/pcaps/AWID3"
pcaps_f_dir="/mnt/vol1/pcaps/pcaps_f/"
pcaps_inz_dir="/mnt/vol1/pcaps/pcaps_inz/"



echo AWID3;

$writer -i "${awid3_dir}6. Kr00k.pcap" -z "${target_dir}awid3_kr00k.npz" -r -l 1 -f "wlan subtype disassoc" -u -s 1555898
$writer -i "${awid3_dir}7. Evil_Twin.pcap" -z "${target_dir}awid3_evil_twin.npz" -r -l 2 -f "((wlan subtype beacon && less 242) || (wlan subtype disassoc || subtype deauth || subtype qos-data)) && (ip host 192.168.30.1 || wlan host c:9d:92:54:f3:35)" -u -s 1420038 -e 3778728
