#! /bin/bash

target_dir="/mnt/vol1/pcaps/dataset_v2/npz/"
awid3_dir="/mnt/vol1/pcaps/AWID3"
pcaps_f_dir="/mnt/vol1/pcaps/pcaps_f/"
pcaps_inz_dir="/mnt/vol1/pcaps/pcaps_inz/"



echo AWID3;

$writer -i "${awid3_dir}3. (Re)Assoc.pcap" -z "${target_dir}awid3_reassoc.npz" -r -l 1 -f "(wlan subtype assoc-req || subtype reassoc-req || subtype beacon) and less 301" -s 1145178 -e 1833964
$writer -i "${awid3_dir}4. Rogue_AP.pcap" -z "${target_dir}awid3_rogue_ap.npz" -r -l 2 -f "subtype beacon && less 264" -s 1198551 -e 1973111
