#! /bin/bash

target_dir="/mnt/vol1/pcaps/dataset_v2/npz/"
awid3_dir="/mnt/vol1/pcaps/AWID3"
pcaps_f_dir="/mnt/vol1/pcaps/pcaps_f/"
pcaps_inz_dir="/mnt/vol1/pcaps/pcaps_inz/"



echo AWID3;

$writer -i "${awid3_dir}1. Deauth.pcap" -z "${target_dir}awid3_deauth.npz" -r -l 1 -f "wlan subtype deauth || wlan subtype disassoc" -u -s 1088022 -e 1626254
$writer -i "${awid3_dir}2. Disass.pcap" -z "${target_dir}awid3_disass.npz" -r -l 1 -f "wlan subtype deauth || wlan subtype disassoc" -u -s 1404237 -e 2013346
$writer -i "${awid3_dir}3. (Re)Assoc.pcap" -z "${target_dir}awid3_reassoc.npz" -r -l 1 -f "(wlan subtype assoc-req || subtype reassoc-req || subtype beacon) and less 301" -s 1145178 -e 1833964
$writer -i "${awid3_dir}4. Rogue_AP.pcap" -z "${target_dir}awid3_rogue_ap.npz" -r -l 2 -f "subtype beacon && less 264" -s 1198551 -e 1973111
$writer -i "${awid3_dir}6. Kr00k.pcap" -z "${target_dir}awid3_kr00k.npz" -r -l 1 -f "wlan subtype disassoc" -u -s 1555898
$writer -i "${awid3_dir}7. Evil_Twin.pcap" -z "${target_dir}awid3_evil_twin.npz" -r -l 2 -f "((wlan subtype beacon && less 242) || (wlan subtype disassoc || subtype deauth || subtype qos-data)) && (ip host 192.168.30.1 || wlan host c:9d:92:54:f3:35)" -u -s 1420038 -e 3778728
# $writer -i "${awid3_dir}8. SQL_Injection_decrypted.pcap" -z "${target_dir}awid3_sql_injection.npz" -r -l 4 -f "ip host 192.168.2.248" -s 1484772 -e 2589042
# $writer -i "${awid3_dir}9. SSH_decrypted.pcap" -z "${target_dir}awid3_ssh.npz" -r -l 4 -f "ip host 192.168.2.248" -s 1356014 -e 2440389
# $writer -i "${awid3_dir}10. Malware_decrypted.pcap" -z "${target_dir}awid3_malware.npz" -r -l 4 -f "(ip host 192.168.2.248 || ip host 192.168.2.42 || ip host 192.168.2.73 || ip host 192.168.2.41 || ip host 192.168.2.254 || ip host 192.168.2.184 || ip host 192.168.2.190) && ip host 192.168.2.130" -s 1484772 -e 2589042
# $writer -i "${awid3_dir}12. Botnet_dtshark -r ecrypted.pcap" -z "${target_dir}awid3_botnet.npz" -r -l 4 -f "ip host 192.168.2.248 && (ip host 192.168.2.130 || ip host 192.168.2.1 || ip host 192.168.2.125 || ip host 192.168.2.42 || ip host 192.168.2.184 || ip host 192.168.2.73)" -s 1135096 -e 3325479
# $writer -i "${awid3_dir}13. Website_spoofing_decrypted.pcap" -z "${target_dir}awid3_website_spoofing.npz" -r -l 4 -f "(wlan src 04:ed:33:e0:24:82 || wlan dst 04:ed:33:e0:24:82 || wlan src 00:C0:CA:A8:29:56 || wlan dst 00:C0:CA:A8:29:56 || wlan src 24:F5:A2:EA:86:C3 || wlan dst 24:F5:A2:EA:86:C3 || wlan src 00:C0:CA:A8:26:3E || wlan dst 00:C0:CA:A8:26:3E) && (wlan subtype data || wlan subtype qos-data)" -s 16409 -e 2668582

# read_numpy /mnt/vol1/datasets/awid2/npz/rtrn/1.npz
