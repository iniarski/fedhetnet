awid3_dir="/mnt/shared/awids/AWID3_Dataset/pcaps/"
pcaps_f_dir="/mnt/shared/pcaps_f/"
pcaps_inz_dir="/mnt/shared/pcaps_inz/"

echo pcaps_f normal;

$writer -i "${pcaps_f_dir}real01.pcap" -z "${target_dir}real01.npz" -r -n
$writer -i "${pcaps_f_dir}real10.pcap" -z "${target_dir}real10.npz" -r -n
$writer -i "${pcaps_f_dir}real15.pcap" -z "${target_dir}real15.npz" -r -n
$writer -i "${pcaps_f_dir}real30.pcap" -z "${target_dir}real30.npz" -r -n
$writer -i "${pcaps_f_dir}real40.pcap" -z "${target_dir}real40.npz" -r -n

echo pcaps_f attacks;

$writer -i "${pcaps_f_dir}beacon_flood01.pcap" -z "${target_dir}beacon_flood01.npz" -r -l 1 -f "subtype beacon && not wlan src 78:32:1b:01:1e:fa"
$writer -i "${pcaps_f_dir}beacon_flood02.pcap" -z "${target_dir}beacon_flood02.npz" -r -l 1 -f "subtype beacon && not wlan src 78:32:1b:01:1e:fa"
$writer -i "${pcaps_f_dir}beacon_flood03.pcap" -z "${target_dir}beacon_flood03.npz" -r -l 1 -f "subtype beacon && not wlan src 78:32:1b:01:1e:fa"
$writer -i "${pcaps_f_dir}deauth_amok01.pcap" -z "${target_dir}deauth_amok01.npz" -r -l 1 -f "subtype deauth || subtype disassoc"
$writer -i "${pcaps_f_dir}deauth_amok02.pcap" -z "${target_dir}deauth_amok02.npz" -r -l 1 -f "subtype deauth || subtype disassoc"
$writer -i "${pcaps_f_dir}deauth_amok03.pcap" -z "${target_dir}deauth_amok03.npz" -r -l 1 -f "subtype deauth || subtype disassoc"
$writer -i "${pcaps_f_dir}deauth_amok04.pcap" -z "${target_dir}deauth_amok04.npz" -r -l 1 -f "subtype deauth || subtype disassoc"
$writer -i "${pcaps_f_dir}deauth_target01.pcap" -z "${target_dir}deauth_target01.npz" -r -l 1 -f "(subtype deauth || subtype disassoc) && wlan host 34:a8:eb:b6:23:6d && less 83"
$writer -i "${pcaps_f_dir}deauth_target02.pcap" -z "${target_dir}deauth_target02.npz" -r -l 1 -f "(subtype deauth || subtype disassoc) && wlan host 30:b5:c2:46:b7:32 && less 83"
$writer -i "${pcaps_f_dir}deauth_target03.pcap" -z "${target_dir}deauth_target03.npz" -r -l 1 -f "subtype deauth && wlan host 34:a8:eb:b6:23:6d && less 83"
$writer -i "${pcaps_f_dir}deauth_target04.pcap" -z "${target_dir}deauth_target04.npz" -r -l 1 -f "subtype deauth && wlan host 34:a8:eb:b6:23:6d && less 83"
$writer -i "${pcaps_f_dir}deauth_target05.pcap" -z "${target_dir}deauth_target05.npz" -r -l 1 -f "subtype deauth && wlan host 18:f0:e4:ec:f3:bf && less 83"
$writer -i "${pcaps_f_dir}disass01.pcap" -z "${target_dir}disass_target01.npz" -r -l 1 -f "subtype disassoc && wlan dst 30:b5:c2:46:b7:32" -s 210539 -e 228943
$writer -i "${pcaps_f_dir}disass02.pcap" -z "${target_dir}disass_target02.npz" -r -l 1 -f "subtype disassoc && wlan dst 30:b5:c2:46:b7:32" -s 516464 -e 758306
$writer -i "${pcaps_f_dir}disass03.pcap" -z "${target_dir}disass_target03.npz" -r -l 1 -f "subtype disassoc && wlan dst 30:b5:c2:46:b7:32" -s 274021 -e 400033
$writer -i "${pcaps_f_dir}probe_req01.pcap" -z "${target_dir}probe_req01.npz" -r -l 1 -f "wlan subtype probe-req && less 107"
$writer -i "${pcaps_f_dir}probe_req02.pcap" -z "${target_dir}probe_req02.npz" -r -l 1 -f "wlan subtype probe-req && less 107"
$writer -i "${pcaps_f_dir}probe_req03.pcap" -z "${target_dir}probe_req03.npz" -r -l 1 -f "wlan subtype probe-req && less 107"
$writer -i "${pcaps_f_dir}probe_resp01.pcap" -z "${target_dir}probe_resp01.npz" -r -l 1 -f "wlan subtype probe-resp && not wlan src 78:32:1b:01:1e:fa"
$writer -i "${pcaps_f_dir}probe_resp02.pcap" -z "${target_dir}probe_resp02.npz" -r -l 1 -f "wlan subtype probe-resp && not wlan src 78:32:1b:01:1e:fa"
$writer -i "${pcaps_f_dir}fake_auth01.pcap" -z "${target_dir}fake_auth01.npz" -r 	-l 1 -f "subtype auth && wlan dst 78:32:1b:01:1e:fa && len==86" -s 441299 -e 513125
$writer -i "${pcaps_f_dir}fake_auth02.pcap" -z "${target_dir}fake_auth02.npz" -r 	-l 1 -f "subtype auth && wlan dst 78:32:1b:01:1e:fa && len==86" -s 177655 -e 419466
$writer -i "${pcaps_f_dir}fake_auth03.pcap" -z "${target_dir}fake_auth03.npz" -r 	-l 1 -f "subtype auth && wlan dst 78:32:1b:01:1e:fa && len==86" -s 178557 -e 737633
$writer -i "${pcaps_f_dir}wp3_01.pcap" -z "${target_dir}wp3_01.npz" -r -l 2 -f "subtype beacon && len==114" -s 121986 -e 611439
$writer -i "${pcaps_f_dir}wp3_02.pcap" -z "${target_dir}wp3_02.npz" -r 	-l 2 -f "wlan src bc:f6:85:03:36:5b"
$writer -i "${pcaps_f_dir}wifiphisher01.pcap" -z "${target_dir}wifiphisher01.npz" -r -l 2 -f "subtype beacon && len==158"
$writer -i "${pcaps_f_dir}wifiphisher02.pcap" -z "${target_dir}wifiphisher02.npz" -r 	-l 2 -f "(subtype beacon && len==158) || (src host 10.0.0.1 || src host 31.13.65.33) || wlan src 00:00:00:d6:01:90"
$writer -i "${pcaps_f_dir}kr00k01.pcap" -z "${target_dir}kr00k01.npz" -r -l 1 -f "subtype deauth && wlan dst 18:f0:e4:ec:f3:bf" -s 788634
$writer -i "${pcaps_f_dir}kr00k02.pcap" -z "${target_dir}kr00k02.npz" -r -l 1 -f "subtype deauth && wlan dst 18:f0:e4:ec:f3:bf" -s 902675
$writer -i "${pcaps_f_dir}kr00k03.pcap" -z "${target_dir}kr00k03.npz" -r -l 1 -f "subtype deauth && wlan dst 18:f0:e4:ec:f3:bf" -s 257084 -e 842863
$writer -i "${pcaps_f_dir}cw_mod01.pcap" -z "${target_dir}cw_mod01.npz" -r -l 3 -f "wlan src a0:f3:c1:20:d5:8a"
$writer -i "${pcaps_f_dir}cw_mod02.pcap" -z "${target_dir}cw_mod02.npz" -r -l 3 -f "wlan src a0:f3:c1:20:d5:8a"
$writer -i "${pcaps_f_dir}cw_mod03.pcap" -z "${target_dir}cw_mod03.npz" -r -l 3 -f "wlan src a0:f3:c1:20:d5:8a"
$writer -i "${pcaps_f_dir}port_scan01.pcap" -z "${target_dir}port_scan01.npz" -r -l 4 -f "wlan src a0:f3:c1:20:d5:8a && (subtype data || subtype qos-data) && less 200" -s 372996 -e 415614
$writer -i "${pcaps_f_dir}port_scan02.pcap" -z "${target_dir}port_scan02.npz" -r -l 4 -f "wlan src a0:f3:c1:20:d5:8a && (subtype data || subtype qos-data) && less 200" -s 149602 -e 185479
$writer -i "${pcaps_f_dir}port_scan03.pcap" -z "${target_dir}port_scan03.npz" -r -l 4 -f "wlan src a0:f3:c1:20:d5:8a && (subtype data || subtype qos-data) && less 200" -s 137883
$writer -i "${pcaps_f_dir}syn_flood01.pcap" -z "${target_dir}syn_flood01.npz" -r -l 4 -f "wlan src a0:f3:c1:20:d5:8a && wlan dst 78:32:1b:01:1e:fa && (subtype data || subtype qos-data) && len==146" -s 192026 -e 428191
$writer -i "${pcaps_f_dir}syn_flood02.pcap" -z "${target_dir}syn_flood02.npz" -r -l 4 -f "wlan src a0:f3:c1:20:d5:8a && wlan dst 78:32:1b:01:1e:fa && (subtype data || subtype qos-data) && len==146" -s 153449 -e 512427
