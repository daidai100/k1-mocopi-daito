#!/usr/bin/env bash
set -euo pipefail

rsync_host='vivi@[fe80::2e0:97ff:fe2b:78be%enp134s0]'
ssh_host='vivi@fe80::2e0:97ff:fe2b:78be%enp134s0'
source_archive=/mnt/storage/k1-motion/datasets/bones-seed/soma_proportional.tar.gz
remote_incomplete=/mnt/ssd512/k1-motion/dataset/soma_proportional.tar.gz.incomplete
remote_archive=/mnt/ssd512/k1-motion/dataset/soma_proportional.tar.gz

rsync --archive --partial --info=progress2 \
    -e 'ssh -6 -o BatchMode=yes -o Compression=no' \
    "$source_archive" "$rsync_host:$remote_incomplete"
ssh -6 -o BatchMode=yes "$ssh_host" "mv '$remote_incomplete' '$remote_archive'"
