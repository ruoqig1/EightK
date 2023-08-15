#!/usr/bin/env bash
#echo "$1G"
sinteractive -p interactive --time=01:00:00 --cpus-per-task=8 --mem "$1G"
