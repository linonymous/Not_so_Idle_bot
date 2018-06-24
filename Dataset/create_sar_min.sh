#!/bin/bash
echo "Generating min sar"
YESTERDATE=`date --date='yesterday' +%Y-%m-%d-`
YESTERDAY=`date --date='yesterday' +%a`
DATE=`date +%Y-%m-%d-`
DAY=`date +%a`
GENERAL="_min"
BASE_PATH=`pwd`"/"
FILE_NAME=$BASE_PATH$DATE$DAY$GENERAL
YESTERDAY_FILENAME=$YESTERDATE$YESTERDAY$GENERAL
if [ ! -d $BASE_PATH"CPU_STAT" ]; then
	mkdir $BASE_PATH"CPU_STAT"
fi
if [ ! -d $BASE_PATH"IO_STAT" ]; then
        mkdir $BASE_PATH"IO_STAT"
fi
if [ ! -d $BASE_PATH"MEM_STAT" ]; then
        mkdir $BASE_PATH"MEM_STAT"
fi
if [ ! -d $BASE_PATH"NET_STAT" ]; then
        mkdir $BASE_PATH"NET_STAT"
fi
CPU_CSV_PATH=$BASE_PATH"CPU_STAT/"$YESTERDAY_FILENAME".csv"
NET_CSV_PATH=$BASE_PATH"NET_STAT/"$YESTERDAY_FILENAME".csv"
MEM_CSV_PATH=$BASE_PATH"MEM_STAT/"$YESTERDAY_FILENAME".csv"
IO_CSV_PATH=$BASE_PATH"IO_STAT/"$YESTERDAY_FILENAME".csv"
#sar -o $FILE_NAME -u 60 1440
#Calculate the yesterday's CPU stat in  local time
sadf -d $BASE_PATH$YESTERDAY_FILENAME | tr ';' ',' > $CPU_CSV_PATH
#Calculate the yesterday's MEM stat in local time
sadf -d $BASE_PATH$YESTERDAY_FILENAME -- -r | tr ";" "," > $MEM_CSV_PATH
#Calculate the yesterday's IO stat in local time
sadf -d $BASE_PATH$YESTERDAY_FILENAME -- -b | tr ";" "," > $IO_CSV_PATH
#Calculate the yesterday's NET stat in local time
sadf -d $BASE_PATH$YESTERDAY_FILENAME -- -n DEV | tr ";" "," > $NET_CSV_PATH
