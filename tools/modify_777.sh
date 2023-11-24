
#!bin/sh
for file in /group/30042/ruyangliu/mmaction2/data/kinetics400/videos_val/*
do
    if test -d $file
    then
        sudo chmod 777 $file
        echo $file 完成
    fi
done