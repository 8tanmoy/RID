#!/bin/bash
module load mathematica
wrk=/projectnb/cui-buchem/tanmoy/projects/RL/methox/try7/
for ii in `seq -w 000 030`
do
    awk '{print 10*$1, 10*$2, $3/(-4.184)}' $wrk/iter.$ii/02.train/fe.out > ./fe_$ii.out #keep it straight 1,2,3
    fname=fe_$ii.out
    imname=$PWD/img_$ii.png
    sed "s|_FILENAME_|$fname|g" plot_master.wl > temp1
    sed "s|_ITER_|$ii|g" temp1 > temp2
    sed "s|_IMGNAME_|$imname|g" temp2 > plot_$ii.wl
    rm temp1 temp2
    wolframscript -file plot_$ii.wl
    sleep 1
done
