#!/bin/bash

#rm -rf d_all.*.xvg
#rm -rf d_aver.*.xvg

d_cmd="gmx_d distance"
file=traj.trr

#cvnames=(p1oh2 p1o4 oh2h1 h1o1 #h1o2 #h1o3)

for ii in `seq -w 00 00`;
do
    code0=`echo "$ii * 2 + 0" | bc `
    code1=`echo "$ii * 2 + 1" | bc `
    echo doing $ii
    echo $code0 | $d_cmd -f $file -n dist.ndx -oav distave.poatt.$ii.xvg -oh disthist.poatt.$ii.xvg &> /dev/null
    echo $code1 | $d_cmd -f $file -n dist.ndx -oav distave.polg.$ii.xvg -oh disthist.polg.$ii.xvg &> /dev/null
    #all distances in a file
    printf "0\n1" | $d_cmd -f $file -n dist.ndx -oav distave.all.$ii.xvg -oh disthist.all.$ii.xvg &> /dev/null
    #more commands here
    #0. get rid of the headers maybe
    sed "/@/d" distave.all.00.xvg | sed "/#/d" > all.dists.temp
    #2. get all 3 cvs in a file and name it all.dists.out. this will be set as bias_out_dist in run_mp.py line 25
    awk '{print $2,$3}' all.dists.temp > all.dists.out
    awk '{print $1, $2, $3}' all.dists.temp > all.$ii.out
    rm all.dists.temp
done
