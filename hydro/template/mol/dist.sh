#!/bin/bash

#rm -rf d_all.*.xvg
#rm -rf d_aver.*.xvg

d_cmd="gmx_d distance"
file=traj.trr

#cvnames=(p1oh2 p1o4 oh2h1 h1o1 #h1o2 #h1o3)

for ii in `seq -w 00 00`;
do
    code0=`echo "$ii * 4 + 0" | bc `
    code1=`echo "$ii * 4 + 1" | bc `
    code2=`echo "$ii * 4 + 2" | bc `
    code3=`echo "$ii * 4 + 3" | bc `
#    code4=`echo "$ii * 5 + 4" | bc `
#    code5=`echo "$ii * 5 + 5" | bc `
    echo doing $ii
    echo $code0 | $d_cmd -f $file -n dist.ndx -oav distave.p1oh2.$ii.xvg -oh disthist.p1oh2.$ii.xvg &> /dev/null
    echo $code1 | $d_cmd -f $file -n dist.ndx -oav distave.p1o4.$ii.xvg -oh disthist.p1o4.$ii.xvg &> /dev/null
    echo $code2 | $d_cmd -f $file -n dist.ndx -oav distave.oh2h1.$ii.xvg -oh disthist.oh2h1.$ii.xvg &> /dev/null
    echo $code3 | $d_cmd -f $file -n dist.ndx -oav distave.h1o1.$ii.xvg -oh disthist.h1o1.$ii.xvg &> /dev/null
#    echo $code4 | $d_cmd -f $file -n dist.ndx -oav distave.h1o2.$ii.xvg -oh disthist.h1o2.$ii.xvg &> /dev/null
#    echo $code5 | $d_cmd -f $file -n dist.ndx -oav distave.h1o3.$ii.xvg -oh disthist.h1o3.$ii.xvg &> /dev/null
    #all distances in a file
    printf "0\n1\n2\n3" | $d_cmd -f $file -n dist.ndx -oav distave.all.$ii.xvg -oh disthist.all.$ii.xvg &> /dev/null
    #more commands here
    #0. get rid of the headers maybe
    sed "/@/d" distave.all.00.xvg | sed "/#/d" > all.dists.temp
    #1. calculate oh2h1(y) - h1o1(x) and write bare vals in a file
    echo "@    title \"Average distance\"
@    xaxis  label \"Time (ps)\"
@    yaxis  label \"Distance (nm)\"
@TYPE xy
@ view 0.15, 0.15, 0.75, 0.85
@ legend on
@ legend box on
@ legend loctype view
@ legend 0.78, 0.8
@ legend length 2
@ s0 legend \"asym\"" > distave.asym.$ii.xvg
    awk '{print $1, $4 - $5}' all.dists.temp >> distave.asym.$ii.xvg 
    #2. get all 3 cvs in a file and name it all.dists.out. this will be set as bias_out_dist in run_mp.py line 25
    awk '{print $2,$3,$4 - $5}' all.dists.temp > all.dists.out
    awk '{print $1, $2, $3, $4 - $5}' all.dists.temp > all.$ii.out
    rm all.dists.temp
done
