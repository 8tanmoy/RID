#!/bin/bash

#./tools/pbc.angle.py plm.res.out pbc.res.out

msp_avg -f plm.res.out -m 2,3 -n 8 -t .9 | grep -v \# > avgins.centers.out          #pbc->plm change -m

grep KAPPA plumed.res.dat  | awk '{print $4}' | cut -d '=' -f 2 > kappa.out             #ok

./tools/cmpf.py

cat force.out | tr '\n' ' ' > tmp.out
mv -f tmp.out force.out
echo "" >> force.out

cat ferror.out | tr '\n' ' ' > tmp.out
mv -f tmp.out ferror.out
echo "" >> ferror.out
