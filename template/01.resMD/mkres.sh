#!/bin/bash

cv_dim=3
targets=$*
kk_p1oh2=400000
kk_p1o4=400000
kk_asym=400000


if test $# -ne 3; then                                                  #tanmoy D1003
    echo should input 3 angle in the unit of rad                        #tanmoy D1003
    exit
fi

rm -f plumed.res.dat
cp plumed.res.templ plumed.res.dat

count=0
centers=""
for ii in $targets; 
do
    centers="$centers $ii"
    mod=`echo "$count % $cv_dim" | bc `
    #teanmoy D1003 removing $pia=`printf %02d $iangle and further use
    if test $mod -eq 0; then
    sed -i "/res-p1oh2/s/AT=.*/AT=$ii/g" plumed.res.dat
    elif test $mod -eq 1; then
    sed -i "/res-p1o4/s/AT=.*/AT=$ii/g" plumed.res.dat
    elif test $mod -eq 2; then
    sed -i "/res-asym/s/AT=.*/AT=$ii/g" plumed.res.dat
    fi
    count=$(($count+1))
done

sed -i "s/ARG=p1oh2 KAPPA=.* /ARG=p1oh2 KAPPA=${kk_p1oh2} /g" plumed.res.dat
sed -i "s/ARG=p1o4 KAPPA=.* /ARG=p1o4 KAPPA=${kk_p1o4} /g" plumed.res.dat
sed -i "s/ARG=asym KAPPA=.* /ARG=asym KAPPA=${kk_asym} /g" plumed.res.dat

echo $centers > centers.out
