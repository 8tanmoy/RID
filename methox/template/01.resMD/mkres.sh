#!/bin/bash

cv_dim=2
targets=$*
kk=400000

if test $# -ne 2; then                                                  #tanmoy D1003
    echo should input 2 dist                                            #tanmoy D1003
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
    #tanmoy D1003 removing $pia=`printf %02d $iangle and further use
    if test $mod -eq 0; then
    sed -i "/res-poatt/s/AT=.*/AT=$ii/g" plumed.res.dat
    elif test $mod -eq 1; then
    sed -i "/res-polg/s/AT=.*/AT=$ii/g" plumed.res.dat
    fi
    count=$(($count+1))
done

sed -i "s/ARG=poatt KAPPA=.* /ARG=poatt KAPPA=${kk} /g" plumed.res.dat
sed -i "s/ARG=polg KAPPA=.* /ARG=polg KAPPA=${kk} /g" plumed.res.dat

echo $centers > centers.out
