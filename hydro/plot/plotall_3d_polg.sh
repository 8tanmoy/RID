#!/bin/bash
echo "min number of iterations in three digs:"
read miniter
echo "miniter: $miniter"
echo "max number of iterations in three digs:"
read maxiter
echo "maxiter: $maxiter"
#1. sort using assm coordinate
for ii in `seq -w $miniter $maxiter`; do
    echo $ii
    dfold=data_polg_$ii
    mkdir $dfold
    cd $dfold
#
    echo $PWD
    sort -k2 -n ../../iter.$ii/02.train/fe.out > fe.out.sort2
#
    cd ../
done

reso=80
echo "resolution is $reso"
#2. separate the files
for ii in `seq -w $miniter $maxiter`; do
    echo "sorting $ii"
    cd $dfold
#
    awk '{print 10*$1, 10*$3, $4/(-4.184), 10*$2}' fe.out.sort2 > temp00                        #flipping 2 and 1 because 1 is P-Oattack, 2 is P-Olg
    femin=`sort -nk3 temp00 | head -1 | cut -d ' ' -f3`
    awk -v femin="$femin" '{print $1, $2, $3 - femin, $4}' temp00 > fe.out.sort2.min
    rm temp00
    for jj in `seq 0 $((reso - 1))`; do
        beg=$(( jj * reso * reso  + 1 ))
        end=$(( (jj + 1) * reso * reso ))
        sed -n "${beg},${end}p" fe.out.sort2.min > fe.out.sort2.min.$((jj + 1))
    done
#
    cd ../
done

#3. generate wolframscript for plotting
#module load mathematica
for ii in `seq -w $miniter $maxiter`; do
    echo "writing wl $ii"
    cd $dfold
#
    for jj in `seq 1 $reso`; do
        fname=fe.out.sort2.min.$jj
        imname=img_iter_${ii}_${jj}.png
        sed "s|_FILENAME_|$fname|g" ../plot_master_polg.wl > temp1
        sed "s|_ITER_|$ii|g" temp1 > temp2
        sed "s|_IMGNAME_|$imname|g" temp2 > plot_$jj.wl
        rm temp1 temp2
        wolframscript -file plot_$jj.wl
    done
#
    cd ../
done

#4. make movie from the images
for ii in `seq -w $miniter $maxiter`; do
    echo "generating movie in iter $ii"
    cd $dfold
#
    arr=(  )
    for jj in `seq 1 $reso`; do
        arr+=( img_iter_${ii}_${jj}.png )
    done
    convert -delay 20 -quality 100 ${arr[@]} -loop 1 fe_movie_iter_${ii}_polg.gif
#
    cd ../
done
