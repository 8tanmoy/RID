imgdir=`echo $PWD`
#
echo "min number of iterations in three digs:"
read miniter
echo "miniter: $miniter"
echo "max number of iterations in three digs:"
read maxiter
echo "maxiter: $maxiter"
#
for ii in `seq -w $miniter $maxiter`; do
    echo "doing $ii"
    wrk=../iter.$ii/02.train/
    cp plot_3d.py $wrk
    cd $wrk
    python3 plot_3d.py -m graph* -n 80
    sleep 2
    cat fe.out | wc -l
    cd $imgdir
done
