imgdir=`echo $PWD`
for ii in `seq -w 000 030`; do
    wrk=../iter.$ii/02.train/
    cp plot_2d.py $wrk
    cd $wrk
    python3 plot_2d.py -m graph* -n 60
    sleep 2
    cat fe.out | wc -l
    cd $imgdir
done
