iters=`seq -w 000 015`
for ii in ${iters[@]}
do
    echo "Cleaning ${ii} ..."
    echo    iter.${ii}/00.biasMD/gmx_mdrun.log
    rm      iter.${ii}/00.biasMD/gmx_mdrun.log
    #
    echo    iter.${ii}/00.biasMD/dftb_in.out
    rm      iter.${ii}/00.biasMD/dftb_in.out
    #
    echo    iter.${ii}/01.resMD/*/gmx_mdrun.log
    rm      iter.${ii}/01.resMD/*/gmx_mdrun.log
    #
    echo    iter.${ii}/01.resMD/*/dftb_in.out
    rm      iter.${ii}/01.resMD/*/dftb_in.out
done