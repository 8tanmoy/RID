arr=(  )
for ii in `seq -w 001 010`; do
    arr+=( img_${ii}.png )
done
convert -delay 50 -quality 100 ${arr[@]} -loop 1 fe_movie.gif
