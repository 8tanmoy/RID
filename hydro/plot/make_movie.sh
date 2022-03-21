arr=(  )
for jj in `seq -85 2 90`; do
	arr+=( fe40_az${jj}.png )
done
convert -delay 10 -quality 100 ${arr[@]} -loop 1 fe_movie.gif 