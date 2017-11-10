mfccdir=mfcc40_23/$1
nj=20
wav=data/$1/wav.scp
line_total=`wc -l $wav | awk '{print $1}'`
echo $line_total
chunk_size=$(( $line_total / $nj))

current_row=1
i=1
mkdir -p $mfccdir/wav
mkdir -p $mfccdir/feats
while [ $current_row -lt $line_total ]; do
    last_row=$(( $current_row + $chunk_size - 1 ))
    echo $current_row, $last_row
    if [ $i -eq $nj ]
    then
	last_row=$line_total
    fi
    sed -n ''"$current_row"','"$last_row"'p' $wav  > $mfccdir/wav/wav.$i.scp
    current_row=$(( $last_row+1 ))
    i=$(( $i+1 ))
done


for(( i=1; i<=$nj; i++))
do
    echo "Processing wav.$i.scp"
    compute-mfcc-feats --num-ceps=40 --num-mel-bins=40 \
		       --low-freq=20 --high-freq=-400 --use-energy=false \
		       --verbose=2 scp,p:$mfccdir/wav/wav.$i.scp  ark:- |  \
    add-deltas ark:- ark:- | \
    splice-feats --left-context=11 --right-context=11  \
		 ark:- ark:$mfccdir/feats/expand_feats.$i.ark
done
