for i in train_small dev_small; #train dev test;
do
    ./feature.sh $i
done
