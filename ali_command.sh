mkdir -p ali
mkdir -p dev_ali
ali-to-pdf exp/tri3_ali/final.mdl "ark:gunzip -c exp/tri3_ali/ali.*.gz |" ark,t:ali/ali.ark
ali-to-pdf exp/tri3_dev_ali/final.mdl "ark:gunzip -c exp/tri3_dev_ali/ali.*.gz |" ark,t:dev_ali/ali.ark
