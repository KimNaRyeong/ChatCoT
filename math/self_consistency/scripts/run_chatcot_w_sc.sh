#!/bin/bash

DATA_SPLIT=counting_and_probability
RESULT_FOLDER=result/math_cp
NUMBER_PER_PROCESS=100

date

for ((i=1;i<=5;i++)) do
{
	startidx=$(((i-1)*NUMBER_PER_PROCESS+1+60))
	endidx=$((i*NUMBER_PER_PROCESS))
	echo python chatcot_w_sc.py \
		--result_path $RESULT_FOLDER/w_sc/turbo-w_sc-5shot-$startidx-$endidx.json \
		--start_prob $startidx \
		--end_prob $endidx \
		--write_mode w \
		--dataset_name math \
		--num_examplar 5 \
		--num_retri 2 \
		--demo_path demo/math.json \
		--data_split $DATA_SPLIT \
		--api_key_idx $i
}&
done

wait

date

python merge.py --result_folder $RESULT_FOLDER/w_sc --target_path $RESULT_FOLDER/w_sc/turbo-w_sc-5shot.json
python evaluate.py --result_path $RESULT_FOLDER/w_sc/turbo-w_sc-5shot.json