python ./train.py \
        --max_epoch 100 \
        --dump_path ./ \
        --exp_name test \
        --exp_id 0 \
        --n_steps_per_epoch 500 \
        --collate_queue_size 10000 \
        --batch_size 512 \
        --save_periodic -1 \
        --save_periodic_from 40\
        --eval_size 500 \
        --batch_size_eval 500 \
        --num_workers 8 \
        --max_len 200 \
        --max_number_bags -1 \
        --max_input_points 300 \
        --tokens_per_batch 20000 \
        --add_consts 1 \
        --device "cuda:3" \
        --use_exprs 200000 \
        --use_dimension_mask 0 \
        --expr_train_data_path "./data/exprs_train.json" \
        --expr_valid_data_path "./data/exprs_valid.json" \
        --sub_expr_train_path "./data/exprs_seperated_train.json"\
        --sub_expr_valid_path "./data/exprs_seperated_valid.json"\
        --decode_physical_units "single-seq" \
        --use_hints "units,complexity,unarys,consts" \
        --random_variables_sequence 0 \
        --max_trials 10\
        --generate_datapoints_distribution "positive,multi"\
        --rescale 0 \