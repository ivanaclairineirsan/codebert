# RUN CODEBERT
CUDA_VISIBLE_DEVICES=1 python run.py --output_dir=./output/130824_combined_imbalance_msg\
    --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --do_eval \
    --train_data_file=../dataset/Combined/context0_commit/imbalance/msg/train.jsonl \
    --eval_data_file=../dataset/Combined/context0_commit/imbalance/msg/valid.jsonl \
    --test_data_file=../dataset/Combined/context0_commit/imbalance/msg/test.jsonl \
    --num_train_epochs 100 \
    --block_size 256 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --seed 123456  2>&1 | tee output/logs/130824_train_combined_imbalance_msg.log

CUDA_VISIBLE_DEVICES=1 python run.py --output_dir=./output/130824_combined_imbalance_msg_diff\
    --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --do_eval \
    --train_data_file=../dataset/Combined/context0_commit/imbalance/msg+diff/train.jsonl \
    --eval_data_file=../dataset/Combined/context0_commit/imbalance/msg+diff/valid.jsonl \
    --test_data_file=../dataset/Combined/context0_commit/imbalance/msg+diff/test.jsonl \
    --num_train_epochs 100 \
    --block_size 256 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --seed 123456  2>&1 | tee output/logs/130824_train_combined_imbalance_msg_diff.log

CUDA_VISIBLE_DEVICES=1 python run.py --output_dir=./output/130824_combined_imbalance_diff \
    --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --do_eval \
    --train_data_file=../dataset/Combined/context0_commit/imbalance/diff/train.jsonl \
    --eval_data_file=../dataset/Combined/context0_commit/imbalance/diff/valid.jsonl \
    --test_data_file=../dataset/Combined/context0_commit/imbalance/diff/test.jsonl \
    --num_train_epochs 100 \
    --block_size 256 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --seed 123456  2>&1 | tee output/logs/130824_train_combined_imbalance_diff.log