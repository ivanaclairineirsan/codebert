# CUDA_VISIBLE_DEVICES=0 python run.py --output_dir=./output/300724_NVD_balance_diff \
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_train \
#     --do_eval \
#     --train_data_file=../dataset/NVD/context0_commit/balance/diff/train.jsonl \
#     --eval_data_file=../dataset/NVD/context0_commit/balance/diff/valid.jsonl \
#     --test_data_file=../dataset/NVD/context0_commit/balance/diff/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 16 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456  2>&1 | tee output/logs/300724_train_NVD_balance_diff.log

# CUDA_VISIBLE_DEVICES=0 python run.py --output_dir=./output/300724_NVD_imbalance_diff \
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_train \
#     --do_eval \
#     --train_data_file=../dataset/NVD/context0_commit/imbalance/diff/train.jsonl \
#     --eval_data_file=../dataset/NVD/context0_commit/imbalance/diff/valid.jsonl \
#     --test_data_file=../dataset/NVD/context0_commit/imbalance/diff/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 16 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456  2>&1 | tee output/logs/300724_train_NVD_imbalance_diff.log

# CUDA_VISIBLE_DEVICES=0 python run.py --output_dir=./output/300724_NVD_balance_msg\
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_train \
#     --do_eval \
#     --train_data_file=../dataset/NVD/context0_commit/balance/msg/train.jsonl \
#     --eval_data_file=../dataset/NVD/context0_commit/balance/msg/valid.jsonl \
#     --test_data_file=../dataset/NVD/context0_commit/balance/msg/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 16 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456  2>&1 | tee output/logs/300724_train_NVD_balance_msg.log

# CUDA_VISIBLE_DEVICES=0 python run.py --output_dir=./output/300724_NVD_imbalance_msg\
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_train \
#     --do_eval \
#     --train_data_file=../dataset/NVD/context0_commit/imbalance/msg/train.jsonl \
#     --eval_data_file=../dataset/NVD/context0_commit/imbalance/msg/valid.jsonl \
#     --test_data_file=../dataset/NVD/context0_commit/imbalance/msg/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 16 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456  2>&1 | tee output/logs/300724_train_NVD_imbalance_msg.log

# CUDA_VISIBLE_DEVICES=0 python run.py --output_dir=./output/300724_NVD_balance_msg_diff\
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_train \
#     --do_eval \
#     --train_data_file=../dataset/NVD/context0_commit/balance/msg+diff/train.jsonl \
#     --eval_data_file=../dataset/NVD/context0_commit/balance/msg+diff/valid.jsonl \
#     --test_data_file=../dataset/NVD/context0_commit/balance/msg+diff/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 16 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456  2>&1 | tee output/logs/300724_train_NVD_balance_msg_diff.log

# CUDA_VISIBLE_DEVICES=0 python run.py --output_dir=./output/300724_NVD_imbalance_msg_diff\
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_train \
#     --do_eval \
#     --train_data_file=../dataset/NVD/context0_commit/imbalance/msg+diff/train.jsonl \
#     --eval_data_file=../dataset/NVD/context0_commit/imbalance/msg+diff/valid.jsonl \
#     --test_data_file=../dataset/NVD/context0_commit/imbalance/msg+diff/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 16 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456  2>&1 | tee output/logs/300724_train_NVD_imbalance_msg_diff.log

# RUN CODEBERT on selected negative sample data
CUDA_VISIBLE_DEVICES=0 python run.py --output_dir=./output/120824_combined_simneg_imbalance_diff \
    --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --do_eval \
    --train_data_file=../dataset/Combined/context0_commit/imbalance/diff/train_simneg.jsonl \
    --eval_data_file=../dataset/Combined/context0_commit/imbalance/diff/valid_simneg.jsonl \
    --test_data_file=../dataset/Combined/context0_commit/imbalance/diff/test.jsonl \
    --num_train_epochs 100 \
    --block_size 256 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --seed 123456  2>&1 | tee output/logs/120824_train_combined_simneg_imbalance_diff.log

CUDA_VISIBLE_DEVICES=0 python run.py --output_dir=./output/120824_combined_simneg_imbalance_msg\
    --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --do_eval \
    --train_data_file=../dataset/Combined/context0_commit/imbalance/msg/train_simneg.jsonl \
    --eval_data_file=../dataset/Combined/context0_commit/imbalance/msg/valid_simneg.jsonl \
    --test_data_file=../dataset/Combined/context0_commit/imbalance/msg/test.jsonl \
    --num_train_epochs 100 \
    --block_size 256 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --seed 123456  2>&1 | tee output/logs/120824_train_combined_simneg_imbalance_msg.log

CUDA_VISIBLE_DEVICES=0 python run.py --output_dir=./output/120824_combined_simneg_imbalance_msg_diff\
    --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --do_eval \
    --train_data_file=../dataset/Combined/context0_commit/imbalance/msg+diff/train_simneg.jsonl \
    --eval_data_file=../dataset/Combined/context0_commit/imbalance/msg+diff/valid_simneg.jsonl \
    --test_data_file=../dataset/Combined/context0_commit/imbalance/msg+diff/test.jsonl \
    --num_train_epochs 100 \
    --block_size 256 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --seed 123456  2>&1 | tee output/logs/120824_train_combined_simneg_imbalance_msg_diff.log