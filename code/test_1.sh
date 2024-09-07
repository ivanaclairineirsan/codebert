# CUDA_VISIBLE_DEVICES=2 python run.py --output_dir=./output/300724_NVD_balance_diff \
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/NVD/context0_commit/balance/diff/train.jsonl \
#     --eval_data_file=../dataset/NVD/context0_commit/balance/diff/valid.jsonl \
#     --test_data_file=../dataset/NVD/context0_commit/balance/diff/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_nvd 2>&1 | tee output/logs/300724_test_NVD_balance_diff.log

# CUDA_VISIBLE_DEVICES=2 python run.py --output_dir=./output/300724_NVD_balance_diff \
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/NVD/context0_commit/balance/diff/train.jsonl \
#     --eval_data_file=../dataset/NVD/context0_commit/balance/diff/valid.jsonl \
#     --test_data_file=../dataset/Git/context0_commit/balance/diff/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_git 2>&1 | tee output/logs/300724_test_NVD_balance_diff_Git.log

# CUDA_VISIBLE_DEVICES=2 python run.py --output_dir=./output/300724_NVD_imbalance_diff \
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/NVD/context0_commit/imbalance/diff/train.jsonl \
#     --eval_data_file=../dataset/NVD/context0_commit/imbalance/diff/valid.jsonl \
#     --test_data_file=../dataset/NVD/context0_commit/imbalance/diff/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_nvd   2>&1 | tee output/logs/300724_test_NVD_imbalance_diff.log

# CUDA_VISIBLE_DEVICES=2 python run.py --output_dir=./output/300724_NVD_imbalance_diff \
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/NVD/context0_commit/imbalance/diff/train.jsonl \
#     --eval_data_file=../dataset/NVD/context0_commit/imbalance/diff/valid.jsonl \
#     --test_data_file=../dataset/Git/context0_commit/imbalance/diff/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_git   2>&1 | tee output/logs/300724_test_NVD_imbalance_diff_Git.log

# CUDA_VISIBLE_DEVICES=2 python run.py --output_dir=./output/300724_NVD_balance_msg\
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/NVD/context0_commit/balance/msg/train.jsonl \
#     --eval_data_file=../dataset/NVD/context0_commit/balance/msg/valid.jsonl \
#     --test_data_file=../dataset/NVD/context0_commit/balance/msg/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_nvd  2>&1 | tee output/logs/300724_test_NVD_balance_msg.log

# CUDA_VISIBLE_DEVICES=2 python run.py --output_dir=./output/300724_NVD_balance_msg\
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/NVD/context0_commit/balance/msg/train.jsonl \
#     --eval_data_file=../dataset/NVD/context0_commit/balance/msg/valid.jsonl \
#     --test_data_file=../dataset/Git/context0_commit/balance/msg/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_git  2>&1 | tee output/logs/300724_test_NVD_balance_msg_Git.log

# CUDA_VISIBLE_DEVICES=2 python run.py --output_dir=./output/300724_NVD_imbalance_msg\
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/NVD/context0_commit/imbalance/msg/train.jsonl \
#     --eval_data_file=../dataset/NVD/context0_commit/imbalance/msg/valid.jsonl \
#     --test_data_file=../dataset/NVD/context0_commit/imbalance/msg/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_nvd  2>&1 | tee output/logs/300724_test_NVD_imbalance_msg.log

# CUDA_VISIBLE_DEVICES=2 python run.py --output_dir=./output/300724_NVD_imbalance_msg\
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/NVD/context0_commit/imbalance/msg/train.jsonl \
#     --eval_data_file=../dataset/NVD/context0_commit/imbalance/msg/valid.jsonl \
#     --test_data_file=../dataset/Git/context0_commit/imbalance/msg/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_git  2>&1 | tee output/logs/300724_test_NVD_imbalance_msg_Git.log

# CUDA_VISIBLE_DEVICES=2 python run.py --output_dir=./output/300724_NVD_balance_msg_diff\
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/NVD/context0_commit/balance/msg+diff/train.jsonl \
#     --eval_data_file=../dataset/NVD/context0_commit/balance/msg+diff/valid.jsonl \
#     --test_data_file=../dataset/NVD/context0_commit/balance/msg+diff/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_nvd 2>&1 | tee output/logs/300724_test_NVD_balance_msg_diff.log

# CUDA_VISIBLE_DEVICES=2 python run.py --output_dir=./output/300724_NVD_balance_msg_diff\
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/NVD/context0_commit/balance/msg+diff/train.jsonl \
#     --eval_data_file=../dataset/NVD/context0_commit/balance/msg+diff/valid.jsonl \
#     --test_data_file=../dataset/Git/context0_commit/balance/msg+diff/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_git 2>&1 | tee output/logs/300724_test_NVD_balance_msg_diff_Git.log

# CUDA_VISIBLE_DEVICES=2 python run.py --output_dir=./output/300724_NVD_imbalance_msg_diff\
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/NVD/context0_commit/imbalance/msg+diff/train.jsonl \
#     --eval_data_file=../dataset/NVD/context0_commit/imbalance/msg+diff/valid.jsonl \
#     --test_data_file=../dataset/NVD/context0_commit/imbalance/msg+diff/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_nvd  2>&1 | tee output/logs/300724_test_NVD_imbalance_msg_diff.log

# CUDA_VISIBLE_DEVICES=2 python run.py --output_dir=./output/300724_NVD_imbalance_msg_diff\
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/NVD/context0_commit/imbalance/msg+diff/train.jsonl \
#     --eval_data_file=../dataset/NVD/context0_commit/imbalance/msg+diff/valid.jsonl \
#     --test_data_file=../dataset/Git/context0_commit/imbalance/msg+diff/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_git  2>&1 | tee output/logs/300724_test_NVD_imbalance_msg_diff_Git.log




# CUDA_VISIBLE_DEVICES=3 python run.py --output_dir=./output/300724_Git_balance_diff \
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/Git/context0_commit/balance/diff/train.jsonl \
#     --eval_data_file=../dataset/Git/context0_commit/balance/diff/valid.jsonl \
#     --test_data_file=../dataset/Git/context0_commit/balance/diff/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_git  2>&1 | tee output/logs/300724_test_Git_balance_diff.log

# CUDA_VISIBLE_DEVICES=3 python run.py --output_dir=./output/300724_Git_balance_diff \
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/Git/context0_commit/balance/diff/train.jsonl \
#     --eval_data_file=../dataset/Git/context0_commit/balance/diff/valid.jsonl \
#     --test_data_file=../dataset/NVD/context0_commit/balance/diff/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_nvd  2>&1 | tee output/logs/300724_test_Git_balance_diff_NVD.log

# CUDA_VISIBLE_DEVICES=3 python run.py --output_dir=./output/300724_Git_imbalance_diff \
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/Git/context0_commit/imbalance/diff/train.jsonl \
#     --eval_data_file=../dataset/Git/context0_commit/imbalance/diff/valid.jsonl \
#     --test_data_file=../dataset/Git/context0_commit/imbalance/diff/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_git  2>&1 | tee output/logs/300724_test_Git_imbalance_diff.log

# CUDA_VISIBLE_DEVICES=3 python run.py --output_dir=./output/300724_Git_imbalance_diff \
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/Git/context0_commit/imbalance/diff/train.jsonl \
#     --eval_data_file=../dataset/Git/context0_commit/imbalance/diff/valid.jsonl \
#     --test_data_file=../dataset/NVD/context0_commit/imbalance/diff/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_nvd  2>&1 | tee output/logs/300724_test_Git_imbalance_diff_NVD.log

# CUDA_VISIBLE_DEVICES=3 python run.py --output_dir=./output/300724_Git_balance_msg\
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/Git/context0_commit/balance/msg/train.jsonl \
#     --eval_data_file=../dataset/Git/context0_commit/balance/msg/valid.jsonl \
#     --test_data_file=../dataset/Git/context0_commit/balance/msg/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_git  2>&1 | tee output/logs/300724_test_Git_balance_msg.log

# CUDA_VISIBLE_DEVICES=3 python run.py --output_dir=./output/300724_Git_balance_msg\
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/Git/context0_commit/balance/msg/train.jsonl \
#     --eval_data_file=../dataset/Git/context0_commit/balance/msg/valid.jsonl \
#     --test_data_file=../dataset/NVD/context0_commit/balance/msg/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_nvd  2>&1 | tee output/logs/300724_test_Git_balance_msg_NVD.log

# CUDA_VISIBLE_DEVICES=3 python run.py --output_dir=./output/300724_Git_imbalance_msg\
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/Git/context0_commit/imbalance/msg/train.jsonl \
#     --eval_data_file=../dataset/Git/context0_commit/imbalance/msg/valid.jsonl \
#     --test_data_file=../dataset/Git/context0_commit/imbalance/msg/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_git  2>&1 | tee output/logs/300724_test_Git_imbalance_msg.log

# CUDA_VISIBLE_DEVICES=3 python run.py --output_dir=./output/300724_Git_imbalance_msg\
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/Git/context0_commit/imbalance/msg/train.jsonl \
#     --eval_data_file=../dataset/Git/context0_commit/imbalance/msg/valid.jsonl \
#     --test_data_file=../dataset/NVD/context0_commit/imbalance/msg/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_nvd  2>&1 | tee output/logs/300724_test_Git_imbalance_msg_NVD.log

# CUDA_VISIBLE_DEVICES=3 python run.py --output_dir=./output/300724_Git_balance_msg_diff\
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/Git/context0_commit/balance/msg+diff/train.jsonl \
#     --eval_data_file=../dataset/Git/context0_commit/balance/msg+diff/valid.jsonl \
#     --test_data_file=../dataset/Git/context0_commit/balance/msg+diff/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_git  2>&1 | tee output/logs/300724_test_Git_balance_msg_diff.log

# CUDA_VISIBLE_DEVICES=3 python run.py --output_dir=./output/300724_Git_balance_msg_diff\
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/Git/context0_commit/balance/msg+diff/train.jsonl \
#     --eval_data_file=../dataset/Git/context0_commit/balance/msg+diff/valid.jsonl \
#     --test_data_file=../dataset/NVD/context0_commit/balance/msg+diff/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_nvd  2>&1 | tee output/logs/300724_test_Git_balance_msg_diff_NVD.log

# CUDA_VISIBLE_DEVICES=3 python run.py --output_dir=./output/300724_Git_imbalance_msg_diff\
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/Git/context0_commit/imbalance/msg+diff/train.jsonl \
#     --eval_data_file=../dataset/Git/context0_commit/imbalance/msg+diff/valid.jsonl \
#     --test_data_file=../dataset/Git/context0_commit/imbalance/msg+diff/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_git  2>&1 | tee output/logs/300724_test_Git_imbalance_msg_diff.log

# CUDA_VISIBLE_DEVICES=3 python run.py --output_dir=./output/300724_Git_imbalance_msg_diff\
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/Git/context0_commit/imbalance/msg+diff/train.jsonl \
#     --eval_data_file=../dataset/Git/context0_commit/imbalance/msg+diff/valid.jsonl \
#     --test_data_file=../dataset/NVD/context0_commit/imbalance/msg+diff/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_nvd  2>&1 | tee output/logs/300724_test_Git_imbalance_msg_diff_NVD.log

# CUDA_VISIBLE_DEVICES=1 python run.py --output_dir=./output/080824_combined_imbalance_msg_diff\
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/Combined/context0_commit/imbalance/msg+diff/train.jsonl \
#     --eval_data_file=../dataset/Combined/context0_commit/imbalance/msg+diff/valid.jsonl \
#     --test_data_file=../dataset/Combined/context0_commit/imbalance/msg+diff/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_combined  2>&1 | tee output/logs/080824_test_Combined_imbalance_msg_diff.log

# CUDA_VISIBLE_DEVICES=1 python run.py --output_dir=./output/080824_combined_imbalance_msg\
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/Combined/context0_commit/imbalance/msg/train.jsonl \
#     --eval_data_file=../dataset/Combined/context0_commit/imbalance/msg/valid.jsonl \
#     --test_data_file=../dataset/Combined/context0_commit/imbalance/msg/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_combined  2>&1 | tee output/logs/080824_test_Combined_imbalance_msg.log

# CUDA_VISIBLE_DEVICES=1 python run.py --output_dir=./output/080824_combined_imbalance_diff\
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/Combined/context0_commit/imbalance/diff/train.jsonl \
#     --eval_data_file=../dataset/Combined/context0_commit/imbalance/diff/valid.jsonl \
#     --test_data_file=../dataset/Combined/context0_commit/imbalance/diff/test.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output test_combined  2>&1 | tee output/logs/080824_test_Combined_imbalance_diff.log

# CUDA_VISIBLE_DEVICES=0 python run.py --output_dir=./output/300724_Git_imbalance_msg_diff\
#     --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
#     --do_test \
#     --train_data_file=../dataset/Git/context0_commit/imbalance/msg+diff/train.jsonl \
#     --eval_data_file=../dataset/Git/context0_commit/imbalance/msg+diff/valid.jsonl \
#     --test_data_file=../../data/HJ-data/test_msg_diff.jsonl \
#     --num_train_epochs 100 \
#     --block_size 256 \
#     --train_batch_size 8 \
#     --eval_batch_size 1 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --seed 123456 --test_output manual_labelling_HJ  2>&1 | tee output/logs/300724_Git_imbalance_msg_diff_manual_labelling_HJ.log

CUDA_VISIBLE_DEVICES=0 python run.py --output_dir=./output/280824_NVD_imbalance_msg_diff\
    --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
    --do_test \
    --train_data_file=../dataset/Git/context0_commit/imbalance/msg+diff/train.jsonl \
    --eval_data_file=../dataset/Git/context0_commit/imbalance/msg+diff/valid.jsonl \
    --test_data_file=../../data/HJ-data/march_june_test_msg_diff.jsonl \
    --num_train_epochs 50 \
    --block_size 256 \
    --train_batch_size 8 \
    --eval_batch_size 1 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --seed 123456 --test_output manual_labelling_HJ_54k  2>&1 | tee output/logs/280824_NVD_imbalance_msg_diff_test_manual_54k.log
