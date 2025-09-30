#!/usr/bin/env bash
set -euo pipefail

# Attiva venv se presente
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
  if [ -z "${VIRTUAL_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
    echo "[INFO] Virtualenv attivata: $VIRTUAL_ENV"
  fi
fi

# Attiva hf-transfer se installato
if python -c "import hf_transfer" 2>/dev/null; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
  echo "[INFO] hf-transfer attivato (download accelerati)."
fi

model_name=TimeLLM
train_epochs=100
learning_rate=0.01
llama_layers=32

# Evitare porte privilegiate (<1024). Porta alta non usata.
master_port=29501

# Autodetect numero GPU disponibili; fallback a 1.
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
if [ -z "$GPU_COUNT" ] || [ "$GPU_COUNT" -lt 1 ]; then
  echo "[WARN] Nessuna GPU visibile. Eseguo in CPU (num_process=1)." >&2
  GPU_COUNT=1
fi

# Se hai una sola GPU non ha senso usare 8 processi: mettiamo 1.
if [ "$GPU_COUNT" -eq 1 ]; then
  num_process=1
  batch_size=24
else
  # Adatta i processi al numero GPU (puoi comunque forzarne di meno se vuoi)
  num_process=$GPU_COUNT
  batch_size=24
fi

d_model=32
d_ff=128

comment='TimeLLM-ETTh1'

accelerate launch $( [ "$num_process" -gt 1 ] && echo --multi_gpu ) --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment

accelerate launch $( [ "$num_process" -gt 1 ] && echo --multi_gpu ) --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 192 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 32 \
  --d_ff 128 \
  --batch_size $batch_size \
  --learning_rate 0.02 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment

accelerate launch $( [ "$num_process" -gt 1 ] && echo --multi_gpu ) --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 336 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --lradj 'COS'\
  --learning_rate 0.001 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment

accelerate launch $( [ "$num_process" -gt 1 ] && echo --multi_gpu ) --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 720 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment