#!/usr/bin/env bash
set -euo pipefail

# Attiva la venv locale se esiste ed è disponibile
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
  # Evita ri-attivazione se già dentro
  if [ -z "${VIRTUAL_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
    echo "[INFO] Virtualenv attivata: $VIRTUAL_ENV"
  else
    echo "[INFO] Virtualenv già attiva: $VIRTUAL_ENV"
  fi
else
  echo "[WARN] Nessuna venv trovata in ./venv (continuo con Python di sistema)."
fi

# Abilita download accelerato Hugging Face se hf-transfer installato
if python -c "import hf_transfer" 2>/dev/null; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
  echo "[INFO] hf-transfer attivato (download accelerati)."
else
  echo "[INFO] hf-transfer non installato (pip install hf-transfer per abilitare)."
fi

# Script di test rapido TimeLLM con modello Llama 3.2 3B (meta-llama/Llama-3.2-3B)
# - 1 sola epoca
# - metà dei layer (16) per ridurre VRAM
# - batch ridotto
# - mixed precision bf16
# Richiede: transformers >=4.44, tokenizers >=0.19, safetensors >=0.4.2

#MODEL_HF_ID="meta-llama/Llama-3.2-3B"
MODEL_HF_ID=openlm-research/open_llama_3b
LLM_LAYERS=16          # aumenta a 32 se la VRAM lo consente
SEQ_LEN=512
PRED_LEN=96
LABEL_LEN=48
BATCH_SIZE=8
DMODEL=32
DFF=128
EPOCHS=1
LR=1e-3
COMMENT="llama3p2_3B_quicktest"
MASTER_PORT=29511

# Autodetect GPU count
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
if [ "${GPU_COUNT}" -lt 1 ]; then
  echo "[WARN] Nessuna GPU visibile: eseguo comunque (potrebbe essere lentissimo)." >&2
  GPU_COUNT=1
fi

echo "== TimeLLM quick test con ${MODEL_HF_ID} (layers: ${LLM_LAYERS}) su ${GPU_COUNT} GPU =="
echo "[INFO] HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}"

accelerate launch --num_processes 1 --mixed_precision bf16 --main_process_port ${MASTER_PORT} run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_Llama3p2_3B_test \
  --model TimeLLM \
  --data ETTh1 \
  --features M \
  --seq_len ${SEQ_LEN} \
  --label_len ${LABEL_LEN} \
  --pred_len ${PRED_LEN} \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model ${DMODEL} \
  --d_ff ${DFF} \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LR} \
  --train_epochs ${EPOCHS} \
  --llm_model LLAMA \
  --llm_hf_id ${MODEL_HF_ID} \
  --llm_layers ${LLM_LAYERS} \
  --model_comment ${COMMENT}

echo "== Fine test rapido. Controlla i log per Train/Val/Test Loss =="
