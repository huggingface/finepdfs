#!/usr/bin/env bash
set -euo pipefail

LR=3e-4
UNFROZEN_LAYERS=4
STEPS=5000

languages="eng_Latn,ces_Latn"

for language in $languages; do
# answerdotai/ModernBERT-large for english
  if [[ "$language" == "eng_Latn" ]]; then
    MODEL="answerdotai/ModernBERT-large"
  else
    MODEL="mmbert-colab/mmBERT-base"
  fi
  [[ "$language" == "language" || -z "$language" ]] && continue

  if [[ "$language" == "eng_Latn" ]]; then
    DATASET_NAME="HuggingFaceFW/finepdfs_eng_Latn_labeled"
  else
    DATASET_NAME="HuggingFaceFW/finepdfs_fw_edu_labeled"
  fi


  echo "Computing samples for $language"
  # Because of edu we have to downscale the highest classes by 10x
  samples=$(python classification/compute_samples_per_class.py \
      --downscale_factor=10 \
      --target_column="edu_score" \
      --dataset_name="$DATASET_NAME" \
      --max_repetitions=40 \
      --subset="$language" \
    | awk -F': ' '/^Resulting samples:/ {print $2}' | tail -n 1)

  if [[ -z "${samples}" ]]; then
    echo "No samples extracted for $language; skipping."
    continue
  fi

  echo "Submitting training for $language with samples: $samples"
  python blocks/classification/submit_training.py \
    --base-model-name "$MODEL" \
    --job-name "classifier_${MODEL}_${language}" \
    --prefix-run-name $language \
    --target-column fw_edu_score \
    --dataset-name "$DATASET_NAME" \
    --log-dir "./training_logs/${language}" \
    --subset "$language" \
    --sample-per-class "$samples" \
    --transformer-layers-unfrozen $UNFROZEN_LAYERS \
    --lr $LR \
    --max-steps $STEPS
done
