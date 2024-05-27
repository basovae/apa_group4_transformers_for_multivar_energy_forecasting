export CUDA_VISIBLE_DEVICES=0

for preLen in 96 192 336 720
do

python -u main.py \
  --is_training True \
  --root_path data \
  --data_path all_countries.csv \
  --data custom \
  --features S \
  --seq_len 96 \
  --label_len 96 \
  --pred_len $preLen \
  --learning_rate 2e-4
done