mkdir -p checkpoints/enwik8/transformers-m

args="
--data datasets/pretraining/enwik8 \
--base_arch transformer \
--architecture sgsgsgsgsgsgsgsgsg \
--gate_name xmoe \
--nlayers 9 \
--hid-sz 336 \
--inner-hid-sz 336 \
--nheads 8 \
--block-sz 512 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 4500 \
--niter 1 \
--batch-sz 48 \
--batch-split 2 \
--nbatches 90000 \
--checkpoint checkpoints/enwik8/transformers-m/xmoe.pt \
"

echo "Training ..."
python train.py $args

echo "Evaluation ..."
python train.py $args --full-eval-mode --batch-sz 8