mkdir -p checkpoints/enwik8/transformers-s

args="
--data datasets/pretraining/enwik8 \
--base_arch transformer \
--architecture sgsgsgsg \
--nlayers 4 \
--hid-sz 192 \
--inner-hid-sz 192 \
--nheads 8 \
--block-sz 512 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 3000 \
--niter 1 \
--batch-sz 64 \
--batch-split 2 \
--nbatches 60000 \
--checkpoint checkpoints/enwik8/transformers-s/smoe.pt \
"

echo "Training ..."
python train.py $args

echo "Evaluation ..."
python train.py $args --full-eval-mode --batch-sz 8