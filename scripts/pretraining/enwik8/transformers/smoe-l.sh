mkdir -p checkpoints/enwik8/transformers-l

args="
--data datasets/pretraining/enwik8 \
--base_arch transformer \
--architecture sgsgsgsgsgsgsgsgsgsgsgsg \
--nlayers 12 \
--hid-sz 576 \
--inner-hid-sz 576 \
--nheads 8 \
--block-sz 512 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 9000 \
--niter 1 \
--batch-sz 64 \
--batch-split 2 \
--nbatches 180000 \
--checkpoint checkpoints/enwik8/transformers-l/smoe.pt \
"

echo "Training ..."
python train.py $args

echo "Evaluation ..."
python train.py $args --full-eval-mode --batch-sz 8