mkdir -p checkpoints/enwik8/transformers-s

args="
--data datasets/pretraining/enwik8 \
--architecture sgsgsgsg \
--nlayers 4 \
--hid-sz 192 \
--inner-hid-sz 192 \
--nheads 8 \
--block-sz 512 \
--attn-span 2048 \
--dropout 0.1 \
--optim adam \
--lr 0.0007 \
--lr-warmup 3000 \
--niter 1 \
--batch-sz 64 \
--batch-split 2 \
--nbatches 60000 \
--checkpoint checkpoints/enwik8/transformers-s/dense.pt \
"

echo "Training ..."
python train.py $args

echo "Evaluation ..."
python train.py $args --full-eval-mode --batch-sz 8