mkdir -p checkpoints/enwik8/transformers-l

args="
--data datasets/pretraining/enwik8 \
--architecture sfsfsfsfsfsfsfsfsfsfsfsf \
--nlayers 12 \
--hid-sz 1152 \
--inner-hid-sz 1152 \
--nheads 8 \
--block-sz 512 \
--attn-span 2048 \
--dropout 0.1 \
--optim adam \
--lr 0.0007 \
--lr-warmup 9000 \
--niter 1 \
--batch-sz 64 \
--batch-split 2 \
--nbatches 180000 \
--checkpoint checkpoints/enwik8/transformers-l/dense.pt \
"

echo "Training ..."
python train.py $args

echo "Evaluation ..."
python train.py $args --full-eval-mode --batch-sz 8