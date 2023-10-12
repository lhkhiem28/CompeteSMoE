mkdir -p checkpoints/enwik8/transformers-l

args="
--data datasets/pretraining/enwik8 \
--base_arch transformer \
--architecture sgsgsgsgsgsgsgsgsgsgsgsg \
--gate_name smoe \
--nlayers 12 \
--hid-sz 504 \
--inner-hid-sz 504 \
--nheads 8 \
--block-sz 512 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 6000 \
--niter 1 \
--batch-sz 48 \
--batch-split 2 \
--nbatches 120000 \
--checkpoint checkpoints/enwik8/transformers-l/smoe.pt \
"

echo "Training ..."
python train.py $args

echo "Evaluation ..."
python train.py $args --full-eval-mode --batch-sz 8