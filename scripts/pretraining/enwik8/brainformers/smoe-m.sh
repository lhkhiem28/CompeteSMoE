mkdir -p checkpoints/enwik8/brainformers-m

args="
--data datasets/pretraining/enwik8 \
--base_arch brainformer \
--architecture sgfgfgsgfgfgsgfgfg \
--nlayers 3 \
--hid-sz 288 \
--inner-hid-sz 288 \
--nheads 8 \
--block-sz 512 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 4500 \
--niter 1 \
--batch-sz 64 \
--batch-split 2 \
--nbatches 90000 \
--checkpoint checkpoints/enwik8/brainformers-m/smoe.pt \
"

echo "Training ..."
python train.py $args

echo "Evaluation ..."
python train.py $args --full-eval-mode --batch-sz 8