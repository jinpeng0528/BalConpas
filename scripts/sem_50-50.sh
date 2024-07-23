python train_continual.py --config-file configs/ade20k/semantic-segmentation/50-50.yaml \
CONT.TASK 1 SOLVER.BASE_LR 0.0001 SOLVER.MAX_ITER 160000 OUTPUT_DIR ./output/ss/50-50/step1

for t in 2 3; do
  python train_continual.py --config-file configs/ade20k/semantic-segmentation/50-50.yaml \
  CONT.TASK ${t} SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 50000 OUTPUT_DIR ./output/ss/50-50/step${t}
done
