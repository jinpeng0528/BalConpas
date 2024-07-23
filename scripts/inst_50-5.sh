python train_continual.py --config-file configs/ade20k/instance-segmentation/50-5.yaml \
CONT.TASK 1 SOLVER.BASE_LR 0.0001 SOLVER.MAX_ITER 160000 OUTPUT_DIR ./output/is/50-5/step1

for t in 2 3 4 5 6 7 8 9 10 11; do
  python train_continual.py --config-file configs/ade20k/instance-segmentation/50-5.yaml \
  CONT.TASK ${t} SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 5000 OUTPUT_DIR ./output/is/50-5/step${t}
done
