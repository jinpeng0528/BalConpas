python train_continual.py --config-file configs/ade20k/panoptic-segmentation/100-10.yaml \
CONT.TASK 1 SOLVER.BASE_LR 0.0001 SOLVER.MAX_ITER 160000 OUTPUT_DIR ./output/ps/100-10/step1

for t in 2 3 4 5 6; do
  python train_continual.py --config-file configs/ade20k/panoptic-segmentation/100-10.yaml \
  CONT.TASK ${t} SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 10000 OUTPUT_DIR ./output/ps/100-10/step${t}
done
