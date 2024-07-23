python train_continual.py --config-file configs/ade20k/semantic-segmentation/100-50.yaml \
CONT.TASK 1 SOLVER.BASE_LR 0.0001 SOLVER.MAX_ITER 160000 OUTPUT_DIR ./output/ss/100-50/step1

python train_continual.py --config-file configs/ade20k/semantic-segmentation/100-50.yaml \
CONT.TASK 2 SOLVER.BASE_LR 0.00005 SOLVER.MAX_ITER 50000 OUTPUT_DIR ./output/ss/100-50/step2
