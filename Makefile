# 指定使用的gpu编号
export CUDA_VISIBLE_DEVICES=0
DATA=/disk1/wangjialei/data/EyePacs_data



train:
	python ./train.py --data-dir $(DATA)

predict:
	python ./eval.py --data-dir $(DATA) --checkpoint /disk1/wangjialei/project/diabetic_demo/models/model-2-2.ckpt

clean:
	rm -rf models

train_debug:
	python -m debugpy --listen 5678 ./train.py --data-dir $(DATA)
predict_debug:
	python -m debugpy --listen 5678 ./eval.py --data-dir $(DATA) --checkpoint /disk1/wangjialei/project/diabetic_demo/models/model-2-2.ckpt
.PHONY: clean train train_debug predict predict_debug