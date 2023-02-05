# Set image tags
TAG = ess-tempo

# Targets & commands
build:
	docker build -t ${TAG} -f Dockerfile .

train:
	docker run -v $(dataset_dir):"/opt/ml/input/data/training/" -v $(output_dir):"/opt/ml/model/" ${TAG} train.py

train-gpu:
	docker run -v $(dataset_dir):"/opt/ml/input/data/training/" -v $(output_dir):"/opt/ml/model/" --gpus all ${TAG} train.py

finetune:
	docker run -v $(dataset_dir):"/opt/ml/input/data/training/" -v $(pretrained_model_dir):"/opt/ml/pretrained_model/" -v $(output_dir):"/opt/ml/model/" ${TAG} finetune.py

finetune-gpu:
	docker run -v $(dataset_dir):"/opt/ml/input/data/training/" -v $(pretrained_model_dir):"/opt/ml/pretrained_model/" -v $(output_dir):"/opt/ml/model/" --gpus all ${TAG} finetune.py

eval:
	docker run -v $(dataset_dir):"/opt/ml/input/data/training/" -v $(pretrained_model_dir):"/opt/ml/pretrained_model/" -v $(output_dir):"/opt/ml/model/" ${TAG} eval.py

eval-gpu:
	docker run -v $(dataset_dir):"/opt/ml/input/data/training/" -v $(pretrained_model_dir):"/opt/ml/pretrained_model/" -v $(output_dir):"/opt/ml/model/" --gpus all ${TAG} eval.py

cleanup:
	docker rmi -f ${TAG}