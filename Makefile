


init:
	pylint --generate > .pylintrc
	sed -i 's/output-format=text/output-format=colorized/g' .pylintrc
	#output-format=text

requi:
	pip3 freeze requirements.txt

model cripto:
	python3 machine/cryptogra.py

auto_formater:
	pre-commit run --all-files -c .pre-commit-config.yaml
train:
	python3 train.py --dataset_path ./palete_06/ --validation_dataset ./palete_06/ --net mb1-ssd-lite --batch_size 32 --num_epochs 10 --lr 0.05
