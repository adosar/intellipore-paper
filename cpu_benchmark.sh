# CPU benchmark: scratch vs pretrained
for target in 'CO2-298-2.5'; do

	echo -e "\033[31;1mTarget task: ${target}\033[0m\n"

	for ckpt_path in 'experiments/pretrain/early_pretrain/lightning_logs/version_0/checkpoints/best.ckpt'; do

		python evaluate.py -c configs/evaluate.yaml \
			--ckpt_path=${ckpt_path} \
			--train_sizes='[100]' \
			--n_runs='[1]' \
			--target=${target} \
			--voxels_path='/home/asarikas/databases/SpbNet/benchmark/hmof/voxels_data_GS32_CB30' \
			--labels_path='/home/asarikas/databases/SpbNet/benchmark/hmof/all.csv' \
			--trainer.default_root_dir='experiments/cpu_benchmark/hmof/' \
			--trainer.max_epochs=1000 \
			--trainer.accelerator='cpu' \
			--n_frozen_layers=6
		done
	done
