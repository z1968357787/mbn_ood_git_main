### Prepare environment

```bash
conda create -n CL_Pytorch python=3.8
conda activate CL_Pytorch
pip install -r requirement.txt
```

### Run experiments

1. Edit the hyperparameters in the corresponding `options/XXX/XXX.yaml` file

2. Train models:

```bash
python main.py --config ./options/multi_steps/mbn_modify/cifar100.yaml
```

3. Test models with checkpoint (ensure save_model option is True before training)

```bash
python main.py --config ./options/multi_steps/mbn_test_metric/cifar100.yaml
```
change `pretrain_base_path` to the checkpoint path, change `metric` to the wanted ood metric, default: unknown, it wants to add more ood method, modify the `min_others_test` function within this method.

If you want to temporary change GPU device in the experiment, you can type `--device #GPU_ID` in terminal without changing `device` in `.yaml` config file.


### Add datasets and your method

Add corresponding dataset .py file to `datasets/`. It is done! The programme can automatically import the newly added datasets.

we put continual learning methods inplementations in `/methods/multi_steps` folder, pretrain methods in `/methods/pretrain` folder

Supported Datasets:

- Natural image datasets: CIFAR-10, CIFAR-100, ImageNet100, ImageNet1K, ImageNet-R, TinyImageNet, CUB-200

- Medical image datasets: MedMNIST, path16, Skin7, Skin8, Skin40

More information about the supported datasets can be found in `datasets/`

We use `os.environ['DATA']` to access image data. You can config your environment variables in your computer by editing `~/.bashrc` or just change the code.



