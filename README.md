<div  align="center">

# Equivariant Self-Supervision for Musical Tempo Estimation

[Elio Quinton](https://scholar.google.com/citations?user=IaciybgAAAAJ) 

(Universal Music Group)

<p align="center">
<img src="sst_diagram.png" width="250">
</p align="center">
</div>

Official implementation of [*Equivariant Self-Supervision for Musical Tempo Estimation*](https://arxiv.org/pdf/2209.01478.pdf), published at [ISMIR 2022](https://ismir2022.ismir.net). 


## Citation

If you use this code and/or paper in your research please cite: 

[1] Elio Quinton, "Equivariant Self-Supervision for Musical Tempo", in *International Society for Music Information Retrieval Conference (ISMIR)*, 2022.


## Datasets

Datasets should all be placed in the same folder, whith one subfolder per dataset, named exactly as follows:

```
datasets_folder
    |-- magnatagatune
    |-- ACM_Mirum_tempo
    |-- gtzan
    |-- hainsworth
    |-- giantsteps-tempo-dataset-master
```

### Datasets folder structure
Each dataset has a specific folder structure, including a sub-folder with audio files. 
Please make sure you read the [datasets README](README_datasets.md) and make sure you strictly follow the folder structure. 


### Dataset indexes
For each dataset, we prepared JSON indexes listing all audio files and corresponding tempo annotations when applicable, available in the [`dataset_indexes`](datasets_indexes) folder of this repo. The dataloader will consume these index files. 

Magnatagatune does not contain tempo annotations and is used for self-supervised pre-training. All other datasets contain tempo annotations and can be used for fine-tuning and evaluation. 

We also included a [`ACM_mirum_tempo_tiny.json`](datasets_indexes/ACM_mirum_tempo_tiny.json) indexe, with only a tiny subset of the ACM Mirum Tempo dataset, as a convenient small dataset to test code. 

## Configuration files

The model and training (hyper)parameters are set in configuration files located in the [`sst/configs`](sst/configs) folder. 

The recommended approach for experimenting with different parameters is to modify the config files. 


## Run with Docker (recommended)

Docker provides a clean environment with well controlled dependencies and CUDA drivers pre-installed. 

If you don't have Docker installed, please follow the [official install instructions](https://docs.docker.com/get-docker/). 

### GPU support

You need to have the [nvidia-container-runtime](https://nvidia.github.io/nvidia-container-runtime/) installed to use GPUs with docker. 

Assuming CUDA and the nvidia-container-runtime are installed and configured on the host machine, the image we provide should allow you to run the code on the GPU without requiring any code configuration. 

Note that the code should also run natively on CPU, without requiring any specific configuration. 

### Using Make

For convenience and getting up and running easily, we provide a Makefile with preset commands. 
Note that the Docker image is configured to expect that the `dataset_dir`, `pretrained_model_dir` and `output_dir` are separate directories. Please make sure you respect this structure to avoid errors. 

1. Build the docker image: 

```
make build
```

2. Run self-supervised training

With GPU:
```
make train-gpu dataset_dir="<path_to_your_local_datasets_folder>" output_dir="<path_to_your_local_output_folder>"
```
Without GPU:
```
make train dataset_dir="<path_to_your_local_datasets_folder>" output_dir="<path_to_your_local_output_folder>"
```

3. Run fine-tuning, with or without GPU

With GPU:
```
make finetune-gpu dataset_dir="<path_to_your_local_datasets_folder>" pretrained_model_dir="<path_to_your_local_pretrained_model_folder>" output_dir="<path_to_your_local_output_folder>"
```
Without GPU:
```
make finetune dataset_dir="<path_to_your_local_datasets_folder>" pretrained_model_dir="<path_to_your_local_pretrained_model_folder>" output_dir="<path_to_your_local_output_folder>"
```

4. Run evaluation

With GPU:
```
make eval-gpu dataset_dir="<path_to_your_local_datasets_folder>" pretrained_model_dir="<path_to_your_local_pretrained_model_folder>" output_dir="<path_to_your_local_output_folder>"
```
Without GPU:
```
make eval dataset_dir="<path_to_your_local_datasets_folder>" pretrained_model_dir="<path_to_your_local_pretrained_model_folder>" output_dir="<path_to_your_local_output_folder>"
```

5. Cleanup

```
make cleanup
```

This command will delete the docker image and associated stopped containers. 


### Using Docker commands

Using Docker commands directly, instead of using Make, is recommended if you need further configuration. See below for reference basic docker commands (equivalent to Make commands): 

<details>
<summary>Basic Docker commands </summary>

The docker image is configured to read and write data, pre-trained models and model checkpoints from/to volumes mounted in the docker container. The commands below show how to achieve this with local directories. 

If need be the run commands shown below can be extended with the [usual features provided by Docker CLI](https://docs.docker.com/engine/reference/run/). 

1. Build the image

```
docker build -t ess-tempo -f Dockerfile .
```

2. Run self-supervised training: 

```
docker run -v "<path_to_your_local_datasets_folder>":"/opt/ml/input/data/training/" -v "<path_to_your_local_output_folder>":"/opt/ml/model/" ess-tempo train.py
```

3. Run fine-tuning:

```
docker run -v "<path_to_your_local_datasets_folder>":"/opt/ml/input/data/training/" -v "<path_to_your_local_pretrained_model_folder>":"/opt/ml/pretrained_model/" -v "<path_to_your_local_output_folder>":"/opt/ml/model/" ess-tempo finetune.py
```

4. Run evaluation:

```
docker run -v "<path_to_your_local_datasets_folder>":"/opt/ml/input/data/training/" -v "<path_to_your_local_pretrained_model_folder>":"/opt/ml/pretrained_model/" -v "<path_to_your_local_output_folder>":"/opt/ml/model/" ess-tempo eval.py
```

5. Cleanup

```
docker rmi -f ess-tempo
``` 

</details>

### GPU support

For running with a GPU, add a `--gpus all` flag to the basic commands.

See the official [Docker documentation](https://docs.docker.com/engine/reference/commandline/run/#gpus) for futher GPU options.

### Running your custom code

By default if you change the code, the image needs to be rebuilt before it can be re-run and reflect the code changes. 

For faster development and not having to re-build the image at each code change, you can mount your code volume into the docker container by adding the following argument to the docker run command: `-v <path_to_the_local_code_dir>:/opt/ml/code/`. Where `<path_to_the_local_code_dir>` is the absolute path to the root folder of this repo. 



## Running outside Docker (not recommended)

If running outside Docker, we recommend using a virtual environment. 
You should first install: `Python 3.8`, `Pytorch 1.11.0`, and `torchaudio 0.11.0`. 

You can then install the other requirements in the [`requirements.txt`](requirements.txt) file.


### Update default Docker filepaths

The config files provided in this repo are set for usage with Docker. The paths to local directories (dataset folder, output folder etc.) are subdirectories of `/opt/ml`. 

When running without Docker, make sure you amend the paths starting with `/opt/ml` in the [config files](sst/configs/) with your appropriate local paths. 

### Run scripts

Move to the `sst` directory: 

```
cd sst
```

The code can be then executed with 3 main scripts for the 3 main phases: 

1. Self-supervised training
```
python train.py
```
2. Supervised fine-tuning
```
python finetune.py
```
3. Evaluation
```
python eval.py
```

## Tests

Tests are located in the [`sst/tests`](sst/tests) folder. Instructions to run them can be found in the dedicated [README file](sst/tests/README.md).
