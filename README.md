# Diffusion-based Multi-layer Semantic Reconstruction for Unsupervised Out-of-Distribution Detection

This is the official repo for

[Diffusion-based Multi-layer Semantic Reconstruction for Unsupervised Out-of-Distribution Detection](https://arxiv.org/abs/2411.10701) (NeurIPS 2024)

## Abstract

Unsupervised out-of-distribution (OOD) detection aims to identify out-of-domain data by learning only from unlabeled In-Distribution (ID) training samples, which is crucial for developing a safe real-world machine learning system. Current reconstruction-based methods provide a good alternative approach by measuring the reconstruction error between the input and its corresponding generative counterpart in the pixel/feature space. However, such generative methods face the key dilemma, \$i.e.\$, *improving the reconstruction power of the generative model while keeping a compact representation of the ID data.* To address this issue, we propose the diffusion-based layer-wise semantic reconstruction approach for unsupervised OOD detection. The innovation of our approach is that we leverage the diffusion model's intrinsic data reconstruction ability to distinguish ID samples from OOD samples in the latent feature space. Moreover, to set up a comprehensive and discriminative feature representation, we devise a multi-layer semantic feature extraction strategy. By distorting the extracted features with Gaussian noises and applying the diffusion model for feature reconstruction, the separation of ID and OOD samples is implemented according to the reconstruction errors. Extensive experimental results on multiple benchmarks built upon various datasets demonstrate that our method achieves state-of-the-art performance in terms of detection accuracy and speed.

## Environment

To ensure compatibility and reproduce the results, please set up your environment using the provided `environment.txt` file.

### Steps to Set Up the Environment:

1. Make sure you have [conda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.
2. Create a new environment and install the dependencies using the `environment.txt` file:
   ```sh
   conda create --name <your_environment_name> --file environment.txt
   ```
3. Activate the newly created environment:
   ```sh
   conda activate <your_environment_name>
   ```

## Usage

### To Train a Model on Datasets such as CIFAR10:

1. **Download the Pre-Trained Checkpoint**\
   Please download the `efficientnet-b4-6ed6700e.pth` file from [this link](https://drive.google.com/file/d/1yAQbBQQtiMvhDYWuXTXdGv5uE9enDdKD/view?usp=drive_link) and place it in the current project directory.

2. **Run the Training Script**\
   You can start training by running the `main.py` script:

   ```bash
   python main.py --config config/Config.yaml --data_path ./data
   ```

3. **Multi-GPU Training**\
   If you wish to utilize multiple GPUs, you can use `torchrun` as follows:

   ```bash
   torchrun --nproc_per_node=<number_of_gpus> main.py --config config/Config.yaml --data_path ./data
   ```

---

### To Evaluate the Model for Out-of-Distribution (OOD) Detection:

#### 1. **Download the Pre-Trained Checkpoints**

Download the pre-trained model checkpoints from the following links and place them under the `/checkpoint/` directory within the current project folder. **Note:** You do not need to rename these files.

- **EfficientNet Pre-trained Models**:

  - [CIFAR-10 Pre-trained Model](https://drive.google.com/file/d/1gHCo53lsiUpFVdKt-XIKV8v7vingNfNp/view?usp=sharing): After downloading, place it under `/checkpoint/checkpoint-0.pth`.
  - [CIFAR-100 Pre-trained Model](https://drive.google.com/file/d/1gHCo53lsiUpFVdKt-XIKV8v7vingNfNp/view?usp=sharing): After downloading, place it under `/checkpoint/checkpoint-last.pth`.

- **ResNet-50 Pre-trained Models**:

  - [CIFAR-10 Pre-trained Model](https://drive.google.com/file/d/14ABPzI-TI-N6wyp9dCYD9bC6V_7y3vA3/view?usp=drive_link): After downloading, place it under `/checkpoint/checkpoint-0.pth`.
  - [CIFAR-100 Pre-trained Model](https://drive.google.com/file/d/1cNvRsIG-SMuPkkh5oxSPa3w6fXjSkdrt/view?usp=drive_link): After downloading, place it under `/checkpoint/checkpoint-last.pth`.

Ensure that you organize these files properly in the `/checkpoint/` directory to easily use them when running experiments.

#### 2. **Prepare OOD Datasets**

- Download the OOD datasets from [this link](https://drive.google.com/file/d/1uTtyywjKurfQnQG9jFjYVKjUFthUI8Qd/view?usp=drive_link).
- Extract and place the datasets inside the `data` folder in your current project directory.
- Specify dataset paths as follows in the configuration:
  - **CIFAR100**: Default will be downloaded automatically.
  - **iSUN**: `/data/iSUN`
  - **SVHN**: `/data/SVHN`
  - **LSUN**: `/data/LSUN`
  - **LSUN (resized)**: `/data/LSUN_resize`
  - **DTD (Describable Textures Dataset)**: `/data/dtd/images`
  - **Places365**: `/data/places365`
  Ensure that you adjust these paths in `test_mse_mfsim.py` where the datasets are being loaded:
  ```python
  # Example Paths to Set
  dataset_test0 = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_train)
  dataset_test1 = datasets.ImageFolder(root='./data/iSUN', transform=transform_train)
  dataset_test2 = torchvision.datasets.SVHN(root='./data/SVHN', split='test', download=True, transform=transform_train)
  dataset_test3 = datasets.ImageFolder(root='./data/LSUN', transform=transform_train)
  dataset_test4 = datasets.ImageFolder(root='./data/LSUN_resize', transform=transform_train)
  dataset_test5 = datasets.ImageFolder(root='./data/dtd/images', transform=transform_train)
  dataset_test6 = datasets.ImageFolder(root='./data/places365', transform=transform_train)
  ```

#### 3. **Run the Evaluation Script**

To evaluate the model, use the `main.py` script as follows:

- Specify the LDM checkpoint (`--pretrained_ldm_ckpt`) and configuration (`--pretrained_ldm_cfg`):

  ```bash
  python test_mse_mfsim.py --pretrained_ldm_ckpt ./checkpoint/checkpoint-last.pth --pretrained_ldm_cfg config/Config.yaml --data_path ./data --evaluate
  ```

- You can choose to calculate OOD detection metrics using either **MSE** or **MFsim**. To do so, specify `--similarity_type`:

  - **MSE**:
    ```bash
    python test_mse_mfsim.py --pretrained_ldm_ckpt ./checkpoint/checkpoint-last.pth --pretrained_ldm_cfg config/Config.yaml --data_path ./data --evaluate --similarity_type MSE
    ```
  - **MFsim**:
    ```bash
    python test_mse_mfsim.py --pretrained_ldm_ckpt ./checkpoint/checkpoint-last.pth --pretrained_ldm_cfg config/Config.yaml --data_path ./data --evaluate --similarity_type MFsim
    ```

- To evaluate the model using the **LR** metric, you need to specify both the initial (`--pretrained_ldm_ckpt_first`) and the end (`--pretrained_ldm_ckpt_end`) pre-trained checkpoints, along with the configuration file:

  ```bash
  python test_LR.py --pretrained_ldm_ckpt_first ./checkpoint/checkpoint-0.pth --pretrained_ldm_ckpt_end ./checkpoint/checkpoint-last.pth --pretrained_ldm_cfg config/Config.yaml --data_path ./data --evaluate --similarity_type MFsim
  ```

## Acknowledgement

This code is primarily based on modifications made from [latent-diffusion](https://github.com/CompVis/latent-diffusion) and [UniAD](https://github.com/zhiyuanyou/UniAD). We would like to express our gratitude to the authors of these repositories for their excellent foundational work, which significantly inspired and supported our research. Their contributions have been invaluable in the development of this project.

