*This is an instruction on how to install the datasets used in our experiments.*

We suggest putting all datasets under the same folder (say `$ImageNet`) to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure looks like:

```
$ImageNet/
|–– ImageNet-1k/
|–– ImageNet-v2/
|–– ImageNet-Sketch/
|–– ImageNet-A/
|–– ImageNet-R/


$retrieved/
|–– imagenet_retrieved_LAION400M-all_synonyms-random/
|   |-- 0/
|   |-- 1/
|   |-- ...
|   |-- 999
```

***

# How to download datasets

Datasets list:
- [ImageNet](#ImageNet)
- [ImageNet-V2](#ImageNet-V2)
- [ImageNet-Sketch](#ImageNet-Sketch)
- [ImageNet-A](#ImageNet-A)
- [ImageNet-R](#ImageNet-R)




### ImageNet
- Create a folder named `ImageNet-1k/` under `$ImageNet`.
- Download `split_ImageNet.json` to this folder from [this link]("https://drive.google.com/file/d/1SvPIN6iV6NP2Oulj19a869rBXrB5SNFo/view").
- Create `images/` under `ImageNet-1k/`.
- Download training  and validation sets, and then decompress to `ImageNet-1k/images/`.

  - Option 1: Download the dataset from [Huggingface repo](https://huggingface.co/datasets/ILSVRC/imagenet-1k).

  - Option 2: Download the dataset from the [official website](https://image-net.org/index.php).


- Put the `classnames.pt` to `$ImageNet/ImageNet-1k/` from `data_resource/imagenet`. The class names are copied from [ImageNet Huggingface repo](https://huggingface.co/datasets/ILSVRC/imagenet-1k).

The directory structure of ImageNet-1k should look like
```
ImageNet-1k/
|–– classnames.pt
|-- split_ImageNet.json
|–– images/
|   |–– train/
|   |   |-- n01440764
|   |   |-- n01443537
|   |   |-- ...
|   |   |-- n15075141
|   |–– val/
```



### ImageNet-V2
- Create a folder named `ImageNet-v2/` under `$ImageNet`.

- Create `images/` under `ImageNet-v2/`.

- Download test set from [Huggingface repo](https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main), and then decompress to `ImageNet-v2/images/`.

   (This repo provides 3 different test set versions. Following the previous work in OOD generalization, we choose `imagenetv2-matched-frequency` version as the test set. )

- Put the `classnames.pt` to `$ImageNet/ImageNet-v2/` from `data_resource/imagenetv2`.

The directory structure of ImageNet-v2 should look like

```
ImageNet-v2/
|–– classnames.pt
|–– images/
|   |-- 0
|   |-- 1
|   |-- ...
|   |-- 999
```



### ImageNet-Sketch
- Create a folder named `ImageNet-Sketch/` under `$ImageNet`.
- Create `images/` under `ImageNet-Sketch/`.
- Download test set from [Huggingface repo](https://huggingface.co/datasets/songweig/imagenet_sketch), and then decompress to `ImageNet-Sketch/images/`.

- Put the `classnames.pt` to `$ImageNet/ImageNet-Sketch/` from `data_resource/imagenet_sketch`.

The directory structure of ImageNet-Sketch should look like

```
ImageNet-Sketch/
|–– classnames.pt
|–– images/
|   |-- n01440764
|   |-- n01443537
|   |-- ...
|   |-- n15075141
```



### ImageNet-A
- Create a folder named `ImageNet-A/` under `$ImageNet`.
- Create `images/` under `ImageNet-A/`.
- Download test set from [Github repo](https://github.com/hendrycks/natural-adv-examples?tab=readme-ov-file), and then decompress to `ImageNet-A/images/`.

- Put the `classnames.pt` to `$ImageNet/ImageNet-A/` from `data_resource/imagenet_a`.

The directory structure of ImageNet-A should look like

```
ImageNet-A/
|–– classnames.pt
|–– images/
|   |-- n01498041
|   |-- n01531178
|   |-- ...
|   |-- n12267677
```




### ImageNet-R
- Create a folder named `ImageNet-R/` under `$ImageNet`.
- Create `images/` under `ImageNet-R/`.
- Download test set from [Github repo](https://github.com/hendrycks/imagenet-r?tab=readme-ov-file), and then decompress to `ImageNet-R/images/`.

- Put the `classnames.pt` to `$ImageNet/ImageNet-R/` from `data_resource/imagenet_r`.

The directory structure of ImageNet-R should look like

```
ImageNet-R/
|–– classnames.pt
|–– images/
|   |-- n01443537
|   |-- n01484850
|   |-- ...
|   |-- n12267677
```



If you had downloaded the ImageNet dataset and its variants before, you can create symbolic links to map the corresponding datasets to `$ImageNet/{dataset name}/images`.



# How to prepare the few-shot dataset
We have already prepared the `fewshot{4/8/16}_seed{1/2/3}.txt` files for ImageNet dataset in `SRAPF/data_resource/imagenet/` folder.



# How to prepare the retrieval dataset

For retrieval augmentation, we follow SWAT's approach. Please refer to [SWAT/retrieval/RETRIEVAL.md](https://github.com/tian1327/SWAT/blob/master/retrieval/RETRIEVAL.md) for instructions on how to set up the retrieved datasets. The annotation file of retrieved dataset (top-500 selected) is available at: `SRAPF/data_resource/T2T500.txt` .
