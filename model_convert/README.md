# EdgeTAM模型转换

本文档介绍如何将EdgeTAM模型导出为ONNX格式，并对ONNX模型使用AXERA NPU工具链Pulsar2进行量化和编译，以便在AXERA NPU设备上运行。

## 环境准备

首先进行环境准备，主要包括EdgeTAM模型的导出环境配置和AXERA NPU工具链Pulsar2的安装。

### EdgeTAM模型导出环境配置

1. 源码安装EdgeTAM

  ```bash
  git clone https://github.com/facebookresearch/EdgeTAM.git && cd EdgeTAM

  pip install -e .
  ```

将本文件夹下的```prompt_encoder.py```替换```EdgeTAM/sam2/modeling
/sam/prompt_encoder.py```，以支持导出能够用于NPU量化编译的ONNX格式模型。

2. 安装相关python依赖

  ```bash
  pip install -r requirements.txt
  ```

### AXERA NPU工具链Pulsar2安装

下载安装AXERA NPU工具链Pulsar2，下载地址：[Pulsar2 Huggingface](https://huggingface.co/AXERA-TECH/Pulsar2/tree/main)，详细介绍请参考AXERA官方文档[Pulsar2 Doc](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html)。

## ONNX格式模型导出

使用```export_onnx.py```脚本将EdgeTAM模型导出为ONNX格式模型。以下是导出命令示例：

```bash
python3 export_onnx.py --sam2_cfg <edgetam.yaml path> \
                       --sam2_checkpoint <edgetam.pt path> \
                       --output_dir <your onnx save path>
```

将会得到以下ONNX模型文件：

- edgetam_image_encoder.onnx - EdgeTAM图像编码器模型
- edgetam_prompt_encoder.onnx - EdgeTAM提示编码器模型
- edgetam_prompt_mask_encoder.onnx - EdgeTAM提示掩码编码器模型
- edgetam_mask_decoder.onnx - EdgeTAM掩码解码器模型
- dense_embeddings_no_mask.npy - EdgeTAM提示编码器Embeddings权重文件

## AXERA Model量化编译

本仓库Release提供相关量化集和已导出的ONNX模型，可以直接下载使用Pulsar2工具链对ONNX模型进行量化编译。

### 使用示例

当前```configs```目录下提供了针对EdgeTAM模型的量化编译配置文件，修改```json```文件中的```input```和```calibration_dataset```参数改为对应的路径后，可以参考以下命令进行量化编译：

```bash
pulsar2 build --config configs/edgetam_image_encoder.json
pulsar2 build --config configs/edgetam_prompt_encoder.json
pulsar2 build --config configs/edgetam_prompt_mask_encoder.json
pulsar2 build --config configs/edgetam_mask_decoder.json
```

完成上述步骤后，即可在AXERA NPU设备上运行量化编译后的EdgeTAM模型。
PS: Pulsar2的更多使用，请参考官方文档[Pulsar2 Doc](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html)。
