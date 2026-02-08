# models/__init__.py 完整代码（包含基线+新增模型）
from .resnet import ResNet18  # 基线ResNet18（原文件不动）
from .resnet_half import resnet18_half  # 通道减半版
from .resnet_se import se_resnet18  # SE注意力版
from .mobilenetv2 import MobileNetV2  # 原仓库自带
from .shufflenetv2 import ShuffleNetV2  # 原仓库自带

# 模型名→模型函数的映射，供main.py调用
__all__ = {
    'resnet18': ResNet18,
    'resnet18_half': resnet18_half,
    'se_resnet18': se_resnet18,
    'mobilenetv2': MobileNetV2,
    'shufflenetv2': ShuffleNetV2
}
