# models/__init__.py 完整代码（包含基线+新增模型）
from .resnet import resnet18  # 基线ResNet18（原文件不动）
from .resnet_half import resnet18_half  # 通道减半版
from .resnet_se import se_resnet18  # SE注意力版
from .mobilenetv2 import mobilenetv2  # 原仓库自带
from .shufflenetv2 import shufflenetv2  # 原仓库自带

# 模型名→模型函数的映射，供main.py调用
__all__ = {
    'resnet18': resnet18,
    'resnet18_half': resnet18_half,
    'se_resnet18': se_resnet18,
    'mobilenetv2': mobilenetv2,
    'shufflenetv2': shufflenetv2
}
