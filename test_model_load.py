#!/usr/bin/env python3
"""
F5-TTS 模型加载测试脚本
用于验证模型文件是否能正确加载和使用
"""

import os
import sys
from pathlib import Path

# 模型路径配置
MODEL_PATH = "models/F5-TTS/F5TTS_v1_Base"
MODEL_FILE = os.path.join(MODEL_PATH, "model_1250000.safetensors")
VOCAB_FILE = os.path.join(MODEL_PATH, "vocab.txt")

def check_files():
    """检查模型文件是否存在"""
    print("=== 检查模型文件 ===")
    
    if os.path.exists(MODEL_FILE):
        size = os.path.getsize(MODEL_FILE) / (1024**3)  # GB
        print(f"✓ 模型文件存在: {MODEL_FILE}")
        print(f"  文件大小: {size:.2f} GB")
    else:
        print(f"✗ 模型文件不存在: {MODEL_FILE}")
        return False
    
    if os.path.exists(VOCAB_FILE):
        with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
            vocab_lines = len(f.readlines())
        print(f"✓ 词汇表文件存在: {VOCAB_FILE}")
        print(f"  词汇表大小: {vocab_lines} 行")
    else:
        print(f"✗ 词汇表文件不存在: {VOCAB_FILE}")
        return False
    
    return True

def test_import():
    """测试F5-TTS模块导入"""
    print("\n=== 测试模块导入 ===")
    
    try:
        # 添加src目录到Python路径
        src_path = os.path.join(os.path.dirname(__file__), 'src')
        if os.path.exists(src_path):
            sys.path.insert(0, src_path)
            print(f"✓ 添加src路径: {src_path}")
        
        # 尝试导入F5-TTS
        import f5_tts
        print(f"✓ 成功导入f5_tts模块")
        print(f"  版本: {getattr(f5_tts, '__version__', 'unknown')}")
        
        return True
        
    except ImportError as e:
        print(f"✗ 导入f5_tts失败: {e}")
        print("  请确保已安装F5-TTS: pip install -e .")
        return False

def test_model_load():
    """测试模型加载"""
    print("\n=== 测试模型加载 ===")
    
    try:
        # 这里可以添加实际的模型加载代码
        # 目前只是验证文件路径
        import torch
        print("✓ PyTorch可用")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA可用，设备数量: {torch.cuda.device_count()}")
        else:
            print("! CUDA不可用，将使用CPU")
        
        # 尝试加载safetensors文件
        from safetensors import safe_open
        with safe_open(MODEL_FILE, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            print(f"✓ 成功读取模型文件")
            print(f"  模型参数数量: {len(keys)}")
            print(f"  前5个参数: {keys[:5]}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return False

def main():
    """主函数"""
    print("F5-TTS 调试环境测试")
    print("=" * 50)
    
    print(f"Python版本: {sys.version}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.path[:3]}...")  # 只显示前3个路径
    
    # 运行所有测试
    all_passed = True
    all_passed &= check_files()
    all_passed &= test_import()
    all_passed &= test_model_load()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ 所有测试通过！F5-TTS调试环境配置成功。")
        print("\n可以开始调试了！使用以下方式：")
        print("1. 在VSCode中按F5运行调试")
        print("2. 设置断点进行调试")
        print("3. 使用终端: python test_model_load.py")
    else:
        print("✗ 部分测试失败，请检查配置。")

if __name__ == "__main__":
    main() 