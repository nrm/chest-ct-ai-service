#!/usr/bin/env python3
"""
GPU Check Script for RadiAssist
Проверяет доступность GPU и выводит диагностическую информацию
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_nvidia_smi():
    """Проверяет доступность nvidia-smi"""
    print("🔍 Проверка nvidia-smi...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ nvidia-smi доступен")
            print("📊 Информация о GPU:")
            print(result.stdout)
            return True
        else:
            print("❌ nvidia-smi недоступен")
            print(f"Ошибка: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⏰ nvidia-smi команда превысила время ожидания")
        return False
    except FileNotFoundError:
        print("❌ nvidia-smi не найден в PATH")
        return False
    except Exception as e:
        print(f"❌ Ошибка при запуске nvidia-smi: {e}")
        return False

def check_pytorch_cuda():
    """Проверяет доступность PyTorch CUDA"""
    print("\n🔥 Проверка PyTorch CUDA...")
    try:
        import torch
        print(f"✅ PyTorch установлен: {torch.__version__}")
        
        if torch.cuda.is_available():
            print("✅ PyTorch CUDA доступен")
            print(f"📊 Количество GPU устройств: {torch.cuda.device_count()}")
            print(f"📊 Текущее устройство: {torch.cuda.current_device()}")
            print(f"📊 Имя устройства: {torch.cuda.get_device_name()}")
            
            # Дополнительная информация о GPU
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"📊 GPU {i}: {props.name}")
                print(f"   - Память: {props.total_memory / 1024**3:.1f} GB")
                print(f"   - Compute Capability: {props.major}.{props.minor}")
            
            return True
        else:
            print("❌ PyTorch CUDA недоступен")
            print("Возможные причины:")
            print("- CUDA не установлен")
            print("- PyTorch скомпилирован без CUDA")
            print("- GPU не поддерживается")
            return False
    except ImportError:
        print("❌ PyTorch не установлен")
        return False
    except Exception as e:
        print(f"❌ Ошибка при проверке PyTorch CUDA: {e}")
        return False

def check_environment_variables():
    """Проверяет переменные окружения"""
    print("\n🌍 Проверка переменных окружения...")
    
    env_vars = [
        'CUDA_VISIBLE_DEVICES',
        'NVIDIA_VISIBLE_DEVICES', 
        'NVIDIA_DRIVER_CAPABILITIES'
    ]
    
    for var in env_vars:
        value = os.getenv(var, 'не установлена')
        print(f"📊 {var}: {value}")

def check_docker_gpu():
    """Проверяет, запущен ли контейнер с GPU поддержкой"""
    print("\n🐋 Проверка Docker GPU...")
    
    try:
        # Проверяем, есть ли файлы NVIDIA в контейнере
        nvidia_files = [
            '/dev/nvidia0',
            '/dev/nvidiactl',
            '/dev/nvidia-modeset',
            '/dev/nvidia-uvm'
        ]
        
        for file_path in nvidia_files:
            if os.path.exists(file_path):
                print(f"✅ {file_path} найден")
            else:
                print(f"❌ {file_path} не найден")
                
        # Проверяем переменные окружения Docker
        if os.getenv('NVIDIA_VISIBLE_DEVICES'):
            print("✅ NVIDIA_VISIBLE_DEVICES установлена")
        else:
            print("❌ NVIDIA_VISIBLE_DEVICES не установлена")
            
    except Exception as e:
        print(f"❌ Ошибка при проверке Docker GPU: {e}")

def main():
    """Основная функция"""
    print("🚀 RadiAssist GPU Diagnostic Tool")
    print("=" * 50)
    
    # Проверяем nvidia-smi
    nvidia_available = check_nvidia_smi()
    
    # Проверяем PyTorch CUDA
    pytorch_cuda_available = check_pytorch_cuda()
    
    # Проверяем переменные окружения
    check_environment_variables()
    
    # Проверяем Docker GPU
    check_docker_gpu()
    
    print("\n" + "=" * 50)
    print("📋 ИТОГОВЫЙ СТАТУС:")
    
    if nvidia_available and pytorch_cuda_available:
        print("✅ GPU полностью доступен и готов к работе!")
        print("🚀 Обработка будет выполняться на GPU")
    elif nvidia_available and not pytorch_cuda_available:
        print("⚠️  NVIDIA драйверы доступны, но PyTorch CUDA недоступен")
        print("🔧 Возможно, нужно переустановить PyTorch с CUDA поддержкой")
    elif not nvidia_available and pytorch_cuda_available:
        print("⚠️  PyTorch CUDA доступен, но nvidia-smi недоступен")
        print("🔧 Возможно, проблема с Docker GPU пробросом")
    else:
        print("❌ GPU недоступен")
        print("🔧 Обработка будет выполняться на CPU (медленно)")
    
    print("\n💡 Рекомендации:")
    if not nvidia_available:
        print("- Убедитесь, что NVIDIA драйверы установлены")
        print("- Проверьте, что nvidia-docker2 установлен")
        print("- Запустите контейнер с флагом --gpus all")
    if not pytorch_cuda_available:
        print("- Установите PyTorch с CUDA поддержкой")
        print("- Проверьте совместимость версий CUDA и PyTorch")

if __name__ == "__main__":
    main()
