#!/usr/bin/env python3
"""
Разбиение COVID19 датасета на стратифицированные фолды для кросс-валидации.

Шаг 3 из плана: docs/covid19_implementation_plan.md
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict


def create_splits(
    file_paths: List[Path],
    labels: np.ndarray,
    n_folds: int = 5,
    holdout_ratio: float = 0.15,
    random_state: int = 42
) -> Dict:
    """
    Создаёт стратифицированные разбиения данных для кросс-валидации.

    Args:
        file_paths: Список путей к NIfTI файлам
        labels: Метки классов (0=норма, 1=патология)
        n_folds: Количество фолдов для CV
        holdout_ratio: Доля данных для hold-out теста
        random_state: Seed для воспроизводимости

    Returns:
        Dict со структурой:
        {
            'holdout': {'file_paths': [...], 'labels': [...]},
            'cv_folds': [
                {
                    'train': {'file_paths': [...], 'labels': [...]},
                    'val': {'file_paths': [...], 'labels': [...]}
                },
                ...
            ],
            'metadata': {
                'n_folds': int,
                'holdout_ratio': float,
                'random_state': int,
                'total_samples': int,
                'class_distribution': {...}
            }
        }
    """
    np.random.seed(random_state)

    # Конвертируем paths в строки для JSON-сериализации
    file_paths_str = [str(p) for p in file_paths]

    # Шаг 1: Отделяем hold-out set
    n_total = len(file_paths_str)
    n_holdout = int(n_total * holdout_ratio)

    # Стратифицированное разбиение для hold-out
    indices = np.arange(n_total)

    # Используем StratifiedShuffleSplit для одного разбиения
    from sklearn.model_selection import train_test_split

    train_val_indices, holdout_indices = train_test_split(
        indices,
        test_size=holdout_ratio,
        stratify=labels,
        random_state=random_state
    )

    # Hold-out данные
    holdout_data = {
        'file_paths': [file_paths_str[i] for i in holdout_indices],
        'labels': labels[holdout_indices].tolist()
    }

    # Данные для кросс-валидации
    cv_file_paths = [file_paths_str[i] for i in train_val_indices]
    cv_labels = labels[train_val_indices]

    # Шаг 2: Создаём k-fold CV splits
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    cv_folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(cv_file_paths, cv_labels)):
        fold_data = {
            'fold': fold_idx,
            'train': {
                'file_paths': [cv_file_paths[i] for i in train_idx],
                'labels': cv_labels[train_idx].tolist()
            },
            'val': {
                'file_paths': [cv_file_paths[i] for i in val_idx],
                'labels': cv_labels[val_idx].tolist()
            }
        }
        cv_folds.append(fold_data)

    # Шаг 3: Метаданные
    def get_class_distribution(labels_list):
        """Подсчитывает распределение классов."""
        labels_arr = np.array(labels_list)
        n_normal = int((labels_arr == 0).sum())
        n_pathology = int((labels_arr == 1).sum())
        total = len(labels_arr)
        return {
            'normal': n_normal,
            'pathology': n_pathology,
            'total': total,
            'normal_ratio': f"{n_normal/total:.2%}",
            'pathology_ratio': f"{n_pathology/total:.2%}"
        }

    metadata = {
        'n_folds': n_folds,
        'holdout_ratio': holdout_ratio,
        'random_state': random_state,
        'total_samples': n_total,
        'holdout_size': len(holdout_indices),
        'cv_size': len(train_val_indices),
        'distribution': {
            'overall': get_class_distribution(labels),
            'holdout': get_class_distribution(holdout_data['labels']),
            'cv_total': get_class_distribution(cv_labels)
        },
        'cv_folds_distribution': [
            {
                'fold': i,
                'train': get_class_distribution(fold['train']['labels']),
                'val': get_class_distribution(fold['val']['labels'])
            }
            for i, fold in enumerate(cv_folds)
        ]
    }

    return {
        'holdout': holdout_data,
        'cv_folds': cv_folds,
        'metadata': metadata
    }


def save_splits(splits: Dict, output_path: Path):
    """
    Сохраняет разбиения в JSON файл.

    Args:
        splits: Результат create_splits()
        output_path: Путь для сохранения JSON
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)

    print(f"✅ Splits сохранены: {output_path}")


def load_splits(splits_path: Path) -> Dict:
    """
    Загружает разбиения из JSON файла.

    Args:
        splits_path: Путь к JSON файлу

    Returns:
        Dict со структурой из create_splits()
    """
    with open(splits_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)

    return splits


def print_splits_summary(splits: Dict):
    """
    Выводит красивое резюме разбиений.

    Args:
        splits: Результат create_splits()
    """
    metadata = splits['metadata']

    print("=" * 80)
    print("РАЗБИЕНИЕ ДАННЫХ COVID19_1110")
    print("=" * 80)

    print(f"\n📊 Общая статистика:")
    print(f"  Всего исследований: {metadata['total_samples']}")
    print(f"  Random seed: {metadata['random_state']}")

    overall = metadata['distribution']['overall']
    print(f"\n  Распределение классов:")
    print(f"    Норма (CT-0):       {overall['normal']:4d} ({overall['normal_ratio']})")
    print(f"    Патология (CT-1-4): {overall['pathology']:4d} ({overall['pathology_ratio']})")

    print(f"\n🔬 Hold-out set ({metadata['holdout_ratio']:.0%}):")
    holdout = metadata['distribution']['holdout']
    print(f"  Размер: {metadata['holdout_size']} исследований")
    print(f"    Норма:      {holdout['normal']:4d} ({holdout['normal_ratio']})")
    print(f"    Патология:  {holdout['pathology']:4d} ({holdout['pathology_ratio']})")

    print(f"\n🔄 Кросс-валидация ({metadata['n_folds']} фолдов):")
    cv_total = metadata['distribution']['cv_total']
    print(f"  Всего для CV: {metadata['cv_size']} исследований")
    print(f"    Норма:      {cv_total['normal']:4d} ({cv_total['normal_ratio']})")
    print(f"    Патология:  {cv_total['pathology']:4d} ({cv_total['pathology_ratio']})")

    print(f"\n  Распределение по фолдам:")
    print(f"  {'Fold':>6s} | {'Train Total':>12s} | {'Train N/P':>15s} | {'Val Total':>10s} | {'Val N/P':>15s}")
    print(f"  {'-'*6}-|-{'-'*12}-|-{'-'*15}-|-{'-'*10}-|-{'-'*15}")

    for fold_dist in metadata['cv_folds_distribution']:
        fold = fold_dist['fold']
        train = fold_dist['train']
        val = fold_dist['val']

        print(f"  {fold:6d} | {train['total']:12d} | "
              f"{train['normal']:3d}/{train['pathology']:3d} "
              f"({train['normal_ratio']:>5s}/{train['pathology_ratio']:>5s}) | "
              f"{val['total']:10d} | "
              f"{val['normal']:3d}/{val['pathology']:3d} "
              f"({val['normal_ratio']:>5s}/{val['pathology_ratio']:>5s})")

    print("\n" + "=" * 80)


def verify_splits_integrity(splits: Dict) -> bool:
    """
    Проверяет целостность разбиений.

    Проверки:
    - Нет пересечений между hold-out и CV
    - Нет пересечений между train/val внутри фолдов
    - Сумма всех сэмплов = исходному датасету
    - Все file_paths уникальны в пределах каждого split

    Args:
        splits: Результат create_splits()

    Returns:
        True если все проверки пройдены
    """
    print("\n🔍 Проверка целостности разбиений...")

    # Получаем все пути из hold-out
    holdout_paths = set(splits['holdout']['file_paths'])

    # Получаем все пути из CV
    all_cv_paths = set()
    for fold in splits['cv_folds']:
        train_paths = set(fold['train']['file_paths'])
        val_paths = set(fold['val']['file_paths'])

        # Проверка: нет пересечений train/val внутри фолда
        intersection = train_paths & val_paths
        if intersection:
            print(f"  ❌ Fold {fold['fold']}: найдены пересечения train/val ({len(intersection)} файлов)")
            return False

        all_cv_paths.update(train_paths)
        all_cv_paths.update(val_paths)

    # Проверка: нет пересечений hold-out и CV
    intersection = holdout_paths & all_cv_paths
    if intersection:
        print(f"  ❌ Найдены пересечения между hold-out и CV ({len(intersection)} файлов)")
        return False

    # Проверка: сумма = total
    total_expected = splits['metadata']['total_samples']
    total_actual = len(holdout_paths) + len(all_cv_paths)

    if total_actual != total_expected:
        print(f"  ❌ Несоответствие размеров: ожидалось {total_expected}, получено {total_actual}")
        return False

    print("  ✅ Нет пересечений между hold-out и CV")
    print("  ✅ Нет пересечений между train/val внутри фолдов")
    print("  ✅ Сумма всех сэмплов совпадает с исходным датасетом")
    print("  ✅ Все проверки пройдены!")

    return True