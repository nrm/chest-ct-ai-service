#!/usr/bin/env python3
"""
КОНФИГУРИРУЕМОЕ ДВОЙНОЕ ТЕЛО:
- Выбор кейса из папки
- Параметры для управления сегментацией
- Временно убираем airways из вывода
- Настраиваемое разделение костей
"""

import sys
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('NUMEXPR_NUM_THREADS','1')
import argparse
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import zipfile
import shutil
from scipy import ndimage
from skimage.morphology import binary_closing, binary_opening, binary_erosion, binary_dilation, convex_hull_image
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import json
import itertools

# Совместимый импорт BrokenProcessPool для разных версий Python
try:
    from concurrent.futures.process import BrokenProcessPool
except Exception:  # на всякий случай, если структура модуля изменится
    class BrokenProcessPool(Exception):
        pass


# Добавляем текущую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

from ct_mip_visualization import CTVisualizer, MIPProjector, SegmentationHelper

def _apply_perm_flips(arr: np.ndarray, perm=(0,1,2), flips=(False,False,False)) -> np.ndarray:
    """Перестановка осей и флипы (всегда возвращает (Z,Y,X))."""
    a = np.transpose(arr, perm)
    if flips[0]: a = a[::-1]
    if flips[1]: a = a[:, ::-1]
    if flips[2]: a = a[:, :, ::-1]
    return a

def _auto_orient_volume(volume: np.ndarray, projector) -> tuple[np.ndarray, tuple, tuple]:
    """
    Подбирает (perm, flips) по данным, чтобы:
    - Z: сверху голова/ключицы (профиль лёгких растёт к середине)
    - Y: сзади кости (позвоночник) > спереди
    - X: правое лёгкое (для наблюдателя) больше левого
    Возвращает (volume_oriented, perm, flips). Маски потом считаем уже на orient-объёме.
    """
    Z, Y, X = volume.shape
    # грубые маски на сырых данных (без тяжёлых чисток)
    body0  = (volume > -600).astype(np.uint8)
    lungs0 = projector._compute_lung_mask_enhanced(volume, body0).astype(np.uint8)
    lungs0 = ndimage.binary_opening(lungs0, structure=np.ones((3,3,3))).astype(np.uint8)
    lungs0 = (ndimage.binary_dilation(lungs0, iterations=1)).astype(np.uint8)
    lungs0 = (lungs0 > 0).astype(np.uint8)

    def _score(v, b, l):
        z,y,x = v.shape
        # 1) сверху — тонко, к середине – больше (профиль лёгких по Z)
        prof = l.sum(axis=(1,2)).astype(np.float32)
        s1 = (prof[z//3:z//2].mean() + 1e-3) / (prof[:z//4].mean() + 1e-3)

        # 2) сзади кости плотнее (по Y)
        bone_like = ((v > 200) & (b>0)).astype(np.uint8)
        back  = bone_like[:, :y//4, :].sum() + 1
        front = bone_like[:, -y//4:, :].sum() + 1
        s2 = back / front

        # 3) правое лёгкое больше левого (по X, у наблюдателя)
        left  = l[:, :, :x//2].sum() + 1
        right = l[:, :, x//2:].sum() + 1
        s3 = right / left
        return 1.0*s1 + 0.6*s2 + 0.4*s3

    best = (None, -1e9)
    for perm in itertools.permutations((0,1,2)):
        v = np.transpose(volume, perm)
        b = np.transpose(body0,  perm)
        l = np.transpose(lungs0, perm)
        # два направления достаточно: Z (вверх/вниз) и X (лево/право), Y фиксируем по позвоночнику
        for flips in [(False,False,False), (True,False,False), (False,False,True), (True,False,True),
                      (False,True,False), (True,True,False), (False,True,True), (True,True,True)]:
            vv, bb, ll = v, b, l
            if flips[0]: vv=vv[::-1]; bb=bb[::-1]; ll=ll[::-1]
            if flips[1]: vv=vv[:, ::-1]; bb=bb[:, ::-1]; ll=ll[:, ::-1]
            if flips[2]: vv=vv[:, :, ::-1]; bb=bb[:, :, ::-1]; ll=ll[:, :, ::-1]
            sc = _score(vv, bb, ll)
            if sc > best[1]:
                best = ((perm, flips), sc)
    perm, flips = best[0]
    oriented = _apply_perm_flips(volume, perm, flips)
    return oriented, perm, flips

def _ensure_two_lungs(lung_mask: np.ndarray) -> np.ndarray:
    """
    Если после сегментации получилась одна компонента, делим по X на две:
    - ищем valley в гистограмме по X; если не нашли — делим по k-means (2 кластера по x-коорд).
    Возвращает (две компоненты) в одной маске.
    """
    lung_mask = (lung_mask > 0).astype(np.uint8)
    labeled, n = ndimage.label(lung_mask)
    if n >= 2:
        # оставим две крупнейшие
        sizes = ndimage.sum(lung_mask, labeled, index=range(1, n+1))
        keep_ids = (np.argsort(sizes)[::-1][:2] + 1).tolist()
        out = np.zeros_like(lung_mask)
        for cid in keep_ids: out[labeled==cid] = 1
        return out

    # одна компонента -> делим
    z,y,x = np.where(lung_mask)
    if x.size < 100:    # слишком мало — вернём как есть
        return lung_mask
    hist, edges = np.histogram(x, bins=min(64, lung_mask.shape[2]//4 + 1))
    # valley как минимум между двумя максимумами
    peaks_idx = np.argsort(hist)[-2:]
    left_p, right_p = min(peaks_idx), max(peaks_idx)
    if right_p - left_p >= 2:
        valley = np.argmin(hist[left_p:right_p+1]) + left_p
        thr = edges[valley]
    else:
        # k-means 1D по x
        c1, c2 = np.percentile(x, 25), np.percentile(x, 75)
        for _ in range(6):
            d1 = np.abs(x - c1); d2 = np.abs(x - c2)
            g1 = d1 <= d2
            if g1.sum()==0 or (~g1).sum()==0: break
            c1 = x[g1].mean(); c2 = x[~g1].mean()
        thr = 0.5*(c1 + c2)

    left_mask  = np.zeros_like(lung_mask);  left_mask[:, :, :int(thr)]  = 1
    right_mask = np.zeros_like(lung_mask); right_mask[:, :, int(thr):] = 1
    out = (lung_mask & (left_mask | right_mask)).astype(np.uint8)
    # финальная аккур. очистка
    out = ndimage.binary_opening(out, structure=np.ones((3,3,3))).astype(np.uint8)
    return out



def build_thoracic_container_from_body_and_bone(body_mask, bone_mask):
    """Грубая оценка внутреннего объёма грудной клетки на основе тела и (опционально) костей."""
    Z, Y, X = body_mask.shape
    thorax = np.zeros_like(body_mask, dtype=np.uint8)
    for z in range(Z):
        sl = body_mask[z].astype(bool)
        if sl.sum() == 0:
            continue
        # мягкий корпус тела
        base = ndimage.binary_opening(sl, structure=np.ones((5,5)))
        # костный каркас (если есть)
        if bone_mask is not None and bone_mask.sum() > 0:
            bone = ndimage.binary_dilation(bone_mask[z].astype(bool), iterations=2)
            base = np.logical_and(base, ndimage.binary_fill_holes(bone | base))
        # сильное закрытие + заливка
        base = ndimage.binary_closing(base, structure=np.ones((15,15)))
        base = ndimage.binary_fill_holes(base)
        thorax[z] = base.astype(np.uint8)
    # 3D сглаживание
    thorax = ndimage.binary_closing(thorax, structure=np.ones((3,7,7))).astype(np.uint8)
    thorax = remove_small_components(thorax, min_voxels=20_000)
    return thorax

def configurable_dual_body():
    """Конфигурируемое двойное тело с параметрами"""
    
    # Парсинг аргументов
    parser = argparse.ArgumentParser(description='Конфигурируемая сегментация CT')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Путь к папке с кейсами')
    parser.add_argument('--case', type=str, default=None,
                       help='Название кейса (папки). Если не указан - обрабатываются все кейсы')
    parser.add_argument('--separate_bones', action='store_true',
                       help='Выделять ли кости отдельно от мягких тканей')
    parser.add_argument('--divide_bones', action='store_true',
                       help='Разделять ли кости на позвоночник и рёбра')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Папка для результатов (по умолчанию: ./visualizations)')
    parser.add_argument('--jobs', type=int, default=1,
                       help='Количество параллельных задач (по умолчанию: 1)')
    
    args = parser.parse_args()
    
    # Настройка путей
    data_root = Path(args.data_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / 'visualizations'
    
    # Определяем кейсы для обработки (папки и/или ZIP)
    cases_info = _prepare_cases(data_root, args.case, output_dir)
    if not cases_info:
        return False

    print("🔧 КОНФИГУРИРУЕМОЕ ДВОЙНОЕ ТЕЛО")
    if len(cases_info) == 1:
        print(f"📁 Кейс: {cases_info[0]['case_name']}")
    else:
        print(f"📁 Кейсы: {len(cases_info)} ({', '.join(c['case_name'] for c in cases_info)})")
    print(f"📂 Данные: {data_root}")
    print(f"📤 Результаты: {output_dir}")
    print(f"🦴 Выделять кости: {'Да' if args.separate_bones else 'Нет'}")
    print(f"🦴 Разделять кости: {'Да' if args.divide_bones else 'Нет'}")
    print(f"⚡ Параллельные задачи: {args.jobs}")
    print("=" * 60)

    cases = cases_info
    try:
        if len(cases) == 1:
            # Один кейс - обрабатываем напрямую
            c = cases[0]
            return process_single_case(c['case_name'], Path(c['data_dir']), output_dir, args)
        else:
            # Множественные кейсы - пакетная обработка
            return process_multiple_cases(cases, output_dir, args)
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


# -------------------------------
# Поддержка ZIP-папок с кейсами
# -------------------------------
def _extract_zip(zip_path: Path, cache_root: Path) -> Path:
    # Распаковывает zip в кэш-папку и возвращает путь к распакованной директории.
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / zip_path.stem
    if target.exists() and any(target.iterdir()):
        return target
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(target)
    top_items = [p for p in target.iterdir() if not p.name.startswith('__MACOSX')]
    if len(top_items) == 1 and top_items[0].is_dir():
        return top_items[0]
    return target

def _prepare_cases(data_root: Path, case_name, output_dir: Path):
    # Возвращает список кейсов: [{'case_name': str, 'data_dir': Path}]
    cache_root = (output_dir or (Path(__file__).parent / 'visualizations')) / '_extracted'
    cases = []
    def add_case(name: str, data_dir: Path):
        if data_dir.exists():
            cases.append({'case_name': name, 'data_dir': data_dir})
    if case_name:
        cand_dir = data_root / case_name
        cand_zip = data_root / (case_name if str(case_name).endswith('.zip') else f'{case_name}.zip')
        if cand_dir.exists() and cand_dir.is_dir():
            add_case(case_name, cand_dir)
        elif cand_zip.exists() and cand_zip.is_file():
            extracted = _extract_zip(cand_zip, cache_root)
            add_case(cand_zip.stem, extracted)
        else:
            pth = Path(case_name)
            if pth.exists() and pth.is_dir():
                add_case(pth.name, pth)
            elif pth.exists() and pth.suffix.lower() == '.zip':
                extracted = _extract_zip(pth, cache_root)
                add_case(pth.stem, extracted)
            else:
                available_dirs = [d.name for d in data_root.iterdir() if d.is_dir()]
                available_zips = [z.name for z in data_root.glob('*.zip')]
                print(f'❌ Кейс не найден: {cand_dir} / {cand_zip}')
                if available_dirs or available_zips:
                    print('Доступные кейсы:')
                    if available_dirs: print('  Папки:', ', '.join(available_dirs))
                    if available_zips: print('  ZIP:', ', '.join(available_zips))
    else:
        for d in data_root.iterdir():
            if d.is_dir():
                add_case(d.name, d)
        for z in data_root.glob('*.zip'):
            extracted = _extract_zip(z, cache_root)
            add_case(z.stem, extracted)
    return cases
def process_single_case(case_name, data_dir, output_dir, args):
    """Обрабатывает один кейс"""

    
    print(f"1. Загрузка данных для {case_name}...")
    visualizer = CTVisualizer(data_dir, output_dir)
    visualizer.load_data()
    volume = visualizer.volume
    
    print(f"   Диапазон HU: [{volume.min():.0f}, {volume.max():.0f}]")

    # volume, perm, flips = _auto_orient_volume(volume, visualizer.projector)
    # print(f"[auto_orient] perm={perm}, flips={flips}, shape={volume.shape}")
    
    # Создаём конфигурируемые маски
    print("\n2. Создание конфигурируемых масок...")
    masks = create_configurable_masks(volume, visualizer.projector, args, case_name)

    if 'lungs' in masks:
        lungs_fixed = _ensure_two_lungs(masks['lungs'])
        if lungs_fixed.sum() != masks['lungs'].sum():
            print(f"[lungs] split single component → two")
        masks['lungs'] = lungs_fixed
        # обновим soft: вычесть обновлённые лёгкие
        if 'soft' in masks:
            masks['soft'][lungs_fixed > 0] = 0
    
    # Анализируем результаты
    print("\n3. Анализ результатов...")
    analyze_configurable_results(volume, masks, args, case_name)
    
    # Создаём визуализации
    print("\n4. Создание визуализаций...")
    create_configurable_visualizations(volume, masks, visualizer.metadata, output_dir, args, case_name)
    
    print(f"\n🔧 ОБРАБОТКА {case_name} ЗАВЕРШЕНА!")
    return True


def process_multiple_cases(cases, output_dir, args):
    """Обрабатывает множественные кейсы устойчиво (без потери задач)."""
    print(f"\n🚀 ПАКЕТНАЯ ОБРАБОТКА {len(cases)} КЕЙСОВ...")
    # Подготавливаем задачи
    tasks = []
    for c in cases:
        task_args = {
            'case_name': c['case_name'],
            'data_dir': str(c['data_dir']),
            'output_dir': str(output_dir),
            'separate_bones': args.separate_bones,
            'divide_bones': args.divide_bones
        }
        tasks.append(task_args)

    start_time = time.time()
    results = []

    if args.jobs == 1:
        # Последовательная обработка
        print("   Режим: последовательная обработка")
        for i, task in enumerate(tasks, 1):
            print(f"\n📋 Обработка {i}/{len(tasks)}: {task['case_name']}")
            res = process_case_task(task)
            results.append(res)
    else:
        import concurrent.futures as cf
        max_workers = min(args.jobs, len(tasks), multiprocessing.cpu_count())
        print(f"   Режим: параллельная обработка (до {max_workers} процессов)")

        def spawn_pool(n):
            print(f"   ⚙️ создаю пул: {n} процессов (spawn)")
            return ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context('spawn'))

        pending = list(tasks)
        inflight = {}
        workers = max_workers
        executor = spawn_pool(workers)

        while pending or inflight:
            # Подать новые задачи, держим ограниченный ин-флайт
            try:
                while pending and len(inflight) < max(1, workers * 2):
                    t = pending.pop(0)
                    f = executor.submit(process_case_task, t)
                    inflight[f] = t
            except Exception as e:
                # Проблема при submit — перестроим пул и повторим
                print(f"   🔁 submit: {e} → перезапуск пула с меньшим числом процессов")
                if 't' in locals():
                    pending.insert(0, t)
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                workers = max(1, workers - 1)
                executor = spawn_pool(workers)
                continue

            if not inflight:
                continue

            done, _ = cf.wait(list(inflight.keys()), timeout=2.0, return_when=cf.FIRST_COMPLETED)
            for f in list(done):
                t = inflight.pop(f)
                cname = t.get('case_name','?')
                try:
                    r = f.result()
                    results.append(r)
                    if r and r.get('success', False):
                        print(f"✅ Завершён: {cname}")
                    else:
                        print(f"❌ Ошибка в {cname}: {r.get('error','unknown')}")
                except BrokenProcessPool as e:
                    print(f"   💥 пул сломан на '{cname}': {e}")
                    pending.insert(0, t)
                    try:
                        executor.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass
                    workers = max(1, workers - 1)
                    executor = spawn_pool(workers)
                except MemoryError as e:
                    print(f"   🧠 MemoryError на '{cname}' → уменьшаю воркеры")
                    pending.insert(0, t)
                    try:
                        executor.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass
                    workers = max(1, workers - 1)
                    executor = spawn_pool(workers)
                except Exception as e:
                    print(f"   ❌ ошибка в '{cname}': {e}")
                    results.append({'case_name': cname, 'success': False, 'error': str(e)})

        try:
            executor.shutdown(wait=True, cancel_futures=False)
        except Exception:
            pass

    # Анализ результатов
    end_time = time.time()
    total_time = end_time - start_time

    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]

    print(f"\n📊 ИТОГИ ПАКЕТНОЙ ОБРАБОТКИ:")
    print(f"   ✅ Успешно: {len(successful)}/{len(results)}")
    print(f"   ❌ Ошибки: {len(failed)}")
    print(f"   ⏱️ Общее время: {total_time:.1f} сек")
    print(f"   ⚡ Среднее время на кейс: {total_time/len(results):.1f} сек" if results else "   ⚡ Нет результатов" )

    if failed:
        print(f"\n❌ НЕУДАЧНЫЕ КЕЙСЫ:")
        for result in failed:
            print(f"   • {result.get('case_name','?')}: {result.get('error', 'Неизвестная ошибка')}")

    # Сохраняем отчёт
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_cases': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'total_time': total_time,
        'avg_time_per_case': (total_time / len(results)) if results else 0.0,
        'settings': {
            'separate_bones': args.separate_bones,
            'divide_bones': args.divide_bones,
            'jobs': args.jobs
        },
        'results': results
    }

    report_path = output_dir / f"batch_report_{int(time.time())}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📋 Отчёт сохранён: {report_path}")
    return len(failed) == 0

def process_case_task(task_args):
    """Обрабатывает один кейс в отдельном процессе"""
    
    try:
        case_name = task_args['case_name']
        data_dir = Path(task_args['data_dir'])
        output_dir = Path(task_args['output_dir'])
        
        # Создаём объект args для совместимости
        class TaskArgs:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        args = TaskArgs(**task_args)
        
        # Обрабатываем кейс
        data_dir = data_dir
        
        # Загружаем данные
        visualizer = CTVisualizer(data_dir, output_dir)
        visualizer.load_data()
        volume = visualizer.volume

        # volume, perm, flips = _auto_orient_volume(volume, visualizer.projector)
        # print(f"[auto_orient] perm={perm}, flips={flips}, shape={volume.shape}")
        
        # Создаём маски
        masks = create_configurable_masks(volume, visualizer.projector, args, case_name)

        if 'lungs' in masks:
            lungs_fixed = _ensure_two_lungs(masks['lungs'])
            if lungs_fixed.sum() != masks['lungs'].sum():
                print(f"[lungs] split single component → two")
            masks['lungs'] = lungs_fixed
            # обновим soft: вычесть обновлённые лёгкие
            if 'soft' in masks:
                masks['soft'][lungs_fixed > 0] = 0
        
        # Создаём визуализации
        create_configurable_visualizations(volume, masks, visualizer.metadata, output_dir, args, case_name)
        
        # Анализ результатов
        total_voxels = masks.get('body', (volume > -1e9)).sum()
        stats = {}
        for name, mask in masks.items():
            voxel_count = mask.sum()
            percentage = 100 * voxel_count / total_voxels
            labeled, num_components = ndimage.label(mask)
            stats[name] = {
                'voxels': int(voxel_count),
                'percentage': float(percentage),
                'components': int(num_components)
            }
        
        return {
            'case_name': case_name,
            'success': True,
            'volume_shape': volume.shape,
            'hu_range': [float(volume.min()), float(volume.max())],
            'stats': stats
        }
        
    except Exception as e:
        return {
            'case_name': task_args['case_name'],
            'success': False,
            'error': str(e)
        }

def create_configurable_masks(volume, projector, args, case_name=None):
    """Создаёт конфигурируемые маски в зависимости от параметров"""
    
    masks = {}

    # тело + выпуклая оболочка
    big_body = create_big_body_mask(volume)
    small_body = create_small_body_mask(volume)
    convex_body = create_convex_hull_body(small_body)
    masks['body'] = small_body

    # опционально — кости
    bone_final = None
    if args.separate_bones:
        bones_big = projector._compute_bone_mask_enhanced(volume, big_body)
        bone_final = clean_bone_mask_configurable((bones_big & convex_body).astype(np.uint8), volume, small_body)
        masks['bone'] = bone_final

    # контейнер грудной клетки
    thorax = build_thoracic_container_from_body_and_bone(convex_body, bone_final) if args.separate_bones else convex_body

    # лёгкие: черновая → чистка → ограничение thorax → анти-спина
    lungs_big = projector._compute_lung_mask_enhanced(volume, big_body)
    lungs_limited = (lungs_big & thorax).astype(np.uint8)
    lungs_final = clean_lung_artifacts_configurable(
        lungs_limited,
        volume,
        body_mask=small_body,
        thorax=thorax
    )

    # анти-спина: вырезаем заднюю 8% «скорлупу»
    posterior_cut = int(lungs_final.shape[1] * 0.08)  # если ось Y — зад/перед в вашем объёме
    lungs_final[:, :posterior_cut, :] = 0

    masks['lungs'] = lungs_final

    airways_big = projector._compute_airways_mask(volume, lungs_big, big_body)
    airways_limited = (airways_big & convex_body).astype(np.uint8)
    airways_final = clean_airways_configurable(airways_limited, lungs_final)

    soft_mask = small_body.copy()
    soft_mask[lungs_final > 0] = 0
    if args.separate_bones and 'bone' in masks:
        soft_mask[masks['bone'] > 0] = 0
    soft_mask[airways_final > 0] = 0
    soft_mask = binary_opening(soft_mask, footprint=np.ones((3,3,3))).astype(np.uint8)
    total_vox = int(small_body.sum())
    min_vox = 800 if total_vox < 5_000_000 else 3000
    soft_mask = remove_small_components(soft_mask, min_voxels=min_vox)
    masks['soft'] = soft_mask

    # деление костей по желанию
    if args.separate_bones and args.divide_bones:
        spine_mask, ribs_mask = separate_bones_configurable(bones_final)
        masks['spine'] = spine_mask
        masks['ribs']  = ribs_mask

    return masks

def create_big_body_mask(volume):
    """Создаёт большое тело (PERFECT алгоритм)"""
    
    print("     Создание большого тела...")
    
    # Сэмплируем данные
    sample_size = min(1000000, volume.size // 10)
    sample_indices = np.random.choice(volume.size, sample_size, replace=False)
    sample_values = volume.flat[sample_indices]
    
    # Находим порог через долину
    air_tissue_threshold = SegmentationHelper.find_valley_threshold(
        sample_values, -1000, 0, bins=100
    )
    print(f"       Порог: {air_tissue_threshold:.1f} HU")
    
    # Создаем маску
    mask = (volume > air_tissue_threshold).astype(np.uint8)
    
    # Выпуклая оболочка по слайсам
    step = max(1, mask.shape[0] // 50)
    for z in range(0, mask.shape[0], step):
        if mask[z].any():
            hull = convex_hull_image(mask[z].astype(bool))
            mask[z] = hull.astype(np.uint8)
    
    # Интерполяция
    for z in range(mask.shape[0]):
        if z % step != 0 and z > 0 and z < mask.shape[0] - 1:
            prev_z = (z // step) * step
            next_z = min(((z // step) + 1) * step, mask.shape[0] - 1)
            if prev_z != next_z:
                alpha = (z - prev_z) / (next_z - prev_z)
                mask[z] = ((1 - alpha) * mask[prev_z] + alpha * mask[next_z] > 0.5).astype(np.uint8)
    
    # Крупнейшая компонента
    mask = SegmentationHelper.get_largest_components(mask, n_components=1)
    
    # Дилатация
    mask = binary_dilation(mask, footprint=np.ones((3,3,3))).astype(np.uint8)
    
    return mask

def create_small_body_mask(volume):
    """Создаёт маленькое тело (final_fix_segmentation алгоритм)"""
    
    print("     Создание маленького тела...")
    
    body_threshold = -280
    print(f"       Порог: {body_threshold} HU")
    
    body_mask = (volume > body_threshold).astype(np.uint8)
    
    # Морфологическая очистка
    body_mask = binary_opening(body_mask, footprint=np.ones((3,3,3))).astype(np.uint8)
    
    # Крупнейшая компонента
    body_mask = SegmentationHelper.get_largest_components(body_mask, n_components=1)
    
    # Выпуклая оболочка для центральных слайсов
    z_center = body_mask.shape[0] // 2
    z_range = body_mask.shape[0] // 4
    
    for z in range(max(0, z_center - z_range), min(body_mask.shape[0], z_center + z_range), 10):
        if body_mask[z].any():
            hull = convex_hull_image(body_mask[z].astype(bool))
            body_mask[z] = hull.astype(np.uint8)
    
    # Интерполяция
    for z in range(body_mask.shape[0]):
        if z % 10 != 0:
            if z > 0 and z < body_mask.shape[0] - 1:
                body_mask[z] = ((body_mask[z-1].astype(float) + 
                               body_mask[z].astype(float) + 
                               body_mask[z+1].astype(float)) / 3 > 0.5).astype(np.uint8)
    
    return body_mask

def create_convex_hull_body(small_body):
    """Создаёт полную выпуклую оболочку маленького тела"""
    
    print("     Создание выпуклой оболочки...")
    
    convex_body = np.zeros_like(small_body)
    
    # Выпуклая оболочка для каждого слайса
    for z in range(small_body.shape[0]):
        if small_body[z].any():
            hull = convex_hull_image(small_body[z].astype(bool))
            convex_body[z] = hull.astype(np.uint8)
    
    return convex_body

from scipy import ndimage
import numpy as np

def clean_lung_artifacts_configurable(lung_mask: np.ndarray,
                                      volume: np.ndarray,
                                      body_mask: np.ndarray | None = None,
                                      thorax: np.ndarray | None = None) -> np.ndarray:
    """
    Чистка лёгких с гарантией БИЛАТЕРАЛЬНОСТИ:
    - сначала мягкая морфология, без жёстких отсечек,
    - определяем срединную плоскость (midline),
    - на КАЖДОЙ стороне выбираем крупнейшую валидную компоненту (не теряем «малое» лёгкое),
    - обрезаем трахею сверху.
    """
    if lung_mask is None or lung_mask.sum() == 0:
        return (lung_mask > 0).astype(np.uint8)

    lm0 = (lung_mask > 0).astype(np.uint8)

    # --- базовый "корпус" для midline ---
    if body_mask is None:
        body_mask = (volume > -600).astype(np.uint8)   # грубая маска тела
    if thorax is not None:
        body_mask = (body_mask & (thorax > 0)).astype(np.uint8)

    # --- мягкая чистка, без жёстких отсечек ---
    lm = ndimage.binary_opening(lm0, structure=np.ones((3,3,3))).astype(np.uint8)
    lm = ndimage.binary_closing(lm, structure=np.ones((3,3,3))).astype(np.uint8)

    # --- midline по телу (надежнее, чем по лёгким) ---
    yz, yy, yx = np.where(body_mask > 0)
    if yx.size == 0:
        x_mid = lm.shape[2] // 2
    else:
        x_mid = int(np.median(yx))

    # --- кандидаты-компоненты ---
    labeled, num = ndimage.label(lm)
    if num == 0:
        return lm

    comps = []
    for cid in range(1, num+1):
        comp = (labeled == cid)
        size = int(comp.sum())
        if size == 0: 
            continue
        zc, yc, xc = np.where(comp)
        mean_hu = float(volume[comp].mean())
        # не жёсткие пороги: пропускаем почти всё "воздушное"
        if mean_hu < -200:
            comps.append(dict(id=cid, size=size, x_mean=float(xc.mean())))

    # если слишком «почистили» и никого не осталось — берём две крупнейшие вообще
    if not comps:
        sizes = ndimage.sum(lm, labeled, index=range(1, num+1))
        order = np.argsort(sizes)[::-1]
        keep_ids = (order[:2] + 1).tolist() if sizes.size >= 2 else ([int(order[0]+1)] if sizes.size else [])
        out = np.zeros_like(lm, dtype=np.uint8)
        for cid in keep_ids:
            out[labeled == cid] = 1
        # обрезка трахеи (аккуратно)
        z_cut = max(0, int(lm.shape[0] * 0.06))
        if z_cut: out[:z_cut] = 0
        return out

    # --- разделяем по сторонам относительно midline ---
    left  = [c for c in comps if c["x_mean"] < x_mid]
    right = [c for c in comps if c["x_mean"] >= x_mid]

    # если сторона пустая — ослабляем критерии: берём крупнейшую из всех, лежащую ближе к этой стороне
    def pick_side(candidates, fallback_pool, is_left: bool):
        if candidates:
            return max(candidates, key=lambda c: c["size"])["id"]
        # fallback: ближайшая по x_mean к левой/правой половине
        if not fallback_pool:
            return None
        target = 0 if is_left else lm.shape[2]-1
        return min(fallback_pool, key=lambda c: abs(c["x_mean"] - target))["id"]

    cid_left  = pick_side(left, comps, True)
    cid_right = pick_side(right, comps, False)

    # если обе ссылки указывают на один и тот же компонент (мост через трахею) — разрываем тонкую перемычку
    if cid_left == cid_right:
        comp = (labeled == cid_left).astype(np.uint8)
        # узкое горло обычно тонкое — уберём 3D opening покрупнее
        comp = ndimage.binary_opening(comp, structure=np.ones((5,5,5))).astype(np.uint8)
        labeled2, num2 = ndimage.label(comp)
        if num2 >= 2:
            # две крупнейшие после разрыва
            sizes2 = ndimage.sum(comp, labeled2, index=range(1, num2+1))
            keep2 = (np.argsort(sizes2)[::-1][:2] + 1).tolist()
            out = np.zeros_like(lm, dtype=np.uint8)
            for k in keep2: out[labeled2 == k] = 1
        else:
            out = comp
    else:
        out = np.zeros_like(lm, dtype=np.uint8)
        if cid_left  is not None:  out[labeled == cid_left]  = 1
        if cid_right is not None:  out[labeled == cid_right] = 1

    # финальные штрихи: чуть закрыть дырочки и обрезать трахею
    out = ndimage.binary_closing(out, structure=np.ones((3,3,3))).astype(np.uint8)
    z_cut = max(0, int(out.shape[0] * 0.06))
    if z_cut:
        out[:z_cut] = 0

    # маленькие огрехи выбросим, но НЕ агрессивно (чтобы не потерять лёгкое)
    out = remove_small_components(out.astype(np.uint8), min_voxels=1500)
    return out.astype(np.uint8)



def remove_small_components(mask, min_voxels=500, keep_top=None):
    """Удаляет мелкие связные компоненты в бинарной маске.
    min_voxels: минимальный размер компоненты (в вокселях). Компоненты меньше удаляются.
    keep_top: если задано, сохраняет только N крупнейших компонент (после фильтра по min_voxels).
    """
    if mask is None:
        return mask
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return mask
    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, index=range(1, num+1))
    sizes = np.asarray(sizes, dtype=np.int64)
    valid_ids = np.where(sizes >= int(min_voxels))[0] + 1
    if valid_ids.size == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    if keep_top is not None:
        order = np.argsort(sizes[valid_ids-1])[::-1]
        valid_ids = valid_ids[order[:int(keep_top)]]
    out = np.zeros_like(mask, dtype=np.uint8)
    for cid in valid_ids:
        out[labeled == cid] = 1
    return out

def clean_bone_mask_configurable(bone_mask, volume, body_mask):
    """Конфигурируемая очистка костей"""
    
    if bone_mask.sum() == 0:
        return bone_mask
    bone_hu_threshold = 150
    hu_filtered = ((volume > bone_hu_threshold) & (bone_mask > 0)).astype(np.uint8)
    body_limited = (hu_filtered & body_mask).astype(np.uint8)
    closed = binary_closing(body_limited, footprint=np.ones((3,3,3))).astype(np.uint8)
    for z in range(closed.shape[0]):
        if closed[z].any():
            closed[z] = ndimage.binary_fill_holes(closed[z]).astype(np.uint8)
    total_vox = int(body_mask.sum())
    min_vox = 600 if total_vox < 5_000_000 else 2000
    filtered = remove_small_components(closed, min_voxels=min_vox)
    filtered = remove_small_components(filtered, min_voxels=1, keep_top=50)
    # отсекаем наружный мусор: компоненты вне тела
    labeled, num = ndimage.label(filtered)
    out = np.zeros_like(filtered)
    for cid in range(1, num+1):
        comp = (labeled == cid)
        if not (comp & body_mask).any():
            continue    # отсекаем мусор совсем вне тела
        out[comp] = 1   # без жёстких проверок на край
    return out

def separate_bones_configurable(bone_mask):
    """Конфигурируемое разделение костей на позвоночник и рёбра"""
    
    if bone_mask.sum() == 0:
        return np.zeros_like(bone_mask), np.zeros_like(bone_mask)
    
    print("       Разделение на позвоночник и рёбра...")
    
    labeled_bones, num_components = ndimage.label(bone_mask)
    
    spine_mask = np.zeros_like(bone_mask)
    
    if num_components > 0:
        spine_candidates = []
        
        for comp_id in range(1, num_components + 1):
            comp_mask = (labeled_bones == comp_id)
            comp_size = comp_mask.sum()
            
            # Анализ геометрии
            z_coords = np.where(comp_mask)[0]
            if len(z_coords) == 0:
                continue
                
            z_span = z_coords.max() - z_coords.min() + 1
            z_coverage = z_span / bone_mask.shape[0]
            
            # Центр масс
            com = ndimage.center_of_mass(comp_mask)
            y_relative = com[1] / bone_mask.shape[1]
            x_relative = com[2] / bone_mask.shape[2]
            
            # Критерии для позвоночника
            is_spine = (
                z_coverage > 0.3 and           # Проходит через >30% высоты
                y_relative > 0.45 and          # В задней части (y > 0.45)
                0.35 < x_relative < 0.65 and   # По центру по X
                comp_size > bone_mask.sum() * 0.1  # >10% от всех костей
            )
            
            if is_spine:
                spine_candidates.append((comp_id, z_coverage, comp_size))
        
        # Выбираем лучшего кандидата
        if spine_candidates:
            spine_candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
            best_spine_id = spine_candidates[0][0]
            spine_mask = (labeled_bones == best_spine_id).astype(np.uint8)
            print(f"         Выбран позвоночник: компонента {best_spine_id}, Z-охват {spine_candidates[0][1]:.1%}")
        else:
            print("         Позвоночник не найден по критериям")
    
    # Рёбра = всё остальное
    ribs_mask = (bone_mask & (spine_mask == 0)).astype(np.uint8)
    
    return spine_mask, ribs_mask

def clean_airways_configurable(airways_mask, lung_mask):
    """Конфигурируемая очистка дыхательных путей"""
    
    if airways_mask.sum() == 0:
        return airways_mask
    
    print("       Очистка дыхательных путей...")
    
    # Убираем пересечения с лёгкими
    cleaned = (airways_mask & (lung_mask == 0)).astype(np.uint8)
    
    # Морфологическая очистка
    cleaned = binary_opening(cleaned, footprint=np.ones((2,2,2))).astype(np.uint8)
    
    # Убираем мелкие фрагменты
    labeled, num_components = ndimage.label(cleaned)
    
    for comp_id in range(1, num_components + 1):
        comp_mask = (labeled == comp_id)
        comp_size = comp_mask.sum()
        
        if comp_size < 50:  # Очень мелкие
            cleaned[comp_mask] = 0
    
    return cleaned

def analyze_configurable_results(volume, masks, args, case_name=None):
    """Анализирует результаты конфигурируемой сегментации"""
    
    total_voxels = masks.get('body', (volume > -1e9)).sum()
    for name, mask in masks.items():
        voxel_count = mask.sum()
        percentage = 100 * voxel_count / total_voxels if total_voxels else 0.0
        print(f"   {name:12}: {voxel_count:8,} вокселей ({percentage:5.1f}%)")
        labeled, num_components = ndimage.label(mask)
        if num_components <= 2:
            conn_status = "✅"
        elif num_components <= 10:
            conn_status = "⚠️"
        else:
            conn_status = "❌"
        print(f"   {'':12}  {conn_status} {num_components} компонентов\n")

def create_configurable_visualizations(volume, masks, metadata, output_dir, args, case_name=None):
    """Создаёт конфигурируемые визуализации"""
    
    # Определяем имя кейса
    case = case_name or getattr(args, 'case', 'unknown')
    
    # Сохраняем маски
    for component, mask in masks.items():
        mask_path = output_dir / f"{case}_mask_{component}_CONFIG.npy"
        np.save(mask_path, mask)
        print(f"   Сохранена маска: {mask_path.name}")
    
    # Создаём компонентные MIP
    # Определяем компоненты для визуализации (БЕЗ airways)
    main_components = ['body', 'lungs']
    if args.separate_bones:
        if args.divide_bones and 'spine' in masks and 'ribs' in masks:
            main_components.extend(['bone', 'spine', 'ribs'])
        else:
            main_components.append('bone')
    main_components.append('soft')
    
    # Фильтруем только существующие компоненты
    available_components = [comp for comp in main_components if comp in masks and masks[comp].sum() > 0]
    
    masked_volumes = {}
    for component in available_components:
        masked_vol = np.where(masks[component] > 0, volume, -1024)
        masked_volumes[component] = masked_vol
    
    projectors = {}
    for component, masked_vol in masked_volumes.items():
        projectors[component] = MIPProjector(masked_vol, metadata['spacing'])
    
    base_projector = MIPProjector(volume, metadata['spacing'])
    
    views = {
        'Аксиальная': 0,
        'Корональная': 1,
        'Сагиттальная': 2,
    }
    
    window_modes = {
        'body': 'auto',
        'lungs': 'lung',
        'bone': 'bone',
        'spine': 'bone',
        'ribs': 'bone',
        'soft': 'soft'
    }
    
    n_components = len(available_components) + 1
    fig, axes = plt.subplots(n_components, 3, figsize=(18, 6 * n_components))
    fig.suptitle(f'🔧 КОНФИГУРИРУЕМАЯ СЕГМЕНТАЦИЯ: {case}', 
                fontsize=16, fontweight='bold', color='darkblue')
    
    # Исходный том
    for col, (view_name, axis) in enumerate(views.items()):
        base_img = base_projector.create_mip(axis=axis)
        base_img = base_projector.normalize_for_display(base_img, mode='auto')
        axes[0, col].imshow(base_img, cmap='gray', aspect='auto')
        axes[0, col].set_title(f'{view_name} (исходный)', fontsize=12, fontweight='bold')
        axes[0, col].axis('off')
    
    # Компонентные MIP
    for row, component in enumerate(available_components, 1):
        projector = projectors[component]
        window_mode = window_modes.get(component, 'auto')
        
        for col, (view_name, axis) in enumerate(views.items()):
            comp_img = projector.create_mip(axis=axis)
            comp_img = base_projector.normalize_for_display(comp_img, mode=window_mode)
            axes[row, col].imshow(comp_img, cmap='gray', aspect='auto')
            
            voxel_count = masks[component].sum()
            title = f'{view_name} ({component})\n{voxel_count:,} вокселей 🔧'
            axes[row, col].set_title(title, fontsize=11, color='darkblue', fontweight='bold')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    out = output_dir / f"{case}_component_mips_CONFIG.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   🔧 КОНФИГУРИРУЕМЫЕ MIP: {out.name}")

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    success = configurable_dual_body()
    sys.exit(0 if success else 1)
