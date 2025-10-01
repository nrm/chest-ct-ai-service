from __future__ import annotations

import math
import os
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

from .aggregation import MedicalAggregator
from .inference import (
    extract_dicom_metadata,
    validate_input_data,
    run_covid_triage,
    run_ksl_analysis,
    run_luna_detection,
)
from .models import load_covid_model, load_luna_model
from .reporting import create_excel_output, print_resource_usage

# Optional KSL analyzer setup
KSL_AVAILABLE = False
KSL_ANALYZER_CLS = None

try:
    from utils.fix_ksl_analyzer import FixedKSLAnalyzer as _KSLAnalyzer

    KSL_AVAILABLE = True
    KSL_ANALYZER_CLS = _KSLAnalyzer
    print("✅ Fixed KSL analyzer imported")
except ImportError as e:
    print(f"❌ KSL analyzer not available: {e}")
    KSL_AVAILABLE = False
    KSL_ANALYZER_CLS = None


class HackathonTester:
    """End-to-end tester for hackathon datasets and validation suites."""

    def __init__(self, max_workers: int = 1, disable_validation: bool = False):
        self.test_data_path = Path("/mnt/pcephfs/lct/LCT-dataset")
        self.workspace_path = Path("/mnt/pcephfs/lct/radiassist_workspace")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.validation_sets_path = self.workspace_path / "testsets"
        self.covid_validation_threshold = 0.40
        self.max_workers = max(1, int(max_workers))
        self.disable_validation = disable_validation
        # For API use, no ground truth needed
        self.case_ground_truth = {}
        self.case_ground_truth_info = {}

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print(f"Using device: {self.device}")

        self.ksl_analyzer = self._init_ksl_analyzer()
        self.covid_model = None
        self.luna_model = None
        self._load_models()
        self.aggregator = MedicalAggregator(self.case_ground_truth, self.ksl_analyzer)
        self.hackathon_summary: Optional[dict] = None

    def _init_ksl_analyzer(self):
        if not KSL_AVAILABLE or KSL_ANALYZER_CLS is None:
            print("⚠️  KSL analyzer not available")
            return None

        try:
            analyzer = KSL_ANALYZER_CLS()
            print("✅ KSL analyzer initialized")
            return analyzer
        except Exception as exc:  # pragma: no cover - logging path
            print(f"⚠️  Failed to initialize KSL analyzer: {exc}")
            return None

    def _load_models(self):
        print("Loading trained models...")
        self.covid_model = load_covid_model(str(self.workspace_path), self.device)
        self.luna_model = load_luna_model(str(self.workspace_path), self.device)

    # ------------------------------------------------------------------
    # Core inference helpers
    # ------------------------------------------------------------------

    def test_single_case(self, zip_path: str) -> Optional[dict]:
        """Test a single ZIP case and return detailed results."""
        case_name = Path(zip_path).stem
        ground_truth_label = self.case_ground_truth.get(case_name, -1)
        print(f"\n🔍 Testing case: {case_name}")

        start_time = time.time()

        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            dicom_dir = self._locate_dicom_directory(temp_dir)
            if not dicom_dir:
                return self._error_result(case_name, start_time)

            # Validate input data first (unless disabled)
            if self.disable_validation:
                print("  ⚡ Validation disabled - skipping input data validation")
                is_valid = True
                metadata = extract_dicom_metadata(Path(dicom_dir))
            else:
                print("  🔍 Validating input data...")
                is_valid, metadata, validation_msg = validate_input_data(Path(dicom_dir))
                print(f"     {validation_msg}")

            if not is_valid:
                print(f"  ❌ Data validation failed for {case_name}")
                processing_time = time.time() - start_time
                return {
                    "case": case_name,
                    "study_uid": metadata.get("study_uid", "UNKNOWN"),
                    "series_uid": metadata.get("series_uid", "UNKNOWN"),
                    "patient_id": metadata.get("patient_id", "UNKNOWN"),
                    "num_slices": metadata.get("num_slices", 0),
                    "validation_result": metadata.get("validation_result", "failed"),
                    "validation_confidence": metadata.get("validation_confidence", 0.0),
                    "covid_probability": 0.5,  # Default uncertain
                    "nodule_count": 0,
                    "pathology_probability": 0.5,
                    "pathology": 1,  # Err on side of caution - flag as pathological
                    "processing_time": processing_time,
                    "ground_truth": ground_truth_label,
                    "status": "validation_failed",
                    "error": validation_msg
                }

            print("  🧠 Running COVID19 triage...")
            covid_prob, covid_status = run_covid_triage(
                self.covid_model, self.device, dicom_dir
            )

            print("  🔍 Running LUNA16 detection...")
            luna_result, luna_status = run_luna_detection(
                self.luna_model, self.device, dicom_dir
            )

            # Extract nodule data
            if isinstance(luna_result, dict):
                nodule_count = luna_result['nodule_count']
                detected_nodules = luna_result['detected_nodules']
                pathology_localization = luna_result['pathology_localization']
                luna_avg_confidence = luna_result['avg_confidence']
            else:
                # Fallback for old return format
                nodule_count = luna_result
                detected_nodules = []
                pathology_localization = None
                luna_avg_confidence = 0.0

            print("  🧬 Running KSL Z-profile analysis...")
            ksl_result, ksl_status = run_ksl_analysis(self.ksl_analyzer, zip_path)

            print("  🏥 Performing enhanced medical aggregation...")
            aggregation = self.aggregator.aggregate(
                covid_prob,
                nodule_count,
                case_name,
                ksl_result,
                ground_truth_label,
            )

            processing_time = time.time() - start_time

            result_dict = {
                "case": case_name,
                "study_uid": metadata["study_uid"] if metadata else "UNKNOWN",
                "series_uid": metadata["series_uid"] if metadata else "UNKNOWN",
                "patient_id": metadata["patient_id"] if metadata else "UNKNOWN",
                "num_slices": metadata["num_slices"] if metadata else 0,
                "covid_probability": covid_prob,
                "covid_status": covid_status,
                "nodule_count": nodule_count,
                "luna_status": luna_status,
                "detected_nodules_count": len(detected_nodules),
                "pathology_localization": pathology_localization,
                "luna_avg_confidence": luna_avg_confidence,
                "probability_of_pathology": aggregation["probability"],
                "pathology": aggregation["prediction"],
                "confidence": aggregation["confidence"],
                "ground_truth": aggregation["ground_truth"],
                "processing_time": processing_time,
                "status": "SUCCESS" if "Success" in covid_status else "PARTIAL",
                "method": aggregation.get("method", "original"),
                "reason": aggregation.get("reason", "Original medical aggregation"),
            }

            info = self.case_ground_truth_info.get(case_name)
            if info:
                result_dict["ground_truth_dataset"] = info.get("dataset")
                result_dict["ground_truth_source"] = info.get("source")
                result_dict["ground_truth_zip_file"] = info.get("zip_file")
                if "row" in info:
                    result_dict["ground_truth_details"] = info["row"]

            if ksl_result and ksl_result.get("available") and not ksl_result.get("error"):
                result_dict.update(
                    {
                        "ksl_available": True,
                        "ksl_status": ksl_status,
                        "ksl_z_profile_score": ksl_result["z_profile_score"],
                        "ksl_prediction": aggregation.get("ksl_prediction", 0),
                        "ksl_confidence": aggregation.get("ksl_confidence", 0.5),
                        "avg_lung_density": ksl_result["medical_features"].get(
                            "avg_dense_500", 0
                        ),
                        "avg_z_score": ksl_result["medical_features"].get("avg_z_score", 0),
                        "motion_artifacts": ksl_result.get("motion_artifacts", {}).get(
                            "avg_correlation", 0
                        ),
                    }
                )
            else:
                result_dict.update(
                    {
                        "ksl_available": False,
                        "ksl_status": ksl_status,
                        "ksl_z_profile_score": 0.5,
                        "ksl_prediction": 0,
                        "ksl_confidence": 0.5,
                    }
                )

            return result_dict

    def _locate_dicom_directory(self, root_dir: str) -> Optional[str]:
        for root, _, files in os.walk(root_dir):
            dicom_candidates = [
                f
                for f in files
                if f.endswith(".dcm") or ("." not in f and len(f) >= 4)
            ]
            if dicom_candidates:
                print(
                    f"  📁 Found DICOM directory: {os.path.basename(root)} "
                    f"({len(dicom_candidates)} files)"
                )
                return root
        return None

    def _error_result(self, case_name: str, start_time: float) -> dict:
        return {
            "case": case_name,
            "study_uid": "UNKNOWN",
            "series_uid": "UNKNOWN",
            "patient_id": "UNKNOWN",
            "num_slices": 0,
            "covid_probability": 0.5,
            "covid_status": "No DICOM files found",
            "nodule_count": 0,
            "luna_status": "Skipped",
            "probability_of_pathology": 0.5,
            "pathology": 0,
            "confidence": 0.0,
            "ground_truth": -1,
            "processing_time": time.time() - start_time,
            "status": "ERROR",
            "error": "No DICOM files found",
        }

    # ------------------------------------------------------------------
    # Hackathon dataset runner
    # ------------------------------------------------------------------

    def run_all_tests(self) -> List[dict]:
        dataset_path = Path(self.test_data_path)
        test_files = sorted(dataset_path.glob("*.zip"))
        results: List[dict] = []
        total_start = time.time()
        self.hackathon_summary = None

        print("🚀 Запуск тестирования RadiAssist для хакатона")
        print("=" * 60)
        print(f"📁 Тестовые данные: {self.test_data_path}")
        if test_files:
            print(f"📋 Файлы для тестирования: {', '.join(p.name for p in test_files)}")
        else:
            print("📋 Файлы для тестирования: (не найдены)")
        print("=" * 60)

        if not test_files:
            print(f"⚠️  No ZIP files found in {self.test_data_path}")
            return results

        for zip_path in test_files:
            if not zip_path.exists():
                print(f"❌ File not found: {zip_path}")
                continue

            result = self.test_single_case(str(zip_path))
            if not result:
                continue
            results.append(result)

            status_ru = "Успех" if result["status"] == "SUCCESS" else "Частично"
            pred_ru = "патология" if result["pathology"] == 1 else "норма"
            if result["ground_truth"] == 1:
                gt_ru = "патология"
            elif result["ground_truth"] == 0:
                gt_ru = "норма"
            else:
                gt_ru = "нет метки"
            correct_icon = (
                "✅"
                if result["ground_truth"] in (0, 1)
                and result["pathology"] == result["ground_truth"]
                else "⚪"
            )

            print(f"  ✅ Статус: {status_ru}")
            print(f"  📊 COVID19 вероятность: {result['covid_probability']:.3f}")
            if result.get("ksl_available", False):
                print(f"  🧬 KSL Z-оценка: {result['ksl_z_profile_score']:.3f}")
                print(
                    f"  🎯 Итоговый прогноз: {pred_ru} (истина: {gt_ru}) "
                    f"{correct_icon} - {result['method']}"
                )
                print(
                    "  🔬 Только KSL: "
                    f"{'патология' if result.get('ksl_prediction', 0) == 1 else 'норма'}"
                )
            else:
                print(f"  🎯 Прогноз: {pred_ru} (истина: {gt_ru}) {correct_icon}")
            print(f"  ⏱️  Время: {result['processing_time']:.1f}с")

        resource_usage = print_resource_usage(torch, psutil)

        total_time = time.time() - total_start
        total_cases = len(results)
        processing_times = [r['processing_time'] for r in results]

        print("\n📋 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
        print("=" * 60)
        success_count = sum(1 for r in results if r["status"] == "SUCCESS")
        print(f"✅ Успешно обработано: {success_count}/{total_cases}")
        avg_time = total_time / total_cases if total_cases else math.nan
        min_time = min(processing_times) if processing_times else math.nan
        max_time = max(processing_times) if processing_times else math.nan
        if total_cases:
            print(f"⏱️  Время обработки: {total_time:.1f}с (среднее: {avg_time:.1f}с/случай)")
            if not (isinstance(min_time, float) and math.isnan(min_time)):
                print(f"   Разброс: min {min_time:.1f}с / max {max_time:.1f}с")
        else:
            print(f"⏱️  Время обработки: {total_time:.1f}с")

        known_results = [r for r in results if r["ground_truth"] in (0, 1)]
        tp = sum(1 for r in known_results if r["ground_truth"] == 1 and r["pathology"] == 1)
        fp = sum(1 for r in known_results if r["ground_truth"] == 0 and r["pathology"] == 1)
        tn = sum(1 for r in known_results if r["ground_truth"] == 0 and r["pathology"] == 0)
        fn = sum(1 for r in known_results if r["ground_truth"] == 1 and r["pathology"] == 0)

        if known_results:
            accuracy = (tp + tn) / len(known_results)
            sensitivity = tp / (tp + fn) if (tp + fn) else math.nan
            specificity = tn / (tn + fp) if (tn + fp) else math.nan
            precision = tp / (tp + fp) if (tp + fp) else math.nan
            f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) else math.nan

            print("\n🎯 Матрица ошибок (по известным меткам):")
            print(f"  TP: {tp}")
            print(f"  FP: {fp}")
            print(f"  TN: {tn}")
            print(f"  FN: {fn}")

            print("\n📈 Диагностические метрики:")
            print(f"  Точность (Accuracy): {accuracy:.3f} ({accuracy * 100:.1f}%)")
            if not (isinstance(sensitivity, float) and math.isnan(sensitivity)):
                print(f"  Чувствительность (Sensitivity/Recall): {sensitivity:.3f} ({sensitivity * 100:.1f}%)")
            if not (isinstance(specificity, float) and math.isnan(specificity)):
                print(f"  Специфичность (Specificity): {specificity:.3f} ({specificity * 100:.1f}%)")
            if not (isinstance(precision, float) and math.isnan(precision)):
                print(f"  Точность предсказания (Precision): {precision:.3f} ({precision * 100:.1f}%)")
            if not (isinstance(f1_score, float) and math.isnan(f1_score)):
                print(f"  F1-мера (F1 Score): {f1_score:.3f} ({f1_score * 100:.1f}%)")
        else:
            accuracy = sensitivity = specificity = precision = f1_score = math.nan
            print("\n🎯 Метрики: ground truth отсутствует, пропускаем расчёт")

        ksl_results = [r for r in results if r.get("ksl_available", False) and r["ground_truth"] in (0, 1)]
        ksl_usage = len(ksl_results)
        if ksl_results:
            print("\n🧬 Анализ KSL vs CNN:")
            ksl_correct = sum(1 for r in ksl_results if r.get("ksl_prediction", 0) == r["ground_truth"])
            cnn_correct = sum(
                1
                for r in ksl_results
                if (1 if r["covid_probability"] > 0.5 else 0) == r["ground_truth"]
            )
            print(
                f"  KSL точность: {ksl_correct}/{len(ksl_results)} "
                f"({100 * ksl_correct / len(ksl_results):.1f}%)"
            )
            print(
                f"  CNN точность: {cnn_correct}/{len(ksl_results)} "
                f"({100 * cnn_correct / len(ksl_results):.1f}%)"
            )
            avg_ksl_conf = float(np.mean([r.get("ksl_confidence", 0.5) for r in ksl_results]))
            avg_cnn_conf = float(
                np.mean([abs(r["covid_probability"] - 0.5) + 0.5 for r in ksl_results])
            )
            print(f"  Средняя уверенность KSL: {avg_ksl_conf:.3f}")
            print(f"  Средняя уверенность CNN: {avg_cnn_conf:.3f}")
        else:
            ksl_correct = 0

        if results:
            excel_path, csv_path = create_excel_output(results, str(self.workspace_path))

        print("\n📁 Итоговая информация о данных:")
        print(f"  Путь к тестовым данным: {self.test_data_path}")
        print(f"  Протестированные файлы: {[p.name for p in test_files]}")
        if results:
            print(f"  Excel отчёт сохранён в: {excel_path.name}")
            print(f"  CSV отчёт сохранён в: {csv_path.name}")
            print(f"  🔍 Легко просмотреть CSV: head {csv_path.name}")
        print("=" * 60)

        self.hackathon_summary = {
            "total": total_cases,
            "labelled_total": len(known_results),
            "correct": tp + tn,
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1_score": f1_score,
            "ksl_usage": ksl_usage,
            "avg_processing_time": avg_time,
            "total_processing_time": float(total_time) if processing_times else float("nan"),
            "min_processing_time": min_time,
            "max_processing_time": max_time,
            "resource_usage": resource_usage,
            "success_count": success_count,
            "cases_processed": total_cases,
            "dataset_title": "Hackathon",
        }

        return results


        return results

    # ------------------------------------------------------------------
    # Validation datasets
    # ------------------------------------------------------------------

    def run_validation_datasets(self, limit: int | None = None) -> Dict[str, dict]:
        if not self.validation_sets_path.exists():
            print(f"⚠️  Validation sets directory not found: {self.validation_sets_path}")
            return {}

        summary: Dict[str, dict] = {}
        friendly_names = {
            "covid19": "COVID19",
            "luna16": "LUNA16",
            "cancer": "Cancer",
            "mosmed": "MosMed",
        }
        summary_records = []

        for dataset_dir in sorted(self.validation_sets_path.glob("*_validation")):
            dataset_key = dataset_dir.name[:-11]
            dataset_title = friendly_names.get(
                dataset_key, dataset_dir.name.replace("_", " ").title()
            )
            ground_truth_path = dataset_dir / "ground_truth.csv"
            if ground_truth_path.exists():
                metrics = evaluate_zip_dataset(
                    dataset_key,
                    dataset_title,
                    self.validation_sets_path,
                    self.test_single_case,
                    limit,
                    workers=self.max_workers,
                )
                if metrics:
                    summary[dataset_key] = metrics
                    summary_records.append((dataset_key, dataset_title, metrics))
            else:
                print(
                    f"⚠️  {dataset_title} validation ground_truth.csv not found: "
                    f"{ground_truth_path}"
                )

        if "mosmed" not in summary and (self.validation_sets_path / "mosmed_validation").exists():
            mosmed_summary = summarize_mosmed(self.validation_sets_path, limit)
            if mosmed_summary:
                summary["mosmed"] = mosmed_summary
                summary_records.append(("mosmed", friendly_names.get("mosmed", "MosMed"), mosmed_summary))

        if self.hackathon_summary:
            summary["hackathon"] = self.hackathon_summary
            summary_records.insert(
                0,
                (
                    "hackathon",
                    self.hackathon_summary.get("dataset_title", "Hackathon"),
                    self.hackathon_summary,
                ),
            )

        if summary_records:
            print()
            print("📊 СВОДКА ПО ВАЛИДАЦИИ")
            print("=" * 60)
            aggregated_total_processed = 0
            aggregated_labelled_total = 0
            aggregated_correct = 0
            aggregated_total_time = 0.0
            has_time_data = False

            def _print_metric(label: str, value: float) -> None:
                if value is None:
                    return
                if isinstance(value, float) and math.isnan(value):
                    return
                print(f"  {label}: {value:.3f} ({value * 100:.1f}%)")

            for dataset_key, dataset_title, metrics in summary_records:
                print()
                print(f"• {dataset_title}")

                total = metrics.get("total", 0) or 0
                labelled_total = metrics.get("labelled_total", total) or 0
                aggregated_total_processed += total
                aggregated_labelled_total += labelled_total

                if "accuracy" in metrics:
                    print(f"  Случаев: {total}")
                    if labelled_total != total:
                        print(f"  Известных меток: {labelled_total}")
                    success_count = metrics.get("success_count")
                    cases_processed = metrics.get("cases_processed")
                    if success_count is not None and cases_processed is not None:
                        print(f"  Успешно обработано: {success_count}/{cases_processed}")
                    _print_metric("Accuracy", metrics.get("accuracy"))
                    _print_metric("Sensitivity", metrics.get("sensitivity"))
                    _print_metric("Specificity", metrics.get("specificity"))
                    _print_metric("Precision", metrics.get("precision"))
                    _print_metric("F1 Score", metrics.get("f1_score"))

                    ksl_usage = metrics.get("ksl_usage")
                    if ksl_usage is not None and labelled_total:
                        print(
                            f"  KSL usage: {ksl_usage}/{labelled_total} ({(ksl_usage / labelled_total) * 100:.1f}%)"
                        )
                else:
                    print(f"  Случаев: {total}")

                avg_time = metrics.get("avg_processing_time")
                total_time_val = metrics.get("total_processing_time")
                if isinstance(total_time_val, float) and math.isnan(total_time_val):
                    total_time_val = None
                aggregated_total_time += total_time_val or 0.0
                if total_time_val is not None:
                    has_time_data = True
                    print(f"  Total processing time: {total_time_val:.2f}s")
                if avg_time is not None and not (isinstance(avg_time, float) and math.isnan(avg_time)):
                    print(f"  Avg processing time: {avg_time:.2f}s")
                min_time = metrics.get("min_processing_time")
                max_time = metrics.get("max_processing_time")
                if (
                    min_time is not None
                    and max_time is not None
                    and not (isinstance(min_time, float) and math.isnan(min_time))
                    and not (isinstance(max_time, float) and math.isnan(max_time))
                ):
                    print(f"  Processing range: min {min_time:.2f}s / max {max_time:.2f}s")

                if "malignant" in metrics and "benign" in metrics:
                    print(f"  Malignant: {metrics['malignant']}, Benign: {metrics['benign']}")

                resource_usage = metrics.get("resource_usage")
                # Show validation failures if any
                validation_failed_count = metrics.get("validation_failed_count", 0)
                validation_failed_files = metrics.get("validation_failed_files", [])
                if validation_failed_count > 0:
                    print(f"  ⚠️  Validation failures: {validation_failed_count}")
                    if validation_failed_files:
                        print("    Failed files:")
                        for i, failed_file in enumerate(validation_failed_files[:5]):  # Show first 5
                            print(f"      • {failed_file}")
                        if len(validation_failed_files) > 5:
                            remaining = len(validation_failed_files) - 5
                            print(f"      ... и еще {remaining} файлов")

                if resource_usage:
                    cpu = resource_usage.get("cpu_rss_gb")
                    gpu = resource_usage.get("gpu")
                    print("  Ресурсы:")
                    if cpu is not None:
                        print(f"    • CPU RSS: {cpu:.2f} GB")
                    if gpu:
                        print(
                            "    • GPU {name}: alloc {alloc:.2f} GB, reserved {reserved:.2f} GB, peak {peak:.2f} GB".format(
                                name=gpu.get("name", "unknown"),
                                alloc=gpu.get("allocated_gb", 0.0),
                                reserved=gpu.get("reserved_gb", 0.0),
                                peak=gpu.get("peak_gb", 0.0),
                            )
                        )

                aggregated_correct += metrics.get("correct", 0) or 0

            if aggregated_total_processed:
                print()
                print("🏁 Итоговая сводка:")
                print(
                    f"  Случаев всего: {aggregated_total_processed}"
                    f" (с известными метками: {aggregated_labelled_total})"
                )
                if aggregated_labelled_total:
                    overall_accuracy = aggregated_correct / aggregated_labelled_total
                    print(
                        f"  Accuracy (взвешенная): {overall_accuracy:.3f} ({overall_accuracy * 100:.1f}%)"
                    )
                if has_time_data and aggregated_total_processed:
                    print(f"  Total processing time: {aggregated_total_time:.2f}s")
                    avg_overall_time = aggregated_total_time / aggregated_total_processed
                    print(f"  Avg time per кейс: {avg_overall_time:.2f}s")

        return summary

__all__ = ["HackathonTester"]
