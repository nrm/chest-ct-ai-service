import React, { useState, useEffect } from 'react';
import { apiService } from '../services/api';

interface MaskSlice {
  slice_index: number;
  filename: string;
  path: string;
  dicom_file: string;
}

interface MaskSlicesData {
  total_dicom_files: number;
  generated_slices: number;
  slices_dir: string;
  slices: MaskSlice[];
  segmentation_method?: string;
  slice_step?: number;
  calculated_slices?: number;
}

interface MaskSlicesViewerProps {
  taskId: string;
}

export const MaskSlicesViewer: React.FC<MaskSlicesViewerProps> = ({ taskId }) => {
  const [maskSlicesData, setMaskSlicesData] = useState<MaskSlicesData | null>(null);
  const [currentSliceIndex, setCurrentSliceIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [imageLoading, setImageLoading] = useState(false);
  const [bonesSegmentationRunning, setBonesSegmentationRunning] = useState(false);

  useEffect(() => {
    loadMaskSlices();
  }, [taskId]);

  const loadMaskSlices = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiService.getMaskSlices(taskId);
      setMaskSlicesData(data);
      if (data.slices.length > 0) {
        setCurrentSliceIndex(0);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load mask slices');
    } finally {
      setLoading(false);
    }
  };

  const handlePrevious = () => {
    if (maskSlicesData && currentSliceIndex > 0) {
      setCurrentSliceIndex(currentSliceIndex - 1);
    }
  };

  const handleNext = () => {
    if (maskSlicesData && currentSliceIndex < maskSlicesData.slices.length - 1) {
      setCurrentSliceIndex(currentSliceIndex + 1);
    }
  };

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setCurrentSliceIndex(parseInt(e.target.value));
  };

  const handleRunBonesSegmentation = async () => {
    try {
      setBonesSegmentationRunning(true);
      const result = await apiService.runBonesSegmentation(taskId);
      
      if (result.status === 'success') {
        // Перезагружаем слайсы после добавления костей
        await loadMaskSlices();
        alert('Сегментация костей завершена! Слайсы обновлены.');
      } else if (result.status === 'already_exists') {
        alert('Сегментация костей уже существует для этой задачи.');
      } else {
        alert(`Ошибка сегментации костей: ${result.message}`);
      }
    } catch (err) {
      alert(`Ошибка запуска сегментации костей: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setBonesSegmentationRunning(false);
    }
  };

  const getCurrentSlice = (): MaskSlice | null => {
    if (!maskSlicesData || maskSlicesData.slices.length === 0) return null;
    return maskSlicesData.slices[currentSliceIndex];
  };

  const getImageUrl = (slice: MaskSlice): string => {
    return apiService.getMaskSliceImageUrl(taskId, slice.filename);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Загрузка слайсов с масками...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
        <div className="text-red-600 mb-2">
          <svg className="w-8 h-8 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
          <p className="font-medium">Ошибка загрузки</p>
        </div>
        <p className="text-red-500 text-sm">{error}</p>
        <button
          onClick={loadMaskSlices}
          className="mt-4 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
        >
          Попробовать снова
        </button>
      </div>
    );
  }

  if (!maskSlicesData || maskSlicesData.slices.length === 0) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
        <div className="text-yellow-600 mb-2">
          <svg className="w-8 h-8 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
          <p className="font-medium">Слайсы с масками не найдены</p>
        </div>
        <p className="text-yellow-600 text-sm mb-4">
          Сегментация еще не выполнена или не завершена для этой задачи.
        </p>
        <button
          onClick={loadMaskSlices}
          className="px-4 py-2 bg-yellow-600 text-white rounded hover:bg-yellow-700 transition-colors"
        >
          Обновить
        </button>
      </div>
    );
  }

  const currentSlice = getCurrentSlice();

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      {/* Заголовок и статистика */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold text-gray-800">
            Слайсы с сегментацией
          </h3>
          <button
            onClick={handleRunBonesSegmentation}
            disabled={bonesSegmentationRunning || !maskSlicesData || maskSlicesData.generated_slices === 0}
            className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {bonesSegmentationRunning ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white inline-block mr-2"></div>
                Добавление костей...
              </>
            ) : !maskSlicesData || maskSlicesData.generated_slices === 0 ? (
              'Нет масок для обработки'
            ) : (
              'Добавить кости'
            )}
          </button>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm text-gray-600">
          <div className="bg-blue-50 p-3 rounded">
            <div className="font-medium text-blue-800">Всего DICOM файлов</div>
            <div className="text-2xl font-bold text-blue-600">{maskSlicesData.total_dicom_files}</div>
          </div>
          <div className="bg-green-50 p-3 rounded">
            <div className="font-medium text-green-800">Сгенерировано слайсов</div>
            <div className="text-2xl font-bold text-green-600">{maskSlicesData.generated_slices}</div>
          </div>
          <div className="bg-purple-50 p-3 rounded">
            <div className="font-medium text-purple-800">Текущий слайс</div>
            <div className="text-2xl font-bold text-purple-600">
              {currentSliceIndex + 1} / {maskSlicesData.slices.length}
            </div>
          </div>
          <div className="bg-orange-50 p-3 rounded">
            <div className="font-medium text-orange-800">Метод сегментации</div>
            <div className="text-sm font-bold text-orange-600">
              {maskSlicesData.segmentation_method === 'sparse' ? 'Sparse (оптимизированный)' : 'Полный'}
            </div>
            {maskSlicesData.segmentation_method === 'sparse' && (
              <div className="text-xs text-orange-500">
                Шаг: {maskSlicesData.slice_step}, Рассчитано: {maskSlicesData.calculated_slices}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Навигация по слайсам */}
      <div className="mb-6">
        <div className="flex items-center space-x-4 mb-4">
          <button
            onClick={handlePrevious}
            disabled={currentSliceIndex === 0}
            className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            ← Предыдущий
          </button>
          
          <div className="flex-1">
            <input
              type="range"
              min="0"
              max={maskSlicesData.slices.length - 1}
              value={currentSliceIndex}
              onChange={handleSliderChange}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
            />
          </div>
          
          <button
            onClick={handleNext}
            disabled={currentSliceIndex === maskSlicesData.slices.length - 1}
            className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Следующий →
          </button>
        </div>

        {currentSlice && (
          <div className="text-center text-sm text-gray-600">
            <p>Слайс {currentSlice.slice_index} • {currentSlice.dicom_file}</p>
          </div>
        )}
      </div>

      {/* Изображение слайса */}
      <div className="relative bg-gray-100 rounded-lg overflow-hidden">
        {currentSlice && (
          <>
            {imageLoading && (
              <div className="absolute inset-0 flex items-center justify-center bg-gray-100">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
                  <p className="text-gray-600">Загрузка изображения...</p>
                </div>
              </div>
            )}
            
            <img
              src={getImageUrl(currentSlice)}
              alt={`Mask slice ${currentSlice.slice_index}`}
              className="w-full h-auto max-h-96 object-contain mx-auto"
              onLoad={() => setImageLoading(false)}
              onLoadStart={() => setImageLoading(true)}
              style={{ display: imageLoading ? 'none' : 'block' }}
            />
          </>
        )}
      </div>

      {/* Легенда цветов */}
      <div className="mt-6 p-4 bg-gray-50 rounded-lg">
        <h4 className="font-medium text-gray-800 mb-3">Цветовая схема сегментации:</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
          <div className="flex items-center">
            <div className="w-4 h-4 bg-cyan-400 rounded mr-2"></div>
            <span>Легкие</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-yellow-400 rounded mr-2"></div>
            <span>Кости</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-orange-400 rounded mr-2"></div>
            <span>Позвоночник</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-green-400 rounded mr-2"></div>
            <span>Дыхательные пути</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-purple-400 rounded mr-2"></div>
            <span>Мягкие ткани</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-gray-400 rounded mr-2"></div>
            <span>Тело</span>
          </div>
        </div>
      </div>
    </div>
  );
};
