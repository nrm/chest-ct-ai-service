import React, { useState, useEffect, useCallback } from 'react';
import { apiService } from '../services/api';

interface ImmediateSlice {
  slice_index: number;
  filename: string;
  path: string;
  dicom_file: string;
  z_index?: number;
  window_center?: number;
  window_width?: number;
}

interface ImmediateSlicesData {
  total_dicom_files: number;
  generated_slices: number;
  slices_dir: string;
  slices: ImmediateSlice[];
}

interface ImmediateSlicesViewerProps {
  taskId: string;
}

export const ImmediateSlicesViewer: React.FC<ImmediateSlicesViewerProps> = ({ taskId }) => {
  const [slicesData, setSlicesData] = useState<ImmediateSlicesData | null>(null);
  const [currentSliceIndex, setCurrentSliceIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [imageLoading, setImageLoading] = useState(false);

  const loadSlicesData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      console.log(`Loading immediate slices for task ${taskId}...`);
      const data = await apiService.getImmediateSlices(taskId);
      console.log('Immediate slices data:', data);
      setSlicesData(data);
      if (data.slices && data.slices.length > 0) {
        setCurrentSliceIndex(0);
      }
    } catch (err) {
      console.error('Error loading immediate slices:', err);
      setError(err instanceof Error ? err.message : 'Ошибка загрузки слайсов');
    } finally {
      setLoading(false);
    }
  }, [taskId]);

  useEffect(() => {
    loadSlicesData();
  }, [loadSlicesData]);

  const goToPreviousSlice = useCallback(() => {
    if (slicesData && currentSliceIndex > 0) {
      setCurrentSliceIndex(currentSliceIndex - 1);
    }
  }, [slicesData, currentSliceIndex]);

  const goToNextSlice = useCallback(() => {
    if (slicesData && currentSliceIndex < slicesData.slices.length - 1) {
      setCurrentSliceIndex(currentSliceIndex + 1);
    }
  }, [slicesData, currentSliceIndex]);

  const goToSlice = useCallback((index: number) => {
    if (slicesData && index >= 0 && index < slicesData.slices.length) {
      setCurrentSliceIndex(index);
    }
  }, [slicesData]);

  const handleImageLoad = useCallback(() => {
    setImageLoading(false);
  }, []);

  const handleImageError = useCallback(() => {
    setImageLoading(false);
    setError('Ошибка загрузки изображения');
  }, []);

  if (loading) {
    return (
      <div className="immediate-slices-viewer p-4 bg-white rounded-lg shadow-md">
        <div className="text-center">
          <i className="fas fa-spinner fa-spin fa-2x text-blue-500"></i>
          <p className="mt-2 text-gray-600">Загрузка немедленных слайсов...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="immediate-slices-viewer p-4 bg-white rounded-lg shadow-md">
        <div className="text-center text-red-600">
          <i className="fas fa-exclamation-triangle fa-2x mb-2"></i>
          <p>{error}</p>
          <button 
            onClick={loadSlicesData}
            className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Попробовать снова
          </button>
        </div>
      </div>
    );
  }

  if (!slicesData || slicesData.slices.length === 0) {
    return (
      <div className="immediate-slices-viewer p-4 bg-white rounded-lg shadow-md">
        <div className="text-center text-gray-600">
          <i className="fas fa-image fa-2x mb-2"></i>
          <p>Немедленные слайсы не найдены</p>
        </div>
      </div>
    );
  }

  const currentSlice = slicesData.slices[currentSliceIndex];

  return (
    <div className="card" onKeyDown={(e) => {
      if (e.key === 'ArrowLeft') goToPreviousSlice();
      if (e.key === 'ArrowRight') goToNextSlice();
    }} tabIndex={0}>
      <div className="card-header">
        <div className="flex justify-between items-center">
          <h4 className="mb-0">
            <i className="fas fa-images text-primary"></i>
            Немедленные слайсы DICOM
          </h4>
          <div className="text-sm text-muted">
            {currentSliceIndex + 1} из {slicesData.slices.length}
          </div>
        </div>
      </div>

      <div className="card-body">
        {/* Навигация */}
        <div className="mb-4">
          <div className="flex items-center justify-center gap-4">
            <button
              onClick={goToPreviousSlice}
              disabled={currentSliceIndex === 0}
              className="btn btn-secondary"
              title="Предыдущий слайс (←)"
            >
              <i className="fas fa-chevron-left"></i>
            </button>

            <input
              type="range"
              min="0"
              max={slicesData.slices.length - 1}
              value={currentSliceIndex}
              onChange={(e) => goToSlice(parseInt(e.target.value))}
              className="form-input"
              style={{ flex: 1 }}
            />

            <button
              onClick={goToNextSlice}
              disabled={currentSliceIndex === slicesData.slices.length - 1}
              className="btn btn-secondary"
              title="Следующий слайс (→)"
            >
              <i className="fas fa-chevron-right"></i>
            </button>
          </div>
        </div>

        {/* Изображение */}
        <div className="text-center mb-4" style={{ position: 'relative' }}>
          {imageLoading && (
            <div style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              color: '#fff'
            }}>
              <i className="fas fa-spinner fa-spin fa-2x"></i>
            </div>
          )}
          <img
            src={apiService.getImmediateSliceImageUrl(taskId, currentSlice.filename)}
            alt={`DICOM slice ${currentSliceIndex + 1}`}
            style={{
              maxWidth: '100%',
              maxHeight: '600px',
              display: 'block',
              margin: '0 auto'
            }}
            onLoad={handleImageLoad}
            onLoadStart={handleImageLoad}
            onError={handleImageError}
          />
        </div>

        {/* Slice Info */}
        <div className="row">
          <div className="col-md-6">
            <div className="p-3 bg-light rounded">
              <strong>Текущий слайс:</strong> {currentSlice.filename}
              {currentSlice.z_index !== undefined && (
                <div><strong>Z-позиция:</strong> {currentSlice.z_index}</div>
              )}
              {currentSlice.window_center !== undefined && (
                <div><strong>Window:</strong> {currentSlice.window_center}/{currentSlice.window_width}</div>
              )}
            </div>
          </div>
          <div className="col-md-6">
            <div className="p-3 bg-light rounded text-right">
              <strong>Навигация:</strong> Клавиши ← → или ползунок
              <div><small className="text-success">⚡ Быстрый просмотр (PNG)</small></div>
            </div>
          </div>
        </div>

        {/* Quick Navigation */}
        <div className="row mt-3">
          <div className="col-12">
            <div className="flex gap-2 justify-center">
              <button
                onClick={() => goToSlice(0)}
                className="btn btn-sm btn-secondary"
              >
                <i className="fas fa-fast-backward"></i> Первый
              </button>
              <button
                onClick={() => goToSlice(Math.floor(slicesData.slices.length / 4))}
                className="btn btn-sm btn-secondary"
              >
                25%
              </button>
              <button
                onClick={() => goToSlice(Math.floor(slicesData.slices.length / 2))}
                className="btn btn-sm btn-secondary"
              >
                <i className="fas fa-play"></i> Середина
              </button>
              <button
                onClick={() => goToSlice(Math.floor(slicesData.slices.length * 3 / 4))}
                className="btn btn-sm btn-secondary"
              >
                75%
              </button>
              <button
                onClick={() => goToSlice(slicesData.slices.length - 1)}
                className="btn btn-sm btn-secondary"
              >
                <i className="fas fa-fast-forward"></i> Последний
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
