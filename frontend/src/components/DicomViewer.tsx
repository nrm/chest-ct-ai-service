import React, { useState, useEffect } from 'react';
import { apiService } from '../services/api';

interface DicomViewerProps {
  taskId: string;
}

interface SliceInfo {
  index: number;
  filename: string;
  z_index?: number;
  window_center?: number;
  window_width?: number;
  dicom_file?: string;
}

export const DicomViewer: React.FC<DicomViewerProps> = ({ taskId }) => {
  const [slices, setSlices] = useState<SliceInfo[]>([]);
  const [currentSliceIndex, setCurrentSliceIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [imageLoading, setImageLoading] = useState(false);

  useEffect(() => {
    loadSlices();
  }, [taskId]);

  const loadSlices = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Сначала пробуем загрузить немедленные слайсы (быстрее)
      try {
        const immediateData = await apiService.getImmediateSlices(taskId);
        const immediateSlices = immediateData.slices.map(slice => ({
          index: slice.slice_index,
          filename: slice.filename,
          z_index: slice.z_index,
          window_center: slice.window_center,
          window_width: slice.window_width,
          dicom_file: slice.dicom_file
        }));
        setSlices(immediateSlices);
        if (immediateSlices.length > 0) {
          setCurrentSliceIndex(Math.floor(immediateSlices.length / 2));
        }
        console.log(`✅ Loaded ${immediateSlices.length} immediate slices for fast viewing`);
        return;
      } catch (immediateErr) {
        console.log('⚠️ Immediate slices not available, falling back to regular slices');
      }
      
      // Fallback к обычным слайсам
      const data = await apiService.getTaskSlices(taskId);
      setSlices(data.slices);
      if (data.slices.length > 0) {
        setCurrentSliceIndex(Math.floor(data.slices.length / 2));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load slices');
    } finally {
      setLoading(false);
    }
  };

  const handlePrevious = () => {
    if (currentSliceIndex > 0) {
      setCurrentSliceIndex(currentSliceIndex - 1);
    }
  };

  const handleNext = () => {
    if (currentSliceIndex < slices.length - 1) {
      setCurrentSliceIndex(currentSliceIndex + 1);
    }
  };

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setCurrentSliceIndex(parseInt(e.target.value));
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft') {
      handlePrevious();
    } else if (e.key === 'ArrowRight') {
      handleNext();
    }
  };

  if (loading) {
    return (
      <div className="card">
        <div className="card-body text-center">
          <i className="fas fa-spinner fa-spin fa-3x text-primary mb-3"></i>
          <h4>Загрузка слайсов...</h4>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div className="card-body">
          <div className="alert alert-danger">
            <i className="fas fa-exclamation-triangle"></i> {error}
          </div>
        </div>
      </div>
    );
  }

  if (slices.length === 0) {
    return (
      <div className="card">
        <div className="card-body text-center">
          <i className="fas fa-images fa-3x text-muted mb-3"></i>
          <h4>Нет доступных слайсов</h4>
          <p className="text-muted">Слайсы DICOM не найдены для этой задачи</p>
        </div>
      </div>
    );
  }

  // Определяем URL изображения в зависимости от типа слайсов
  const currentImageUrl = slices.length > 0 && slices[0].z_index !== undefined 
    ? apiService.getImmediateSliceImageUrl(taskId, slices[currentSliceIndex].filename)
    : apiService.getSliceImageUrl(taskId, currentSliceIndex);

  return (
    <div className="card" onKeyDown={handleKeyDown} tabIndex={0}>
      <div className="card-header">
        <div className="flex justify-between items-center">
          <h4 className="mb-0">
            <i className="fas fa-x-ray text-primary"></i>
            Визуализация DICOM
          </h4>
          <div className="text-muted">
            Слайс {currentSliceIndex + 1} из {slices.length}
          </div>
        </div>
      </div>

      <div className="card-body">
        {/* Image Display */}
        <div className="mb-4" style={{
          backgroundColor: '#000',
          borderRadius: '8px',
          overflow: 'hidden',
          position: 'relative',
          minHeight: '400px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
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
            src={currentImageUrl}
            alt={`DICOM slice ${currentSliceIndex + 1}`}
            style={{
              maxWidth: '100%',
              maxHeight: '600px',
              display: 'block',
              margin: '0 auto'
            }}
            onLoad={() => setImageLoading(false)}
            onLoadStart={() => setImageLoading(true)}
          />
        </div>

        {/* Controls */}
        <div className="row mb-3">
          <div className="col-12">
            <div className="flex gap-2 items-center">
              <button
                onClick={handlePrevious}
                disabled={currentSliceIndex === 0}
                className="btn btn-secondary"
                title="Предыдущий слайс (←)"
              >
                <i className="fas fa-chevron-left"></i>
              </button>

              <input
                type="range"
                min="0"
                max={slices.length - 1}
                value={currentSliceIndex}
                onChange={handleSliderChange}
                className="form-input"
                style={{ flex: 1 }}
              />

              <button
                onClick={handleNext}
                disabled={currentSliceIndex === slices.length - 1}
                className="btn btn-secondary"
                title="Следующий слайс (→)"
              >
                <i className="fas fa-chevron-right"></i>
              </button>
            </div>
          </div>
        </div>

        {/* Slice Info */}
        <div className="row">
          <div className="col-md-6">
            <div className="p-3 bg-light rounded">
              <strong>Текущий слайс:</strong> {slices[currentSliceIndex]?.filename}
              {slices[currentSliceIndex]?.z_index !== undefined && (
                <div><strong>Z-позиция:</strong> {slices[currentSliceIndex].z_index}</div>
              )}
              {slices[currentSliceIndex]?.window_center !== undefined && (
                <div><strong>Window:</strong> {slices[currentSliceIndex].window_center}/{slices[currentSliceIndex].window_width}</div>
              )}
            </div>
          </div>
          <div className="col-md-6">
            <div className="p-3 bg-light rounded text-right">
              <strong>Навигация:</strong> Клавиши ← → или ползунок
              {slices.length > 0 && slices[0].z_index !== undefined && (
                <div><small className="text-success">⚡ Быстрый просмотр (PNG)</small></div>
              )}
            </div>
          </div>
        </div>

        {/* Quick Navigation */}
        <div className="row mt-3">
          <div className="col-12">
            <div className="flex gap-2 justify-center">
              <button
                onClick={() => setCurrentSliceIndex(0)}
                className="btn btn-sm btn-secondary"
              >
                <i className="fas fa-fast-backward"></i> Первый
              </button>
              <button
                onClick={() => setCurrentSliceIndex(Math.floor(slices.length / 4))}
                className="btn btn-sm btn-secondary"
              >
                25%
              </button>
              <button
                onClick={() => setCurrentSliceIndex(Math.floor(slices.length / 2))}
                className="btn btn-sm btn-secondary"
              >
                50%
              </button>
              <button
                onClick={() => setCurrentSliceIndex(Math.floor(slices.length * 3 / 4))}
                className="btn btn-sm btn-secondary"
              >
                75%
              </button>
              <button
                onClick={() => setCurrentSliceIndex(slices.length - 1)}
                className="btn btn-sm btn-secondary"
              >
                Последний <i className="fas fa-fast-forward"></i>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

