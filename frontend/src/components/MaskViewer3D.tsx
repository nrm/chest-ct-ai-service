import React, { useState, useEffect, useRef } from 'react';
import { apiService } from '../services/api';

interface MaskViewer3DProps {
  taskId: string;
}

interface MaskMetadata {
  task_id: string;
  volume_shape: number[];
  spacing: number[];
  components: {
    [key: string]: {
      mask_3d_file: string;
      voxel_count: number;
      volume_ml: number;
    };
  };
}

export const MaskViewer3D: React.FC<MaskViewer3DProps> = ({ taskId }) => {
  const [metadata, setMetadata] = useState<MaskMetadata | null>(null);
  const [selectedComponent, setSelectedComponent] = useState<string>('lungs');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [maskData, setMaskData] = useState<{shape: number[], data: number[]} | null>(null);
  const [loadingMask, setLoadingMask] = useState(false);
  
  // Параметры для 3D визуализации
  const [rotation, setRotation] = useState({ x: 30, y: 45, z: 0 });
  const [sliceIndex, setSliceIndex] = useState(0);
  const [viewMode, setViewMode] = useState<'3d' | 'slice'>('slice');
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDragging = useRef(false);
  const lastMouse = useRef({ x: 0, y: 0 });

  useEffect(() => {
    loadMetadata();
  }, [taskId]);

  useEffect(() => {
    if (metadata && selectedComponent) {
      loadMask(selectedComponent);
    }
  }, [selectedComponent, metadata]);

  useEffect(() => {
    if (maskData && viewMode === 'slice') {
      renderSlice();
    } else if (maskData && viewMode === '3d') {
      render3D();
    }
  }, [maskData, sliceIndex, rotation, viewMode]);

  const loadMetadata = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiService.getSegmentationMetadata(taskId);
      setMetadata(data);
      
      // Выбираем первый доступный компонент
      const components = Object.keys(data.components || {});
      if (components.length > 0) {
        setSelectedComponent(components[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load segmentation metadata');
    } finally {
      setLoading(false);
    }
  };

  const loadMask = async (component: string) => {
    try {
      setLoadingMask(true);
      const data = await apiService.getSegmentationMask3D(taskId, component);
      setMaskData(data);
      
      // Устанавливаем средний срез
      if (data.shape.length > 0) {
        setSliceIndex(Math.floor(data.shape[0] / 2));
      }
    } catch (err) {
      console.error(`Failed to load mask for ${component}:`, err);
    } finally {
      setLoadingMask(false);
    }
  };

  const renderSlice = () => {
    if (!maskData || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const [, height, width] = maskData.shape;
    
    // Устанавливаем размер canvas
    canvas.width = width;
    canvas.height = height;

    // Очищаем canvas
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, width, height);

    // Получаем данные среза
    const sliceData = maskData.data.slice(
      sliceIndex * height * width,
      (sliceIndex + 1) * height * width
    );

    // Создаём ImageData для отрисовки
    const imageData = ctx.createImageData(width, height);
    
    for (let i = 0; i < sliceData.length; i++) {
      const value = sliceData[i];
      const idx = i * 4;
      
      if (value > 0) {
        // Цвет маски (зелёный с прозрачностью)
        imageData.data[idx] = 100;     // R
        imageData.data[idx + 1] = 255; // G
        imageData.data[idx + 2] = 100; // B
        imageData.data[idx + 3] = 180; // A
      } else {
        // Прозрачный фон
        imageData.data[idx + 3] = 0;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  };

  const render3D = () => {
    if (!maskData || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const [depth, height, width] = maskData.shape;
    
    // Устанавливаем размер canvas
    canvas.width = 600;
    canvas.height = 600;

    // Очищаем canvas
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Простая 3D проекция: рисуем несколько слоёв с разной прозрачностью
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const scale = Math.min(canvas.width / width, canvas.height / height) * 0.8;

    // Применяем вращение к индексам слоёв
    const cosX = Math.cos(rotation.x * Math.PI / 180);
    const sinX = Math.sin(rotation.x * Math.PI / 180);
    const cosY = Math.cos(rotation.y * Math.PI / 180);
    const sinY = Math.sin(rotation.y * Math.PI / 180);

    // Рисуем каждый N-й слой для производительности
    const step = Math.max(1, Math.floor(depth / 50));
    
    for (let z = 0; z < depth; z += step) {
      const sliceData = maskData.data.slice(
        z * height * width,
        (z + 1) * height * width
      );

      // Прозрачность зависит от глубины
      const alpha = 0.03 + (0.02 * (z / depth));
      
      for (let y = 0; y < height; y += 2) {
        for (let x = 0; x < width; x += 2) {
          const value = sliceData[y * width + x];
          
          if (value > 0) {
            // Применяем 3D трансформацию
            let x3d = x - width / 2;
            let y3d = y - height / 2;
            let z3d = z - depth / 2;

            // Вращение вокруг X
            const y1 = y3d * cosX - z3d * sinX;
            const z1 = y3d * sinX + z3d * cosX;

            // Вращение вокруг Y
            const x2 = x3d * cosY + z1 * sinY;
            // const z2 = -x3d * sinY + z1 * cosY;  // Не используется в простой проекции

            // Проекция на 2D
            const screenX = centerX + x2 * scale;
            const screenY = centerY + y1 * scale;

            // Рисуем точку
            ctx.fillStyle = `rgba(100, 255, 150, ${alpha})`;
            ctx.fillRect(screenX, screenY, 2, 2);
          }
        }
      }
    }
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    isDragging.current = true;
    lastMouse.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging.current || viewMode !== '3d') return;

    const deltaX = e.clientX - lastMouse.current.x;
    const deltaY = e.clientY - lastMouse.current.y;

    setRotation(prev => ({
      x: (prev.x + deltaY * 0.5) % 360,
      y: (prev.y + deltaX * 0.5) % 360,
      z: prev.z
    }));

    lastMouse.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseUp = () => {
    isDragging.current = false;
  };

  if (loading) {
    return (
      <div className="card">
        <div className="card-body text-center">
          <i className="fas fa-spinner fa-spin fa-3x text-primary mb-3"></i>
          <h4>Загрузка сегментации...</h4>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div className="card-body">
          <div className="alert alert-warning">
            <i className="fas fa-exclamation-triangle"></i> {error}
            <p className="mt-2 mb-0">Сегментация недоступна для этой задачи</p>
          </div>
        </div>
      </div>
    );
  }

  if (!metadata || Object.keys(metadata.components).length === 0) {
    return (
      <div className="card">
        <div className="card-body text-center">
          <i className="fas fa-images fa-3x text-muted mb-3"></i>
          <h4>Сегментация недоступна</h4>
          <p className="text-muted">Маски сегментации не были созданы для этой задачи</p>
        </div>
      </div>
    );
  }

  const components = Object.keys(metadata.components);

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex justify-between items-center">
          <h4 className="mb-0">
            <i className="fas fa-cube text-primary"></i>
            Визуализация сегментации
          </h4>
          <div className="flex gap-2">
            <button
              onClick={() => setViewMode('slice')}
              className={`btn btn-sm ${viewMode === 'slice' ? 'btn-primary' : 'btn-secondary'}`}
            >
              <i className="fas fa-layer-group"></i> 2D Срез
            </button>
            <button
              onClick={() => setViewMode('3d')}
              className={`btn btn-sm ${viewMode === '3d' ? 'btn-primary' : 'btn-secondary'}`}
            >
              <i className="fas fa-cube"></i> 3D Вид
            </button>
          </div>
        </div>
      </div>

      <div className="card-body">
        {/* Выбор компонента */}
        <div className="row mb-4">
          <div className="col-12">
            <label className="form-label"><strong>Компонент:</strong></label>
            <div className="flex gap-2 flex-wrap">
              {components.filter(comp => comp !== 'airways').map(comp => {
                const isDisabled = loadingMask || !metadata?.components[comp] || metadata.components[comp].voxel_count === 0;
                return (
                  <button
                    key={comp}
                    onClick={() => setSelectedComponent(comp)}
                    className={`btn btn-sm ${selectedComponent === comp ? 'btn-primary' : 'btn-secondary'} ${isDisabled ? 'opacity-50 cursor-not-allowed' : ''}`}
                    disabled={isDisabled}
                    title={isDisabled ? 'Маска пуста или недоступна' : ''}
                  >
                    {comp.replace('_', ' ').toUpperCase()}
                    {isDisabled && ' (пусто)'}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Canvas для отрисовки */}
        <div className="mb-4" style={{
          backgroundColor: '#1a1a2e',
          borderRadius: '8px',
          overflow: 'hidden',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '400px',
          position: 'relative'
        }}>
          {loadingMask ? (
            <div style={{ color: '#fff', textAlign: 'center' }}>
              <i className="fas fa-spinner fa-spin fa-2x mb-2"></i>
              <p>Загрузка маски...</p>
            </div>
          ) : (
            <canvas
              ref={canvasRef}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              style={{
                maxWidth: '100%',
                maxHeight: '600px',
                cursor: viewMode === '3d' ? 'grab' : 'default',
                imageRendering: 'pixelated'
              }}
            />
          )}
        </div>

        {/* Контролы для 2D режима */}
        {viewMode === 'slice' && maskData && (
          <div className="row mb-3">
            <div className="col-12">
              <label className="form-label">
                <strong>Срез:</strong> {sliceIndex + 1} / {maskData.shape[0]}
              </label>
              <input
                type="range"
                min="0"
                max={maskData.shape[0] - 1}
                value={sliceIndex}
                onChange={(e) => setSliceIndex(parseInt(e.target.value))}
                className="form-input"
                style={{ width: '100%' }}
              />
            </div>
          </div>
        )}

        {/* Контролы для 3D режима */}
        {viewMode === '3d' && (
          <div className="row mb-3">
            <div className="col-md-4">
              <label className="form-label"><strong>Вращение X:</strong> {rotation.x.toFixed(0)}°</label>
              <input
                type="range"
                min="0"
                max="360"
                value={rotation.x}
                onChange={(e) => setRotation(prev => ({ ...prev, x: parseInt(e.target.value) }))}
                className="form-input"
              />
            </div>
            <div className="col-md-4">
              <label className="form-label"><strong>Вращение Y:</strong> {rotation.y.toFixed(0)}°</label>
              <input
                type="range"
                min="0"
                max="360"
                value={rotation.y}
                onChange={(e) => setRotation(prev => ({ ...prev, y: parseInt(e.target.value) }))}
                className="form-input"
              />
            </div>
            <div className="col-md-4">
              <div className="p-3 bg-light rounded text-center">
                <strong>💡 Совет:</strong><br />
                Перетаскивайте мышью для вращения
              </div>
            </div>
          </div>
        )}

        {/* Информация о компоненте */}
        {selectedComponent && metadata.components[selectedComponent] && (
          <div className="row">
            <div className="col-md-6">
              <div className="p-3 bg-light rounded">
                <strong>Объём:</strong> {metadata.components[selectedComponent].volume_ml.toFixed(1)} мл
              </div>
            </div>
            <div className="col-md-6">
              <div className="p-3 bg-light rounded">
                <strong>Вокселей:</strong> {metadata.components[selectedComponent].voxel_count.toLocaleString()}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

