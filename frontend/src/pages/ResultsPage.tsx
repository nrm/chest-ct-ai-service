import React, { useState, useEffect } from 'react';
import { useParams, Link, useSearchParams } from 'react-router-dom';
import { apiService } from '../services/api';
import { DicomViewer } from '../components/DicomViewer';
import { MaskViewer3D } from '../components/MaskViewer3D';
import { MaskSlicesViewer } from '../components/MaskSlicesViewer';

export const ResultsPage: React.FC = () => {
  const { taskId } = useParams<{ taskId: string }>();
  const [searchParams] = useSearchParams();
  const [results, setResults] = useState<any[]>([]);
  const [taskStatus, setTaskStatus] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'summary' | 'visualization' | 'segmentation' | 'mask-slices'>('summary');

  useEffect(() => {
    if (taskId) {
      loadResults();
    }
  }, [taskId]);

  useEffect(() => {
    const tab = searchParams.get('tab');
    if (tab && ['summary', 'visualization', 'segmentation', 'mask-slices'].includes(tab)) {
      setActiveTab(tab as any);
    }
  }, [searchParams]);

  const loadResults = async () => {
    if (!taskId) return;

    try {
      setLoading(true);
      setError(null);

      // Загружаем статус задачи
      const status = await apiService.getTaskStatus(taskId);
      setTaskStatus(status);

      // Если задача завершена, пытаемся получить результаты
      if (status.status === 'completed' && status.result_files) {
        try {
          // Здесь можно добавить логику для получения детальных результатов
          // Пока показываем базовую информацию
          setResults([]);
        } catch (err) {
          console.warn('Could not load detailed results:', err);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load results');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async (format: 'excel' | 'csv') => {
    if (!taskId) return;

    try {
      const blob = format === 'excel'
        ? await apiService.downloadExcel(taskId)
        : await apiService.downloadCSV(taskId);

      // Создаем ссылку для скачивания
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `task_${taskId}_results.${format === 'excel' ? 'xlsx' : 'csv'}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      setError(`Failed to download ${format.toUpperCase()}: ${error}`);
    }
  };

  if (loading) {
    return (
      <div className="main-content">
        <div className="container">
          <div className="text-center">
            <i className="fas fa-spinner fa-spin fa-3x text-primary mb-3"></i>
            <h3>Загрузка результатов...</h3>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="main-content">
        <div className="container">
          <div className="alert alert-danger">
            <i className="fas fa-exclamation-triangle"></i>
            {error}
          </div>
          <div className="text-center">
            <Link to="/tasks" className="btn btn-primary">
              <i className="fas fa-arrow-left"></i>
              Вернуться к задачам
            </Link>
          </div>
        </div>
      </div>
    );
  }

  if (!taskStatus) {
    return (
      <div className="main-content">
        <div className="container">
          <div className="alert alert-warning">
            <i className="fas fa-exclamation-triangle"></i>
            Задача не найдена
          </div>
          <div className="text-center">
            <Link to="/tasks" className="btn btn-primary">
              <i className="fas fa-arrow-left"></i>
              Вернуться к задачам
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="main-content">
      <div className="container">
        <div className="flex justify-between items-center mb-4">
          <h1>
            <i className="fas fa-chart-bar text-primary"></i>
            Результаты анализа
          </h1>

          <div className="flex gap-2">
            <Link to="/tasks" className="btn btn-secondary">
              <i className="fas fa-arrow-left"></i>
              К задачам
            </Link>
            <button
              onClick={loadResults}
              className="btn btn-secondary"
              disabled={loading}
            >
              <i className="fas fa-refresh"></i>
              Обновить
            </button>
          </div>
        </div>

        {/* Task Status Card */}
        <div className="card mb-4">
          <div className="card-header">
            <h3 className="mb-0">
              <i className="fas fa-info-circle"></i>
              Информация о задаче
            </h3>
          </div>
          <div className="card-body">
            <div className="row">
              <div className="col-md-6">
                <div className="mb-3">
                  <strong>ID задачи:</strong><br />
                  <code>{taskStatus.task_id}</code>
                </div>
                <div className="mb-3">
                  <strong>Статус:</strong><br />
                  <span className={`status-badge ${
                    taskStatus.status === 'completed' ? 'status-completed' :
                    taskStatus.status === 'failed' ? 'status-failed' :
                    taskStatus.status === 'processing' ? 'status-processing' : 'status-pending'
                  }`}>
                    {taskStatus.status === 'completed' && 'Завершена'}
                    {taskStatus.status === 'failed' && 'Ошибка'}
                    {taskStatus.status === 'processing' && 'Обрабатывается'}
                    {taskStatus.status === 'pending' && 'Ожидает'}
                  </span>
                </div>
              </div>
              <div className="col-md-6">
                <div className="mb-3">
                  <strong>Создано:</strong><br />
                  {new Date(taskStatus.created_at).toLocaleString('ru-RU')}
                </div>
                {taskStatus.completed_at && (
                  <div className="mb-3">
                    <strong>Завершено:</strong><br />
                    {new Date(taskStatus.completed_at).toLocaleString('ru-RU')}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Tabs */}
        {(taskStatus.status === 'processing' || taskStatus.status === 'completed') && (
          <div className="card mb-4">
            <div className="card-header">
              <div className="flex gap-3">
                <button
                  onClick={() => setActiveTab('summary')}
                  className={`btn ${activeTab === 'summary' ? 'btn-primary' : 'btn-secondary'} ${taskStatus.status !== 'completed' ? 'opacity-50 cursor-not-allowed' : ''}`}
                  disabled={taskStatus.status !== 'completed'}
                  title={taskStatus.status !== 'completed' ? 'Доступно после завершения анализа' : ''}
                >
                  <i className="fas fa-chart-line"></i> Результаты анализа
                  {taskStatus.status !== 'completed' && ' (ожидание)'}
                </button>
                <button
                  onClick={() => setActiveTab('visualization')}
                  className={`btn ${activeTab === 'visualization' ? 'btn-primary' : 'btn-secondary'}`}
                >
                  <i className="fas fa-x-ray"></i> Визуализация DICOM
                </button>
                <button
                  onClick={() => setActiveTab('segmentation')}
                  className={`btn ${activeTab === 'segmentation' ? 'btn-primary' : 'btn-secondary'}`}
                >
                  <i className="fas fa-cube"></i> 3D Сегментация
                </button>
                <button
                  onClick={() => setActiveTab('mask-slices')}
                  className={`btn ${activeTab === 'mask-slices' ? 'btn-primary' : 'btn-secondary'}`}
                >
                  <i className="fas fa-layer-group"></i> Слайсы с масками
                </button>
              </div>
            </div>
          </div>
        )}

            {/* Results Section */}
            {taskStatus.status === 'processing' || taskStatus.status === 'completed' ? (
          <>
            {activeTab === 'summary' ? (
            <>
            <div className="card mb-4">
              <div className="card-header">
                <h3 className="mb-0">
                  <i className="fas fa-download"></i>
                  Скачать результаты
                </h3>
              </div>
              <div className="card-body">
                <p className="text-muted mb-4">
                  Результаты анализа сохранены в форматах Excel и CSV для удобства работы.
                </p>

                <div className="row">
                  <div className="col-md-6">
                    <div className="card border-success">
                      <div className="card-body text-center">
                        <i className="fas fa-file-excel fa-3x text-success mb-3"></i>
                        <h5>Excel отчет</h5>
                        <p className="text-muted">Полный отчет с форматированием</p>
                        <button
                          onClick={() => handleDownload('excel')}
                          className="btn btn-success"
                        >
                          <i className="fas fa-download"></i>
                          Скачать Excel
                        </button>
                      </div>
                    </div>
                  </div>

                  <div className="col-md-6">
                    <div className="card border-info">
                      <div className="card-body text-center">
                        <i className="fas fa-file-csv fa-3x text-info mb-3"></i>
                        <h5>CSV данные</h5>
                        <p className="text-muted">Данные для обработки в программах</p>
                        <button
                          onClick={() => handleDownload('csv')}
                          className="btn btn-info"
                        >
                          <i className="fas fa-download"></i>
                          Скачать CSV
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Analysis Summary */}
            <div className="card mb-4">
              <div className="card-header">
                <h3 className="mb-0">
                  <i className="fas fa-chart-line"></i>
                  Краткие результаты
                </h3>
              </div>
              <div className="card-body">
                <div className="row text-center">
                  <div className="col-md-4">
                    <div className="p-3">
                      <i className="fas fa-clock fa-2x text-primary mb-2"></i>
                      <div className="text-2xl font-bold">
                        {taskStatus.result_data?.processing_time ?
                          `${Math.round(taskStatus.result_data.processing_time)}с` :
                          'N/A'
                        }
                      </div>
                      <div className="text-muted">Время обработки</div>
                    </div>
                  </div>

                  <div className="col-md-4">
                    <div className="p-3">
                      <i className="fas fa-percentage fa-2x text-success mb-2"></i>
                      <div className="text-2xl font-bold">
                        {taskStatus.result_data?.probability_of_pathology ?
                          `${(taskStatus.result_data.probability_of_pathology * 100).toFixed(1)}%` :
                          'N/A'
                        }
                      </div>
                      <div className="text-muted">Вероятность патологии</div>
                    </div>
                  </div>

                  <div className="col-md-4">
                    <div className="p-3">
                      <i className="fas fa-stethoscope fa-2x text-info mb-2"></i>
                      <div className="text-2xl font-bold">
                        {taskStatus.result_data?.pathology === 0 ? 'Норма' :
                         taskStatus.result_data?.pathology === 1 ? 'Патология' : 'N/A'}
                      </div>
                      <div className="text-muted">Заключение</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Detailed Results Table */}
            {results.length > 0 && (
              <div className="card">
                <div className="card-header">
                  <h3 className="mb-0">
                    <i className="fas fa-table"></i>
                    Детальные результаты
                  </h3>
                </div>
                <div className="card-body">
                  <div className="table-responsive">
                    <table className="table">
                      <thead>
                        <tr>
                          <th>Исследование</th>
                          <th>Вероятность патологии</th>
                          <th>Заключение</th>
                          <th>Тип патологии</th>
                          <th>Локализация</th>
                        </tr>
                      </thead>
                      <tbody>
                        {results.map((result, index) => (
                          <tr key={index}>
                            <td>{result.case}</td>
                            <td>
                              <span className={`badge ${
                                result.probability_of_pathology > 0.7 ? 'bg-danger' :
                                result.probability_of_pathology > 0.3 ? 'bg-warning' : 'bg-success'
                              }`}>
                                {(result.probability_of_pathology * 100).toFixed(1)}%
                              </span>
                            </td>
                            <td>
                              {result.pathology === 0 ? (
                                <span className="text-success">Норма</span>
                              ) : (
                                <span className="text-danger">Патология</span>
                              )}
                            </td>
                            <td>{result.most_dangerous_pathology_type || 'Не определено'}</td>
                            <td>
                              {result.pathology_localization ?
                                <small className="text-muted">{result.pathology_localization}</small> :
                                'Не указана'
                              }
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
            </> 
            ) : activeTab === 'visualization' ? (
              /* Visualization Tab */
              <DicomViewer taskId={taskId!} />
                ) : activeTab === 'segmentation' ? (
                  /* Segmentation Tab */
                  taskStatus.segmentation ? (
                    <MaskViewer3D taskId={taskId!} />
                  ) : (
                    <div className="card">
                      <div className="card-body text-center">
                        <i className="fas fa-cube fa-3x text-muted mb-3"></i>
                        <h4>3D Сегментация</h4>
                        <p className="text-muted">Сегментация еще не завершена. Пожалуйста, подождите.</p>
                        <div className="mt-3">
                          <i className="fas fa-spinner fa-spin text-primary"></i>
                          <span className="ml-2">Обработка...</span>
                        </div>
                      </div>
                    </div>
                  )
                ) : activeTab === 'mask-slices' ? (
                  /* Mask Slices Tab */
                  taskStatus.segmentation ? (
                    <MaskSlicesViewer taskId={taskId!} />
                  ) : (
                    <div className="card">
                      <div className="card-body text-center">
                        <i className="fas fa-layer-group fa-3x text-muted mb-3"></i>
                        <h4>Слайсы с масками</h4>
                        <p className="text-muted">Сегментация еще не завершена. Пожалуйста, подождите.</p>
                        <div className="mt-3">
                          <i className="fas fa-spinner fa-spin text-primary"></i>
                          <span className="ml-2">Обработка...</span>
                        </div>
                      </div>
                    </div>
                  )
                ) : (
                  /* Default to visualization */
                  <DicomViewer taskId={taskId!} />
                )}
          </>
        ) : (
          <div className="card">
            <div className="card-body text-center">
              <i className="fas fa-clock fa-3x text-muted mb-3"></i>
              <h3>Анализ еще не завершен</h3>
              <p className="text-muted">
                Результаты будут доступны после завершения обработки задачи.
              </p>
              <Link to="/tasks" className="btn btn-primary">
                <i className="fas fa-tasks"></i>
                Мониторить задачи
              </Link>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
