import React, { useState, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useFileUpload } from '../hooks/useFileUpload';

export const UploadPage: React.FC = () => {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  const { uploading, progress, error, lastUpload, uploadFile } = useFileUpload();

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      await handleFileUpload(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      await handleFileUpload(e.target.files[0]);
    }
  }, []);

  const handleFileUpload = async (file: File) => {
    try {
      await uploadFile(file);

      // После успешной загрузки переходим к мониторингу задачи
      setTimeout(() => {
        navigate(`/tasks`);
      }, 1000);

    } catch (error) {
      console.error('Upload failed:', error);
      // Ошибка уже обработана в хуке
    }
  };

  const handleBrowseClick = () => {
    fileInputRef.current?.click();
  };

  const handleNewUpload = () => {
    // reset(); // Удалено, так как переменная не используется
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="main-content">
      <div className="container">
        <div className="row">
          <div className="col-lg-8 offset-lg-2">
            <div className="card">
              <div className="card-header">
                <h2 className="mb-0">
                  <i className="fas fa-upload text-primary"></i>
                  Загрузка исследования
                </h2>
              </div>

              <div className="card-body">
                {!lastUpload ? (
                  <>
                    {/* Processing Timeout Settings - ЗАКОММЕНТИРОВАНО */}
                    {/* 
                    <div className="mb-4 p-3 bg-light rounded">
                      <h5 className="mb-3">
                        <i className="fas fa-clock text-info"></i>
                        Настройки обработки
                      </h5>
                      <div className="row">
                        <div className="col-md-8">
                          <label htmlFor="timeout-input" className="form-label">
                            Лимит времени обработки (секунды)
                          </label>
                          <input
                            id="timeout-input"
                            type="number"
                            className="form-control"
                            min="60"
                            max="3600"
                            step="60"
                            value={600}
                            onChange={(e) => {}}
                            disabled={uploading}
                          />
                          <div className="form-text">
                            Рекомендуется: 600 секунд (10 минут). Максимум: 3600 секунд (1 час).
                          </div>
                        </div>
                        <div className="col-md-4">
                          <div className="mt-4">
                            <small className="text-muted">
                              <strong>Текущее значение:</strong><br />
                              10 мин 0 сек
                            </small>
                          </div>
                        </div>
                      </div>
                    </div>
                    */}

                    {/* File Upload Area */}
                    <div
                      className={`file-upload ${dragActive ? 'dragover' : ''}`}
                      onDragEnter={handleDrag}
                      onDragLeave={handleDrag}
                      onDragOver={handleDrag}
                      onDrop={handleDrop}
                      onClick={handleBrowseClick}
                    >
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept=".zip"
                        onChange={handleFileSelect}
                        style={{ display: 'none' }}
                        disabled={uploading}
                      />

                      <div className="text-center">
                        <i className="fas fa-cloud-upload-alt fa-3x text-muted mb-3"></i>
                        <h4>
                          {dragActive ? 'Отпустите файл здесь' : 'Перетащите ZIP файл сюда'}
                        </h4>
                        <p className="text-muted mb-3">
                          Или нажмите для выбора файла
                        </p>
                        <button
                          className="btn btn-primary"
                          disabled={uploading}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleBrowseClick();
                          }}
                        >
                          <i className="fas fa-folder-open"></i>
                          Выбрать файл
                        </button>
                      </div>
                    </div>

                    {/* Upload Progress */}
                    {uploading && (
                      <div className="mt-4">
                        <div className="flex justify-between mb-2">
                          <span>Загрузка файла...</span>
                          <span>{progress}%</span>
                        </div>
                        <div className="progress">
                          <div
                            className="progress-bar"
                            style={{ width: `${progress}%` }}
                          ></div>
                        </div>
                      </div>
                    )}

                    {/* Error Display */}
                    {error && (
                      <div className="alert alert-danger mt-3" role="alert">
                        <i className="fas fa-exclamation-triangle"></i>
                        {error}
                      </div>
                    )}

                    {/* File Requirements */}
                    <div className="mt-4 p-3 bg-light rounded">
                      <h5 className="mb-3">
                        <i className="fas fa-info-circle text-info"></i>
                        Требования к файлу
                      </h5>
                      <ul className="list-unstyled">
                        <li><i className="fas fa-check text-success"></i> Формат: ZIP архив</li>
                        <li><i className="fas fa-check text-success"></i> Содержимое: DICOM файлы КТ грудной клетки</li>
                        <li><i className="fas fa-check text-success"></i> Максимальный размер: 1 ГБ</li>
                        <li><i className="fas fa-check text-success"></i> Количество файлов: не ограничено</li>
                        <li><i className="fas fa-clock text-info"></i> Время обработки: автоматически определяется системой</li>
                      </ul>
                    </div>
                  </>
                ) : (
                  /* Success State */
                  <div className="text-center">
                    <div className="mb-4">
                      <i className="fas fa-check-circle fa-4x text-success"></i>
                    </div>

                    <h3 className="mb-3">Файл успешно загружен!</h3>

                    <div className="card bg-light mb-4">
                      <div className="card-body">
                        <div className="row">
                          <div className="col-md-6">
                            <strong>ID задачи:</strong><br />
                            <code>{lastUpload.task_id}</code>
                          </div>
                          <div className="col-md-6">
                            <strong>Найдено DICOM файлов:</strong><br />
                            <span className="text-primary">{lastUpload.message.match(/\d+/)?.[0] || 'N/A'}</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    <p className="text-muted">
                      Анализ начнётся автоматически. Вы можете отслеживать прогресс на странице задач.
                    </p>

                    <div className="flex gap-3 justify-center">
                      <button
                        onClick={() => navigate(`/tasks`)}
                        className="btn btn-primary"
                      >
                        <i className="fas fa-tasks"></i>
                        Перейти к задачам
                      </button>

                      <button
                        onClick={handleNewUpload}
                        className="btn btn-secondary"
                      >
                        <i className="fas fa-plus"></i>
                        Загрузить другой файл
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Help Section */}
            <div className="card mt-4">
              <div className="card-header">
                <h5 className="mb-0">
                  <i className="fas fa-question-circle text-info"></i>
                  Помощь
                </h5>
              </div>
              <div className="card-body">
                <div className="row">
                  <div className="col-md-6">
                    <h6>Как подготовить данные:</h6>
                    <ol>
                      <li>Соберите DICOM файлы КТ грудной клетки в папку</li>
                      <li>Создайте ZIP архив из этой папки</li>
                      <li>Загрузите архив через форму выше</li>
                    </ol>
                  </div>
                  <div className="col-md-6">
                    <h6>Что происходит после загрузки:</h6>
                    <ol>
                      <li>Файл проходит валидацию</li>
                      <li>Создаётся задача анализа с заданным лимитом времени</li>
                      <li>Начинается ИИ-обработка (максимум 10 мин)</li>
                      <li>Результаты сохраняются в Excel и CSV</li>
                    </ol>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
