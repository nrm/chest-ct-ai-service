import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { apiService } from '../services/api';

export const HomePage: React.FC = () => {
  const [health, setHealth] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      setLoading(true);
      const healthData = await apiService.healthCheck();
      setHealth(healthData);
    } catch (error) {
      console.error('Health check failed:', error);
      setHealth(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="main-content">
      <div className="container">
        {/* Hero Section */}
        <div className="text-center mb-4 fade-in">
          <h1 className="mb-3">
            <i className="fas fa-heartbeat text-primary"></i>
            Добро пожаловать в RadiAssist
          </h1>
          <p className="text-muted" style={{ fontSize: '1.1rem', maxWidth: '600px', margin: '0 auto' }}>
            Современный ИИ-сервис для анализа компьютерных томографий органов грудной клетки.
            Автоматическое выявление патологий с высокой точностью.
          </p>
        </div>

        {/* Service Status */}
        <div className="row mb-4">
          <div className="col-md-8 offset-md-2">
            <div className="card medical-card">
              <div className="card-header">
                <h3 className="mb-0">
                  <i className="fas fa-server"></i>
                  Состояние сервиса
                </h3>
              </div>
              <div className="card-body">
                {loading ? (
                  <div className="text-center">
                    <i className="fas fa-spinner fa-spin"></i>
                    <span className="ml-2">Проверка состояния...</span>
                  </div>
                ) : health ? (
                  <div className="row">
                    <div className="col-md-6">
                      <div className="mb-3">
                        <strong>Сервис:</strong> {health.service} v{health.version}
                      </div>
                      <div className="mb-3">
                        <strong>Статус:</strong>
                        <span className={`ml-2 status-badge status-${health.status === 'healthy' ? 'completed' : 'failed'}`}>
                          {health.status === 'healthy' ? 'Работает' : 'Ошибка'}
                        </span>
                      </div>
                    </div>
                    <div className="col-md-6">
                      <div className="mb-3">
                        <strong>Активных задач:</strong> {health.config.current_processing}/{health.config.max_concurrent_jobs}
                      </div>
                      <div className="mb-3">
                        <strong>Таймаут:</strong> {health.config.processing_timeout} сек
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center text-error">
                    <i className="fas fa-exclamation-triangle"></i>
                    <span className="ml-2">Не удалось подключиться к сервису</span>
                  </div>
                )}

                <div className="text-center mt-3">
                  <button onClick={checkHealth} className="btn btn-secondary btn-sm">
                    <i className="fas fa-refresh"></i>
                    Обновить
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Features Grid */}
        <div className="row mb-4">
          <div className="col-md-4 mb-3">
            <div className="card text-center h-100">
              <div className="card-body">
                <i className="fas fa-upload fa-3x text-primary mb-3"></i>
                <h4>Загрузка исследований</h4>
                <p className="text-muted">
                  Загружайте ZIP-архивы с DICOM файлами для анализа.
                  Поддерживается до 1ГБ на исследование.
                </p>
                <Link to="/upload" className="btn btn-primary">
                  Загрузить исследование
                </Link>
              </div>
            </div>
          </div>

          <div className="col-md-4 mb-3">
            <div className="card text-center h-100">
              <div className="card-body">
                <i className="fas fa-brain fa-3x text-success mb-3"></i>
                <h4>ИИ-анализ</h4>
                <p className="text-muted">
                  Автоматическое выявление патологий с использованием
                  современных моделей машинного обучения.
                </p>
                <Link to="/tasks" className="btn btn-success">
                  Мониторить задачи
                </Link>
              </div>
            </div>
          </div>

          <div className="col-md-4 mb-3">
            <div className="card text-center h-100">
              <div className="card-body">
                <i className="fas fa-chart-bar fa-3x text-info mb-3"></i>
                <h4>Детальные результаты</h4>
                <p className="text-muted">
                  Получайте подробные отчеты в форматах Excel и CSV
                  с вероятностями и локализациями патологий.
                </p>
                <Link to="/results" className="btn btn-info">
                  Посмотреть результаты
                </Link>
              </div>
            </div>
          </div>
        </div>

        {/* Info Cards */}
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card">
              <div className="card-header">
                <h5 className="mb-0">
                  <i className="fas fa-info-circle text-primary"></i>
                  О сервисе
                </h5>
              </div>
              <div className="card-body">
                <p>
                  RadiAssist использует современные алгоритмы машинного обучения для
                  автоматического анализа компьютерных томографий органов грудной клетки.
                </p>
                <ul className="list-unstyled">
                  <li><i className="fas fa-check text-success"></i> Высокая точность выявления патологий</li>
                  <li><i className="fas fa-check text-success"></i> Быстрая обработка исследований</li>
                  <li><i className="fas fa-check text-success"></i> Подробные отчеты и визуализация</li>
                  <li><i className="fas fa-check text-success"></i> Соответствие медицинским стандартам</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card">
              <div className="card-header">
                <h5 className="mb-0">
                  <i className="fas fa-cogs text-secondary"></i>
                  Технические характеристики
                </h5>
              </div>
              <div className="card-body">
                <div className="row">
                  <div className="col-6">
                    <strong>Максимум задач:</strong><br />
                    {health?.config.max_concurrent_jobs || 'N/A'}
                  </div>
                  <div className="col-6">
                    <strong>Таймаут:</strong><br />
                    {health?.config.processing_timeout || 'N/A'} сек
                  </div>
                </div>
                <hr />
                <div className="row">
                  <div className="col-6">
                    <strong>Формат входа:</strong><br />
                    ZIP с DICOM
                  </div>
                  <div className="col-6">
                    <strong>Формат выхода:</strong><br />
                    Excel + CSV
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="text-center mt-4">
          <Link to="/upload" className="btn btn-primary btn-lg mr-3">
            <i className="fas fa-upload"></i>
            Начать анализ
          </Link>
          <Link to="/tasks" className="btn btn-secondary btn-lg">
            <i className="fas fa-tasks"></i>
            Просмотреть задачи
          </Link>
        </div>
      </div>
    </div>
  );
};
