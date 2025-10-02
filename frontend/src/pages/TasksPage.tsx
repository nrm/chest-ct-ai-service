import React from 'react';
import { useTasks } from '../hooks/useTasks';
import { Link } from 'react-router-dom';

export const TasksPage: React.FC = () => {
  const { tasks, loading, error, refreshTask, downloadResults } = useTasks();

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending': return <i className="fas fa-clock text-warning"></i>;
      case 'processing': return <i className="fas fa-cog fa-spin text-primary"></i>;
      case 'completed': return <i className="fas fa-check-circle text-success"></i>;
      case 'failed': return <i className="fas fa-times-circle text-error"></i>;
      default: return <i className="fas fa-question-circle text-muted"></i>;
    }
  };

  const getStatusBadge = (status: string) => {
    const statusMap = {
      'pending': 'status-pending',
      'processing': 'status-processing',
      'completed': 'status-completed',
      'failed': 'status-failed'
    };
    return statusMap[status as keyof typeof statusMap] || 'status-pending';
  };

  const formatTime = (isoString: string) => {
    try {
      return new Date(isoString).toLocaleString('ru-RU');
    } catch {
      return isoString;
    }
  };

  const handleDownload = async (taskId: string, format: 'excel' | 'csv') => {
    try {
      await downloadResults(taskId, format);
    } catch (error) {
      console.error(`Failed to download ${format}:`, error);
    }
  };

  if (loading && tasks.length === 0) {
    return (
      <div className="main-content">
        <div className="container">
          <div className="text-center">
            <i className="fas fa-spinner fa-spin fa-3x text-primary mb-3"></i>
            <h3>Загрузка задач...</h3>
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
        </div>
      </div>
    );
  }

  return (
    <div className="main-content">
      <div className="container">
        <div className="flex justify-between items-center mb-4">
          <h1>
            <i className="fas fa-tasks text-primary"></i>
            Задачи анализа
          </h1>

          <div className="flex gap-2">
            <Link to="/upload" className="btn btn-primary">
              <i className="fas fa-plus"></i>
              Новая загрузка
            </Link>
            <button
              onClick={() => window.location.reload()}
              className="btn btn-secondary"
              disabled={loading}
            >
              <i className="fas fa-refresh"></i>
              Обновить
            </button>
          </div>
        </div>

        {tasks.length === 0 ? (
          <div className="card text-center">
            <div className="card-body">
              <i className="fas fa-inbox fa-3x text-muted mb-3"></i>
              <h3>Нет активных задач</h3>
              <p className="text-muted">
                Загрузите исследование для начала анализа
              </p>
              <Link to="/upload" className="btn btn-primary">
                <i className="fas fa-upload"></i>
                Загрузить исследование
              </Link>
            </div>
          </div>
        ) : (
          <div className="row">
            {tasks.map((task) => (
              <div key={task.task_id} className="col-lg-6 mb-4">
                <div className="card">
                  <div className="card-header">
                    <div className="flex justify-between items-center">
                      <h5 className="mb-0">
                        {getStatusIcon(task.status)}
                        Задача {task.task_id.slice(0, 8)}...
                      </h5>
                      <span className={`status-badge ${getStatusBadge(task.status)}`}>
                        {task.status === 'pending' && 'Ожидает'}
                        {task.status === 'processing' && 'Обрабатывается'}
                        {task.status === 'completed' && 'Завершена'}
                        {task.status === 'failed' && 'Ошибка'}
                      </span>
                    </div>
                  </div>

                  <div className="card-body">
                    <div className="row mb-3">
                      <div className="col-sm-6">
                        <strong>Создано:</strong><br />
                        {formatTime(task.created_at)}
                      </div>
                      {task.completed_at && (
                        <div className="col-sm-6">
                          <strong>Завершено:</strong><br />
                          {formatTime(task.completed_at)}
                        </div>
                      )}
                    </div>

                    {task.status === 'processing' && (
                      <div className="mb-3">
                        <div className="flex justify-between mb-1">
                          <span>Прогресс анализа</span>
                          <span className="pulse">
                            <i className="fas fa-brain"></i>
                          </span>
                        </div>
                        <div className="progress">
                          <div className="progress-bar" style={{ width: '75%' }}></div>
                        </div>
                      </div>
                    )}

                    {task.status === 'completed' && (
                      <div className="mb-3">
                        <div className="row">
                          <div className="col-12">
                            <strong>Результаты анализа:</strong>
                          </div>
                          <div className="col-12 mt-2">
                            <div className="flex gap-2">
                              <button
                                onClick={() => handleDownload(task.task_id, 'excel')}
                                className="btn btn-success btn-sm"
                              >
                                <i className="fas fa-file-excel"></i>
                                Excel
                              </button>
                              <button
                                onClick={() => handleDownload(task.task_id, 'csv')}
                                className="btn btn-info btn-sm"
                              >
                                <i className="fas fa-file-csv"></i>
                                CSV
                              </button>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {task.error_message && (
                      <div className="alert alert-danger">
                        <i className="fas fa-exclamation-triangle"></i>
                        {task.error_message}
                      </div>
                    )}

                    <div className="flex gap-2">
                      <button
                        onClick={() => refreshTask(task.task_id)}
                        className="btn btn-secondary btn-sm"
                        disabled={loading}
                      >
                        <i className="fas fa-refresh"></i>
                        Обновить
                      </button>

                      {(task.status === 'processing' || task.status === 'completed') && (
                        <Link
                          to={`/results/${task.task_id}?tab=visualization`}
                          className="btn btn-info btn-sm"
                        >
                          <i className="fas fa-images"></i>
                          Просмотр
                        </Link>
                      )}

                      {task.status === 'completed' && (
                        <Link
                          to={`/results/${task.task_id}`}
                          className="btn btn-primary btn-sm"
                        >
                          <i className="fas fa-chart-bar"></i>
                          Детали
                        </Link>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Summary Stats */}
        {tasks.length > 0 && (
          <div className="row mt-4">
            <div className="col-md-12">
              <div className="card">
                <div className="card-header">
                  <h5 className="mb-0">
                    <i className="fas fa-chart-pie text-primary"></i>
                    Статистика задач
                  </h5>
                </div>
                <div className="card-body">
                  <div className="row text-center">
                    <div className="col-md-3">
                      <div className="p-3">
                        <div className="text-2xl font-bold text-primary">
                          {tasks.filter(t => t.status === 'completed').length}
                        </div>
                        <div className="text-muted">Завершено</div>
                      </div>
                    </div>
                    <div className="col-md-3">
                      <div className="p-3">
                        <div className="text-2xl font-bold text-warning">
                          {tasks.filter(t => t.status === 'processing').length}
                        </div>
                        <div className="text-muted">Обрабатывается</div>
                      </div>
                    </div>
                    <div className="col-md-3">
                      <div className="p-3">
                        <div className="text-2xl font-bold text-info">
                          {tasks.filter(t => t.status === 'pending').length}
                        </div>
                        <div className="text-muted">Ожидает</div>
                      </div>
                    </div>
                    <div className="col-md-3">
                      <div className="p-3">
                        <div className="text-2xl font-bold text-error">
                          {tasks.filter(t => t.status === 'failed').length}
                        </div>
                        <div className="text-muted">Ошибок</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
