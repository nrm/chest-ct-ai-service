import { useState, useEffect, useCallback } from 'react';
import { apiService } from '../services/api';
import type { TaskStatus } from '../types/api';

export interface TaskWithResults extends TaskStatus {
  results?: any;
  error?: string;
}

export function useTasks() {
  const [tasks, setTasks] = useState<TaskWithResults[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchTasks = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const taskStatuses = await apiService.getTasks();

      // Для каждой задачи попробуем получить результаты если она завершена
      const tasksWithResults = await Promise.all(
        taskStatuses.map(async (task) => {
          if (task.status === 'completed' && task.result_files) {
            try {
              // Здесь можно добавить логику для получения результатов анализа
              // Пока просто возвращаем задачу как есть
              return task;
            } catch (err) {
              console.warn(`Failed to load results for task ${task.task_id}:`, err);
              return { ...task, error: 'Failed to load results' };
            }
          }
          return task;
        })
      );

      setTasks(tasksWithResults);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load tasks');
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshTask = useCallback(async (taskId: string) => {
    try {
      const taskStatus = await apiService.getTaskStatus(taskId);

      setTasks(prev => prev.map(task =>
        task.task_id === taskId ? taskStatus : task
      ));

      return taskStatus;
    } catch (err) {
      console.error(`Failed to refresh task ${taskId}:`, err);
      throw err;
    }
  }, []);

  const downloadResults = useCallback(async (taskId: string, format: 'excel' | 'csv') => {
    try {
      const blob = format === 'excel'
        ? await apiService.downloadExcel(taskId)
        : await apiService.downloadCSV(taskId);

      // Создаем ссылку для скачивания
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;

      // Получаем имя файла из заголовков или генерируем
      const task = tasks.find(t => t.task_id === taskId);
      const filename = task?.result_files?.[format] || `task_${taskId}_results.${format === 'excel' ? 'xlsx' : 'csv'}`;

      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      throw new Error(`Failed to download ${format.toUpperCase()}: ${err}`);
    }
  }, [tasks]);

  useEffect(() => {
    fetchTasks();

    // Автообновление каждые 5 секунд для активных задач
    const interval = setInterval(() => {
      const hasActiveTasks = tasks.some(task =>
        task.status === 'pending' || task.status === 'processing'
      );

      if (hasActiveTasks) {
        fetchTasks();
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [fetchTasks, tasks]);

  return {
    tasks,
    loading,
    error,
    fetchTasks,
    refreshTask,
    downloadResults,
  };
}
