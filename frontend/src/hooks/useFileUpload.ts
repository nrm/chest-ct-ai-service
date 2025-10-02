import { useState, useCallback } from 'react';
import { apiService } from '../services/api';
import type { UploadResponse } from '../types/api';

export interface UploadState {
  uploading: boolean;
  progress: number;
  error: string | null;
  lastUpload: UploadResponse | null;
  processingTimeout: number;
}

export function useFileUpload() {
  const [state, setState] = useState<UploadState>({
    uploading: false,
    progress: 0,
    error: null,
    lastUpload: null,
    processingTimeout: 600, // 10 минут по умолчанию
  });

  const uploadFile = useCallback(async (file: File): Promise<UploadResponse> => {
    if (!file.name.endsWith('.zip')) {
      throw new Error('Только ZIP файлы поддерживаются');
    }

    setState(prev => ({
      ...prev,
      uploading: true,
      progress: 0,
      error: null,
    }));

    try {
      // Имитируем прогресс загрузки
      const progressInterval = setInterval(() => {
        setState(prev => ({
          ...prev,
          progress: Math.min(prev.progress + 10, 90),
        }));
      }, 100);

      const response = await apiService.uploadFile(file, state.processingTimeout);

      clearInterval(progressInterval);

      setState(prev => ({
        ...prev,
        uploading: false,
        progress: 100,
        lastUpload: response,
      }));

      return response;
    } catch (error) {
      setState(prev => ({
        ...prev,
        uploading: false,
        progress: 0,
        error: error instanceof Error ? error.message : 'Ошибка загрузки',
      }));
      throw error;
    }
  }, []);

  const setProcessingTimeout = useCallback((timeout: number) => {
    setState(prev => ({
      ...prev,
      processingTimeout: timeout,
    }));
  }, []);

  const reset = useCallback(() => {
    setState({
      uploading: false,
      progress: 0,
      error: null,
      lastUpload: null,
      processingTimeout: 600,
    });
  }, []);

  return {
    ...state,
    uploadFile,
    setProcessingTimeout,
    reset,
  };
}
