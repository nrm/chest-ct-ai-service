// API сервис для взаимодействия с RadiAssist backend

// Определяем базовый URL API
function getApiBaseUrl(): string {
  // Проверяем глобальную переменную (из nginx)
  if (typeof window !== 'undefined' && (window as any).RADIASSIST_API_URL) {
    return (window as any).RADIASSIST_API_URL;
  }

  // Проверяем переменную окружения (для development)
  if (typeof process !== 'undefined' && process.env?.REACT_APP_API_URL) {
    return process.env.REACT_APP_API_URL;
  }

  // Default fallback - локальный backend для development
  return 'http://localhost:8000';
}

const API_BASE_URL = getApiBaseUrl();

export class ApiService {
  private baseURL: string;

  constructor(baseURL?: string) {
    this.baseURL = baseURL || API_BASE_URL;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;

    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Загружает DICOM ZIP файл для анализа
   */
  async uploadFile(file: File, processingTimeout: number = 600): Promise<import('../types/api').UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseURL}/upload?processing_timeout=${processingTimeout}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(errorData.detail || `Upload failed: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Получает статус задачи по ID
   */
  async getTaskStatus(taskId: string): Promise<import('../types/api').TaskStatus> {
    return this.request(`/tasks/${taskId}/status`);
  }

  /**
   * Скачивает Excel результат анализа
   */
  async downloadExcel(taskId: string): Promise<Blob> {
    const response = await fetch(`${this.baseURL}/tasks/${taskId}/result/excel`);

    if (!response.ok) {
      throw new Error(`Failed to download Excel: ${response.status}`);
    }

    return response.blob();
  }

  /**
   * Скачивает CSV результат анализа
   */
  async downloadCSV(taskId: string): Promise<Blob> {
    const response = await fetch(`${this.baseURL}/tasks/${taskId}/result/csv`);

    if (!response.ok) {
      throw new Error(`Failed to download CSV: ${response.status}`);
    }

    return response.blob();
  }

  /**
   * Получает список всех задач
   */
  async getTasks(): Promise<import('../types/api').TaskStatus[]> {
    const response = await this.request<{ tasks: import('../types/api').TaskStatus[] }>('/tasks');
    return response.tasks || [];
  }

  /**
   * Проверка здоровья сервиса
   */
  async healthCheck(): Promise<import('../types/api').HealthCheck> {
    return this.request('/health');
  }

  /**
   * Получает информацию о задаче и её файлах
   */
  async getTaskFiles(taskId: string): Promise<any> {
    return this.request(`/tasks/${taskId}/files`);
  }

  /**
   * Просматривает структуру данных
   */
  async browseData(): Promise<any> {
    return this.request('/data/browse');
  }

  /**
   * Получает список слайсов для задачи
   */
  async getTaskSlices(taskId: string): Promise<{task_id: string, slice_count: number, slices: Array<{index: number, filename: string}>}> {
    return this.request(`/tasks/${taskId}/slices`);
  }

  /**
   * Получает URL для изображения слайса
   */
  getSliceImageUrl(taskId: string, sliceIndex: number): string {
    return `${this.baseURL}/tasks/${taskId}/slices/${sliceIndex}`;
  }

  /**
   * Получает метаданные сегментации для задачи
   */
  async getSegmentationMetadata(taskId: string): Promise<any> {
    return this.request(`/tasks/${taskId}/segmentation/metadata`);
  }

  /**
   * Получает 3D маску для конкретного компонента
   */
  async getSegmentationMask3D(taskId: string, component: string): Promise<{shape: number[], data: number[], dtype: string}> {
    return this.request(`/tasks/${taskId}/segmentation/mask/${component}`);
  }

  /**
   * Получает URL для preview изображения сегментации
   */
  getSegmentationPreviewUrl(taskId: string): string {
    return `${this.baseURL}/tasks/${taskId}/segmentation/preview`;
  }

  /**
   * Получает список слайсов с масками для задачи
   */
  async getMaskSlices(taskId: string): Promise<{
    total_dicom_files: number;
    generated_slices: number;
    slices_dir: string;
    slices: Array<{
      slice_index: number;
      filename: string;
      path: string;
      dicom_file: string;
    }>;
  }> {
    return this.request(`/tasks/${taskId}/segmentation/slices`);
  }

  /**
   * Получает URL для изображения слайса с масками
   */
  getMaskSliceImageUrl(taskId: string, sliceFilename: string): string {
    return `${this.baseURL}/tasks/${taskId}/segmentation/slices/${sliceFilename}`;
  }

  /**
   * Запускает сегментацию костей для завершенной задачи
   */
  async runBonesSegmentation(taskId: string): Promise<{
    message: string;
    status: string;
  }> {
    return this.request(`/tasks/${taskId}/segmentation/bones`, {
      method: 'POST',
    });
  }

  /**
   * Получает список немедленных слайсов (до сегментации)
   */
  async getImmediateSlices(taskId: string): Promise<{
    total_dicom_files: number;
    generated_slices: number;
    slices_dir: string;
    slices: Array<{
      slice_index: number;
      filename: string;
      path: string;
      dicom_file: string;
      z_index?: number;
      window_center?: number;
      window_width?: number;
    }>;
  }> {
    return this.request(`/tasks/${taskId}/immediate-slices`);
  }

  /**
   * Получает URL для изображения немедленного слайса
   */
  getImmediateSliceImageUrl(taskId: string, sliceFilename: string): string {
    return `${this.baseURL}/tasks/${taskId}/immediate-slices/${sliceFilename}`;
  }
}

// Экспортируем экземпляр сервиса
export const apiService = new ApiService();
