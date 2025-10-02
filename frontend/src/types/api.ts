// Типы данных для API RadiAssist

export interface UploadResponse {
  task_id: string;
  status: string;
  message: string;
}

export interface TaskStatus {
  task_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  created_at: string;
  completed_at?: string;
  error_message?: string;
  result_files?: {
    excel: string;
    csv: string;
  };
  result_data?: {
    processing_time?: number;
    probability_of_pathology?: number;
    pathology?: number;
    most_dangerous_pathology_type?: string;
    pathology_localization?: string;
  };
}

export interface AnalysisResult {
  case: string;
  study_uid: string;
  series_uid: string;
  probability_of_pathology: number;
  pathology: number; // 0 = нормальный, 1 = патология
  processing_status: string;
  time_of_processing: number;
  most_dangerous_pathology_type?: string;
  pathology_localization?: string;
  nodule_count?: number;
  luna_confidence?: number;
  covid_probability?: number;
  ksl_score?: number;
  timestamp?: string;
}

export interface ApiError {
  detail: string;
}

export interface HealthCheck {
  status: string;
  service: string;
  version: string;
  config: {
    max_concurrent_jobs: number;
    processing_timeout: number;
    current_processing: number;
    available_slots: number;
  };
}
