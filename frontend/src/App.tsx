import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Header } from './components/Header';
import { HomePage } from './pages/HomePage';
import { UploadPage } from './pages/UploadPage';
import { TasksPage } from './pages/TasksPage';
import { ResultsPage } from './pages/ResultsPage';
import { apiService } from './services/api';

function App() {
  const handleHealthCheck = async () => {
    try {
      const health = await apiService.healthCheck();
      alert(`Сервис работает: ${health.service} v${health.version}`);
    } catch (error) {
      alert(`Ошибка проверки сервиса: ${error}`);
    }
  };

  return (
    <Router>
      <div className="min-h-screen bg-secondary">
        <Header onHealthCheck={handleHealthCheck} />

        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/tasks" element={<TasksPage />} />
          <Route path="/results/:taskId" element={<ResultsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
