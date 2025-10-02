import React from 'react';
import { Link, useLocation } from 'react-router-dom';

interface HeaderProps {
  onHealthCheck?: () => void;
}

export const Header: React.FC<HeaderProps> = ({ onHealthCheck }) => {
  const location = useLocation();

  const isActive = (path: string) => location.pathname === path;

  return (
    <header className="header">
      <div className="container">
        <div className="header-content">
          <Link to="/" className="logo">
            <i className="fas fa-heartbeat logo-icon"></i>
            <span className="logo-text">RadiAssist</span>
          </Link>

          <nav className="nav">
            <Link
              to="/"
              className={`nav-link ${isActive('/') ? 'active' : ''}`}
            >
              <i className="fas fa-home"></i>
              Главная
            </Link>

            <Link
              to="/upload"
              className={`nav-link ${isActive('/upload') ? 'active' : ''}`}
            >
              <i className="fas fa-upload"></i>
              Загрузка
            </Link>

            <Link
              to="/tasks"
              className={`nav-link ${isActive('/tasks') ? 'active' : ''}`}
            >
              <i className="fas fa-tasks"></i>
              Задачи
            </Link>

            <Link
              to="/results"
              className={`nav-link ${isActive('/results') ? 'active' : ''}`}
            >
              <i className="fas fa-chart-bar"></i>
              Результаты
            </Link>

            <button
              onClick={onHealthCheck}
              className="btn btn-secondary btn-sm"
              title="Проверить состояние сервиса"
            >
              <i className="fas fa-heart"></i>
              Health
            </button>
          </nav>
        </div>
      </div>
    </header>
  );
};
