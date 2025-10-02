# Frontend Dockerfile для RadiAssist

FROM node:18-alpine AS build

WORKDIR /app

# Копируем package.json и устанавливаем зависимости
COPY package.json ./
RUN npm install

# Копируем исходный код
COPY . .

# Создаем production сборку
RUN npm run build

# Production образ с nginx
FROM nginx:alpine

# Копируем nginx конфигурацию
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Копируем собранное приложение из dist
COPY --from=build /app/dist /usr/share/nginx/html

# Копируем styles.css в корень
COPY --from=build /app/styles.css /usr/share/nginx/html/styles.css

# Создаем директорию для логов
RUN mkdir -p /var/log/nginx

# Скрипт для поддержки переменных окружения во время выполнения
RUN echo '#!/bin/sh' > /docker-entrypoint.sh && \
    echo 'echo "window.RADIASSIST_API_URL = \"${REACT_APP_API_URL:-http://localhost:8000}\";" > /usr/share/nginx/html/env.js' >> /docker-entrypoint.sh && \
    echo 'nginx -g "daemon off;"' >> /docker-entrypoint.sh && \
    chmod +x /docker-entrypoint.sh

EXPOSE 80

CMD ["/docker-entrypoint.sh"]
