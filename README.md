# Canteen Chatbot API

REST API for Canteen Chatbot application built with FastAPI.

## Features

- Chat interface with AI-powered responses
- Menu management and search
- Business analytics and insights
- Graph data for visualizations
- AI-powered recommendations

## API Endpoints

- `POST /api/chat` - Process natural language queries
- `GET /api/menu` - Get all menu items
- `GET /api/menu/{item_name}` - Get item details
- `GET /api/analytics/overview` - Business analytics
- `GET /api/graphs/*` - Graph data endpoints
- `GET /api/recommendations` - AI recommendations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
gunicorn api:app --bind 0.0.0.0:$PORT --workers 2 --worker-class uvicorn.workers.UvicornWorker --timeout 120
```

## Environment Variables

- `GEMINI_API_KEY` - Google Gemini API key (optional)
- `PORT` - Server port (default: 8000)

## Documentation

API documentation available at `/docs` when server is running.

## Deployment

Configured for deployment on Render. See `render.yaml` for configuration.
