# Pechay Detection System

A Flask web application for detecting pechay leaf diseases using Supabase as the backend database.

## Setup

1. Install dependencies:
```bash
pip install flask supabase python-dotenv
```

2. Configure Supabase:
   - Supabase URL: https://zqkqmjlepigpwfykwzey.supabase.co
   - Supabase Key: Configured in app.py

## Running the Application

```bash
python app.py
```

The application will be available at: http://localhost:5000

## Database Connection

The application is configured to connect to Supabase. Make sure your Supabase project has the following tables:
- Users table (managed by Supabase Auth)
- Detections table (for storing leaf detection results)
- Dataset table (for storing training images)

## Note on Authentication

For server-side authentication operations, you may need a Service Role Key instead of a Publishable Key. The current implementation uses the provided publishable key. If you encounter authentication issues, consider using a Service Role Key for backend operations.











