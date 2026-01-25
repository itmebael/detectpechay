# Database Setup Instructions for Supabase

## Step 1: Run the SQL Schema

1. Open your Supabase Dashboard: https://supabase.com/dashboard
2. Select your project: `zqkqmjlepigpwfykwzey`
3. Go to **SQL Editor**
4. Click **New Query**
5. Copy and paste the entire contents of `database_schema.sql`
6. Click **Run** (or press Ctrl+Enter)

## Step 2: Verify Tables

After running the SQL, verify that all tables were created:

```sql
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;
```

You should see:
- `users`
- `file_uploads`
- `detection_results`
- `petchay_dataset`
- `dataset_images`
- `yolo_files`
- `embeddings`

## Step 3: Create Storage Buckets

1. Go to **Storage** in Supabase Dashboard
2. Create these buckets:

### Bucket: `user-uploads`
- **Public**: Yes
- **File size limit**: 10MB
- **Allowed MIME types**: image/*

### Bucket: `dataset-images`
- **Public**: Yes
- **File size limit**: 10MB
- **Allowed MIME types**: image/*

### Bucket: `detection-results`
- **Public**: Yes
- **File size limit**: 10MB
- **Allowed MIME types**: image/*

## Step 4: Enable pgvector Extension (if not already enabled)

If you haven't already, enable the pgvector extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## Step 5: Test Database Connection

You can test if everything is set up correctly by running:

```sql
-- Check users table
SELECT COUNT(*) FROM users;

-- Check detection_results table structure
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'detection_results';

-- Test insert (optional)
INSERT INTO users (username, email, password) 
VALUES ('test_user', 'test@example.com', 'hashed_password')
ON CONFLICT DO NOTHING;
```

## Notes

- **Password Storage**: The current implementation uses SHA256 hashing. For production, consider using bcrypt.
- **RLS (Row Level Security)**: Currently disabled. Enable and add policies if you need user-level data isolation.
- **Embeddings**: The `vector(512)` type requires the pgvector extension. Make sure it's enabled.
- **Foreign Keys**: All foreign keys are set to `ON DELETE SET NULL` to preserve data if a user is deleted.

## Troubleshooting

### If you get "extension vector does not exist":
```sql
-- Install pgvector extension
CREATE EXTENSION vector;
```

### If you get "function uuid_generate_v4() does not exist":
```sql
-- Enable uuid-ossp extension
CREATE EXTENSION "uuid-ossp";
```

### If tables already exist:
You can drop and recreate them (WARNING: This will delete all data):
```sql
DROP TABLE IF EXISTS detection_results CASCADE;
DROP TABLE IF EXISTS dataset_images CASCADE;
DROP TABLE IF EXISTS petchay_dataset CASCADE;
DROP TABLE IF EXISTS file_uploads CASCADE;
DROP TABLE IF EXISTS yolo_files CASCADE;
DROP TABLE IF EXISTS embeddings CASCADE;
DROP TABLE IF EXISTS users CASCADE;
```
Then re-run the schema SQL.



















