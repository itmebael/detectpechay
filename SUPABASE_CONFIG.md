# Supabase Configuration

## Current Configuration

Your Supabase configuration is set up in `config/database.py`:

```
SUPABASE_URL: https://zqkqmjlepigpwfykwzey.supabase.co
SUPABASE_KEY: sb_publishable_HNgog4XZVoR6FqaKuzIcGQ_7yrDAjFn
```

## Tables Configured

The following tables are configured and mapped:

1. **users** - User authentication
2. **detection_results** - Detection history
3. **dataset_images** - Dataset management
4. **petchay_dataset** - Pechay dataset with embeddings
5. **file_uploads** - File upload tracking
6. **yolo_files** - YOLO file management

## Storage Buckets

Configured buckets:
- `user-uploads` - For user uploaded images
- `dataset-images` - For dataset images
- `detection-results` - For detection result images

## Testing Connection

Run the test script to verify your connection:

```bash
python test_supabase_connection.py
```

## Next Steps

1. Run `database_schema.sql` in your Supabase SQL Editor to create all tables
2. Create the storage buckets in Supabase Dashboard
3. Test the connection using the test script
4. Start using the application!











