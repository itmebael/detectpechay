-- =====================================================
-- Pechay Detection System - Supabase Database Schema
-- =====================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";  -- For pgvector embeddings

-- =====================================================
-- 1. USERS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS public.users (
  id UUID NOT NULL DEFAULT uuid_generate_v4(),
  username CHARACTER VARYING(100) NOT NULL,
  email CHARACTER VARYING(255) NOT NULL,
  password CHARACTER VARYING(255) NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE NULL DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE NULL DEFAULT NOW(),
  CONSTRAINT users_pkey PRIMARY KEY (id),
  CONSTRAINT users_email_key UNIQUE (email),
  CONSTRAINT users_username_key UNIQUE (username)
) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_users_username ON public.users USING btree (username) TABLESPACE pg_default;
CREATE INDEX IF NOT EXISTS idx_users_email ON public.users USING btree (email) TABLESPACE pg_default;

-- =====================================================
-- 2. FILE_UPLOADS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS public.file_uploads (
  id UUID NOT NULL DEFAULT uuid_generate_v4(),
  filename TEXT NOT NULL,
  original_name TEXT NULL,
  file_path TEXT NOT NULL,
  file_size BIGINT NULL,
  upload_source CHARACTER VARYING(50) NULL,
  user_id UUID NULL,
  created_at TIMESTAMP WITH TIME ZONE NULL DEFAULT NOW(),
  CONSTRAINT file_uploads_pkey PRIMARY KEY (id),
  CONSTRAINT file_uploads_user_id_fkey FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_file_uploads_user_id ON public.file_uploads USING btree (user_id) TABLESPACE pg_default;
CREATE INDEX IF NOT EXISTS idx_file_uploads_created_at ON public.file_uploads USING btree (created_at DESC) TABLESPACE pg_default;

-- =====================================================
-- 3. DETECTION_RESULTS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS public.detection_results (
  id UUID NOT NULL DEFAULT uuid_generate_v4(),
  filename CHARACTER VARYING(255) NOT NULL,
  condition CHARACTER VARYING(50) NOT NULL,
  confidence NUMERIC(5, 2) NOT NULL,
  timestamp TIMESTAMP WITH TIME ZONE NULL DEFAULT NOW(),
  image_path TEXT NOT NULL,
  user_id UUID NULL,
  all_probabilities JSONB NULL,
  recommendations JSONB NULL,
  disease_name CHARACTER VARYING(100) NULL,
  detection_method CHARACTER VARYING(50) NULL,
  created_at TIMESTAMP WITH TIME ZONE NULL DEFAULT NOW(),
  CONSTRAINT detection_results_pkey PRIMARY KEY (id),
  CONSTRAINT detection_results_filename_timestamp_key UNIQUE (filename, timestamp),
  CONSTRAINT detection_results_user_id_fkey FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL,
  CONSTRAINT detection_results_condition_check CHECK (
    (condition)::text = ANY (
      (ARRAY['Healthy'::character varying, 'Diseased'::character varying])::text[]
    )
  )
) TABLESPACE pg_default;

ALTER TABLE public.detection_results
  ADD COLUMN IF NOT EXISTS disease_name CHARACTER VARYING(100),
  ADD COLUMN IF NOT EXISTS detection_method CHARACTER VARYING(50);

CREATE INDEX IF NOT EXISTS idx_detection_results_filename ON public.detection_results USING btree (filename) TABLESPACE pg_default;
CREATE INDEX IF NOT EXISTS idx_detection_results_condition ON public.detection_results USING btree (condition) TABLESPACE pg_default;
CREATE INDEX IF NOT EXISTS idx_detection_results_user_id ON public.detection_results USING btree (user_id) TABLESPACE pg_default;
CREATE INDEX IF NOT EXISTS idx_detection_results_timestamp ON public.detection_results USING btree (timestamp DESC) TABLESPACE pg_default;
CREATE INDEX IF NOT EXISTS idx_detection_results_created_at ON public.detection_results USING btree (created_at DESC) TABLESPACE pg_default;

-- =====================================================
-- 4. PETCHAY_DATASET TABLE (with embeddings)
-- =====================================================
CREATE TABLE IF NOT EXISTS public.petchay_dataset (
  id UUID NOT NULL DEFAULT gen_random_uuid(),
  filename TEXT NOT NULL,
  condition TEXT NOT NULL,
  disease_name TEXT NULL,
  image_url TEXT NOT NULL,
  embedding vector(512) NULL,
  user_id UUID NULL,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT timezone('utc'::text, NOW()),
  updated_at TIMESTAMP WITH TIME ZONE NULL DEFAULT NOW(),
  label TEXT NULL,
  label_confidence DOUBLE PRECISION NULL,
  image_region TEXT NULL,
  bounding_box JSONB NULL,
  annotation_notes TEXT NULL,
  quality_score DOUBLE PRECISION NULL,
  is_verified BOOLEAN NULL DEFAULT FALSE,
  verified_by UUID NULL,
  verified_at TIMESTAMP WITH TIME ZONE NULL,
  CONSTRAINT petchay_dataset_pkey PRIMARY KEY (id),
  CONSTRAINT petchay_dataset_user_id_fkey FOREIGN KEY (user_id) REFERENCES users (id)
) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_petchay_dataset_filename ON public.petchay_dataset USING btree (filename) TABLESPACE pg_default;
CREATE INDEX IF NOT EXISTS idx_petchay_dataset_condition ON public.petchay_dataset USING btree (condition) TABLESPACE pg_default;
CREATE INDEX IF NOT EXISTS idx_petchay_dataset_disease_name ON public.petchay_dataset USING btree (disease_name) TABLESPACE pg_default;
CREATE INDEX IF NOT EXISTS idx_petchay_dataset_user_id ON public.petchay_dataset USING btree (user_id) TABLESPACE pg_default;
CREATE INDEX IF NOT EXISTS idx_petchay_dataset_created_at ON public.petchay_dataset USING btree (created_at DESC) TABLESPACE pg_default;
CREATE INDEX IF NOT EXISTS idx_petchay_dataset_label ON public.petchay_dataset USING btree (label) TABLESPACE pg_default;
CREATE INDEX IF NOT EXISTS idx_petchay_dataset_image_region ON public.petchay_dataset USING btree (image_region) TABLESPACE pg_default;
CREATE INDEX IF NOT EXISTS idx_petchay_dataset_is_verified ON public.petchay_dataset USING btree (is_verified) TABLESPACE pg_default;
CREATE INDEX IF NOT EXISTS idx_petchay_dataset_quality_score ON public.petchay_dataset USING btree (quality_score DESC) TABLESPACE pg_default;

-- Vector similarity search index for embeddings (pgvector)
CREATE INDEX IF NOT EXISTS idx_petchay_dataset_embedding ON public.petchay_dataset 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100) TABLESPACE pg_default;

-- =====================================================
-- 5. DATASET_IMAGES TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS public.dataset_images (
  id UUID NOT NULL DEFAULT uuid_generate_v4(),
  filename CHARACTER VARYING(255) NOT NULL,
  label CHARACTER VARYING(50) NOT NULL,
  image_path TEXT NOT NULL,
  split CHARACTER VARYING(20) NULL DEFAULT 'train'::character varying,
  metadata JSONB NULL DEFAULT '{}'::jsonb,
  user_id UUID NULL,
  timestamp TIMESTAMP WITH TIME ZONE NULL DEFAULT NOW(),
  created_at TIMESTAMP WITH TIME ZONE NULL DEFAULT NOW(),
  CONSTRAINT dataset_images_pkey PRIMARY KEY (id),
  CONSTRAINT dataset_images_user_id_fkey FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_dataset_images_label ON public.dataset_images USING btree (label) TABLESPACE pg_default;
CREATE INDEX IF NOT EXISTS idx_dataset_images_split ON public.dataset_images USING btree (split) TABLESPACE pg_default;
CREATE INDEX IF NOT EXISTS idx_dataset_images_user_id ON public.dataset_images USING btree (user_id) TABLESPACE pg_default;
CREATE INDEX IF NOT EXISTS idx_dataset_images_created_at ON public.dataset_images USING btree (created_at DESC) TABLESPACE pg_default;

-- =====================================================
-- 6. YOLO_FILES TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS public.yolo_files (
  id BIGINT GENERATED ALWAYS AS IDENTITY NOT NULL,
  filename TEXT NOT NULL,
  file_type TEXT NULL DEFAULT 'image'::text,
  dataset_type TEXT NULL,
  url TEXT NULL,
  treatment TEXT NULL,
  uploaded_at TIMESTAMP WITHOUT TIME ZONE NULL DEFAULT NOW(),
  label TEXT NULL,
  label_confidence DOUBLE PRECISION NULL,
  image_region TEXT NULL,
  bounding_box JSONB NULL,
  annotation_notes TEXT NULL,
  quality_score DOUBLE PRECISION NULL,
  is_verified BOOLEAN NULL DEFAULT FALSE,
  verified_by UUID NULL,
  verified_at TIMESTAMP WITH TIME ZONE NULL,
  CONSTRAINT yolo_files_pkey PRIMARY KEY (id)
) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_yolo_files_filename ON public.yolo_files USING btree (filename) TABLESPACE pg_default;
CREATE INDEX IF NOT EXISTS idx_yolo_files_dataset_type ON public.yolo_files USING btree (dataset_type) TABLESPACE pg_default;
CREATE INDEX IF NOT EXISTS idx_yolo_files_uploaded_at ON public.yolo_files USING btree (uploaded_at DESC) TABLESPACE pg_default;
CREATE INDEX IF NOT EXISTS idx_yolo_files_label ON public.yolo_files USING btree (label) TABLESPACE pg_default;
CREATE INDEX IF NOT EXISTS idx_yolo_files_image_region ON public.yolo_files USING btree (image_region) TABLESPACE pg_default;
CREATE INDEX IF NOT EXISTS idx_yolo_files_is_verified ON public.yolo_files USING btree (is_verified) TABLESPACE pg_default;
CREATE INDEX IF NOT EXISTS idx_yolo_files_quality_score ON public.yolo_files USING btree (quality_score DESC) TABLESPACE pg_default;

-- =====================================================
-- 7. EMBEDDINGS TABLE (Optional - for similarity search)
-- =====================================================
CREATE TABLE IF NOT EXISTS public.embeddings (
  id UUID NOT NULL DEFAULT uuid_generate_v4(),
  image_id UUID NULL,
  embedding vector(512) NULL,
  created_at TIMESTAMP WITH TIME ZONE NULL DEFAULT NOW(),
  CONSTRAINT embeddings_pkey PRIMARY KEY (id)
) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS idx_embeddings_image_id ON public.embeddings USING btree (image_id) TABLESPACE pg_default;
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON public.embeddings 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100) TABLESPACE pg_default;

-- =====================================================
-- TRIGGER FUNCTION: Update updated_at column
-- =====================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- TRIGGERS: Auto-update updated_at
-- =====================================================
DROP TRIGGER IF EXISTS update_users_updated_at ON public.users;
CREATE TRIGGER update_users_updated_at
BEFORE UPDATE ON public.users
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_petchay_dataset_updated_at ON public.petchay_dataset;
CREATE TRIGGER update_petchay_dataset_updated_at
BEFORE UPDATE ON public.petchay_dataset
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- ROW LEVEL SECURITY (RLS) - Optional
-- Enable if you want to add security policies
-- =====================================================

-- Enable RLS on tables (optional - uncomment if needed)
-- ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE public.detection_results ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE public.dataset_images ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE public.petchay_dataset ENABLE ROW LEVEL SECURITY;

-- Example RLS Policy (users can only see their own data)
-- CREATE POLICY "Users can view own detections" ON public.detection_results
--   FOR SELECT USING (auth.uid() = user_id);

-- =====================================================
-- GRANT PERMISSIONS (if using service role)
-- =====================================================

-- Grant permissions to authenticated users (adjust as needed)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO authenticated;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO authenticated;

-- =====================================================
-- COMPLETION MESSAGE
-- =====================================================
DO $$
BEGIN
    RAISE NOTICE 'Database schema created successfully!';
    RAISE NOTICE 'Tables created: users, file_uploads, detection_results, petchay_dataset, dataset_images, yolo_files, embeddings';
END $$;










