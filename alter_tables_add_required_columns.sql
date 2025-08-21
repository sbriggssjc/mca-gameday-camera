BEGIN;

-- Add required columns for medicare_clinics
ALTER TABLE IF EXISTS medicare_clinics
  ADD COLUMN IF NOT EXISTS facility_name TEXT,
  ADD COLUMN IF NOT EXISTS medicare_id TEXT;

-- Add required columns for facility_patient_counts
ALTER TABLE IF EXISTS facility_patient_counts
  ADD COLUMN IF NOT EXISTS medicare_id TEXT,
  ADD COLUMN IF NOT EXISTS snapshot_date DATE,
  ADD COLUMN IF NOT EXISTS total_patients INTEGER;

-- Add required columns for ownership_history
ALTER TABLE IF EXISTS ownership_history
  ADD COLUMN IF NOT EXISTS end_date DATE,
  ADD COLUMN IF NOT EXISTS property_id TEXT,
  ADD COLUMN IF NOT EXISTS start_date DATE;

-- Add required columns for scrub_cache
ALTER TABLE IF EXISTS scrub_cache
  ADD COLUMN IF NOT EXISTS file_name TEXT,
  ADD COLUMN IF NOT EXISTS gpt_output TEXT,
  ADD COLUMN IF NOT EXISTS raw_text TEXT;

COMMIT;
