BEGIN;

-- Populate missing clinic_id values from the clinics table based on facility_id
UPDATE facility_patient_counts fpc
SET clinic_id = c.id
FROM clinics c
WHERE fpc.clinic_id IS NULL
  AND fpc.facility_id = c.facility_id;

-- Remove any remaining rows without a clinic_id
DELETE FROM facility_patient_counts
WHERE clinic_id IS NULL;

-- Enforce NOT NULL constraint on clinic_id
ALTER TABLE facility_patient_counts
  ALTER COLUMN clinic_id SET NOT NULL;

-- Ensure uniqueness of clinic and snapshot_date pairs
CREATE UNIQUE INDEX IF NOT EXISTS facility_patient_counts_clinic_snapshot_unique
  ON facility_patient_counts (clinic_id, snapshot_date);

COMMIT;
