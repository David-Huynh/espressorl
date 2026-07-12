-- Add algorithm-neutral pairwise comparison records to the signed raw queue.
-- These records can supervise online preference optimizers and offline
-- preference-supervised offline models without embedding an optimizer-specific schema.

ALTER TABLE public.raw_upload_queue
    DROP CONSTRAINT IF EXISTS raw_upload_queue_local_record_type_check;

ALTER TABLE public.raw_upload_queue
    ADD CONSTRAINT raw_upload_queue_local_record_type_check
    CHECK (local_record_type IN ('shot', 'recommendation', 'comparison'));

ALTER TABLE public.raw_upload_queue
    DROP CONSTRAINT IF EXISTS raw_upload_queue_event_type_check;

ALTER TABLE public.raw_upload_queue
    ADD CONSTRAINT raw_upload_queue_event_type_check
    CHECK (event_type IN ('shot_record', 'recommendation_record', 'comparison_record'));
