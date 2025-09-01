SELECT
    (SELECT count(*) FROM leases WHERE rent IS NOT NULL) AS rent,
    (SELECT count(*) FROM leases WHERE rent_per_sf IS NOT NULL AND "sqft" IS NOT NULL) AS rent_psf;
