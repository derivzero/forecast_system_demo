-- ------------------------------------------------------------
-- User Access Table
-- ------------------------------------------------------------

INSERT INTO security.users_access (email, role, is_active, notes)
VALUES
  ('erickarlson@derivzero.com', 'admin', TRUE, 'Initial admin user'),
  ('viewer@example.com', 'viewer', TRUE, 'Example viewer - replace with real users')
ON CONFLICT (email) DO NOTHING;