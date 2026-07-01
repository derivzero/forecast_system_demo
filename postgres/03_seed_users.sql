-- ------------------------------------------------------------
-- User Access Table
-- ------------------------------------------------------------

INSERT INTO security.users_access (email, role, is_active, notes)
VALUES
  ('erickarlson@derivzero.com', 'admin', TRUE, 'Initial admin user'),
  ('viewer1@example.com', 'viewer', TRUE, 'user'),
  ('viewer2@example.com', 'viewer', TRUE, 'user'),
  ('viewer3@example.com', 'viewer', TRUE, 'user')
  ('viewer4@example.com', 'viewer', TRUE, 'user')
ON CONFLICT (email) DO NOTHING;