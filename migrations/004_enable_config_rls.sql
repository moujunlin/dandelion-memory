alter table aml.config enable row level security;
revoke all on aml.config from public, anon, authenticated;
