# Database Migrations

Run these migrations against the configured PostgreSQL database in order:

```bash
psql -U postgres -d foresightx_db -f src/migrations/001_create_core_schema.sql
psql -U postgres -d foresightx_db -f src/migrations/002_create_core_indexes.sql
```

The schema includes:

- `users`
- `user_risk_groups`
- `advisory_rules`
- `advisories_sent`
- `notifications`
- `execution_log`

No business logic is implemented in these migrations.
