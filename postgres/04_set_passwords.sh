#!/bin/bash
psql -U postgres -c "ALTER USER derivzero WITH PASSWORD '$PG_PASSWORD';"
psql -U postgres -c "ALTER USER forecast_app_user WITH PASSWORD '$PG_APP_PASSWORD';"