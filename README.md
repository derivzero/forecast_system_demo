# U.S. Grocery Market Navigator

A six-month forecasting system for U.S. grocery retail. It forecasts market-level
demand and prices, translates those into a single retailer's units, sales, costs and
margins, and presents the result as a Streamlit dashboard with break-even pricing and
scenario planning.

The whole thing runs locally in Docker. Clone it, start three containers, run one
script, open a browser.

> **The retailer data is synthetic.** Market series (CPI, PPI, oil, real disposable
> income, home prices, grocery sales) are real FRED data. The retailer — its units,
> visits, prices, costs and margins — is generated from those series by
> `py_files/client_dataset.py`. This is a demonstration of the modelling and
> deployment approach, not an analysis of a real company's results.

---

## Quick start

You need [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed
and running. Nothing else — no Python, no database, no API keys.

**1. Clone and enter the project**

```
git clone https://github.com/derivzero/forecast_system_demo.git
cd forecast_system_demo
```

**2. Create your `.env` file**

Copy `.env.example` to `.env` and replace the three `changeme` values with any
passwords you like. This is a local database on your own machine, so they only need to
match themselves.

```
copy .env.example .env      (Windows)
cp .env.example .env        (Mac/Linux)
```

Leave `PG_HOST=postgres` exactly as it is. That is the name of the database container,
not a machine name — the containers find each other by service name.

**3. Start the three containers**

```
docker compose up -d
```

`-d` runs them in the background. The first run downloads and builds images, which
takes a few minutes. Afterwards it is a few seconds.

Confirm all three are running:

```
docker compose ps
```

You should see `postgres`, `dev_env` and `streamlit`, each `Up`.

**4. Run the forecast**

```
docker compose exec dev_env python master.py
```

This runs the full pipeline — five Kalman filter models, rolling validation, 1,000
bootstrap simulations per model, downstream KPIs, price optimization and around a
hundred charts. Expect it to take several minutes.

At the end it asks:

```
Type True to publish:
```

Type **`True`** — capital T, exactly. Anything else skips the database write, and the
dashboard will come up empty with no error explaining why.

**5. Open the dashboard**

```
http://localhost:8501
```

Streamlit is already running as one of the three containers. There is no command to
start it.

---

## Running it again

The publish step refuses to overwrite an existing run:

```
Publish aborted: run_id already exists (2026_01.0)
```

This is deliberate — it protects a published forecast from being silently replaced.
To re-run, clear the tables first:

```
docker compose exec -T postgres psql -U postgres -d postgres < postgres/truncate.sql
```

That empties the data tables but leaves the schema, roles and permissions intact.

## Stopping and starting

```
docker compose stop     stop all three, keep everything
docker compose start    start them again
docker compose logs -f streamlit    watch the dashboard's output
```

Your forecast data lives in a Docker volume and survives stopping, starting, and
rebooting.

---

## What you are looking at

The sidebar has six sections.

**Market** — the U.S. grocery market as a whole. Trailing-twelve-month charts, then
forecasts for CPI Food-at-Home, market units and market sales.

**Retailer History (TTM)** — where the retailer stands today. A KPI snapshot, trend
charts against the prior year, and a KPI tree decomposing sales into units and price,
and gross margin into sales and cost of goods.

**Retailer Demand Forecasts** — six-month forecasts for units, sales, visits, and units
per visit.

**Retailer Cost Forecasts** — average cost, cost of goods sold, and total cost.

**Retailer Margin Forecasts** — gross margin and net income, plus a forecast KPI tree
and market share trends.

**Optimization** — the part that turns a forecast into a decision. A price grid showing
how forecast units respond across a range of average prices, the two break-even prices
implied by holding sales or units flat against last year, and a scenario planner where
you set average price, average cost and fixed cost and see the six-month financial
outcome.

Each forecast page follows the same four-part layout:

- **Driver importance** — how much each input is contributing, and how that has shifted
  over time
- **Forecast: 6-month outlook** — the forecast in levels and year-over-year
- **Forecast distributions** — the range of outcomes from the bootstrap simulation, not
  just the central path
- **Holdout analysis** — how the model performed on data it had not seen

---

## How the forecast works

### Time-varying parameter regression

Each of the five statistical models is a regression whose coefficients are allowed to
drift over time, estimated with a Kalman filter (`py_files/forecast_helpers.py`,
`tvp_reg_kf`). The coefficients follow a random walk: they stay where they are unless
the data pushes them.

This matters for grocery. A fixed-coefficient regression assumes the relationship
between, say, producer prices and shelf prices is the same in 2015 and 2022. It is not.
Letting the coefficients move lets the model track a changing pass-through relationship
instead of averaging across two different regimes.

Two tuning parameters control the behaviour, set per model in `py_files/model_configs.py`:

- **Q** — how freely coefficients drift. Larger means faster adaptation, noisier estimates.
- **R** — how much the observations are trusted. Larger means more smoothing.

### The five upstream models

Estimated in order, because later ones consume earlier ones:

| model | what it forecasts | main drivers |
|---|---|---|
| `cpi_fah` | CPI Food-at-Home | oil, farm products PPI, food manufacturing PPI, grocery PPI (lagged) |
| `units_mkt_trend` | market unit volume | market price, real disposable income, home prices |
| `units` | retailer units | retailer price, real disposable income, home prices, seasonality |
| `visits` | retailer visits | price, units, seasonality |
| `avg_cost` | retailer average unit cost | food manufacturing PPI (lagged) |

All are estimated in logs and include a lagged dependent variable, so the forward
forecast is recursive — each month's prediction feeds the next month's lag
(`recursive_kf_forecast`).

### Uncertainty

The fan charts are not analytic confidence intervals. The model is refit across a
series of rolling six-month windows stepped through a twelve-month holdout period, and
the percentage errors are collected by horizon — all the one-month-ahead errors, all
the two-month-ahead errors, and so on. Those errors are then resampled a thousand times
onto the forward forecast (`bootstrap_simulate`).

The result is an empirical distribution built from how this model has actually
performed, rather than from an assumption about how the residuals are distributed.

### Derived KPIs

Everything below the five statistical models is arithmetic, not a separate regression
(`py_files/forecast_downstream.py`):

```
sales       = units x average price
cogs        = units x average cost
gross margin = sales - cogs
total cost  = cogs + fixed cost
net income  = sales - total cost
units/visit = units / visits
```

The simulated paths flow through the same arithmetic, so the uncertainty in net income
is inherited from the uncertainty in units, cost and visits rather than estimated
separately.

### Price optimization

The units model's price coefficient is a demand elasticity. Holding everything else at
its forecast value and varying price across a grid gives a demand curve
(`py_files/optimize_helpers.py`, `create_price_grid_df`). From that the dashboard
derives two break-even prices — the price at which forecast sales match last year's,
and the price at which forecast units match last year's — which together bracket a
defensible pricing range.

---

## Configuration

Almost everything is driven by one line in `py_files/run_configs.py`:

```python
forecast_month: str = "2026-01-01"
```

Training runs from `min_date` to the month before this, and the forecast covers this
month plus five. Changing it requires a matching
`data/shared/forward_values_<forecast_month>.csv`, which holds the assumed
year-over-year paths for the forward-looking inputs.

`run_id` in the same file labels the published run and is what the duplicate guard
checks.

## Refreshing FRED data

The repo ships with the FRED extract it needs, so no API key is required. To pull fresh
data instead, get a free key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html),
add it to `.env` as `FRED_API_KEY`, and uncomment the block at the top of
`build_artifacts()` in `master.py`.

---

## Project structure

```
docker-compose.yml        three services: postgres, dev_env, streamlit
.env.example              template for your local .env

postgres/                 database schema, run once on first start
  01_init.sql             schemas, roles, tables
  02_init_views.sql       views the dashboard reads
  03_seed_users.sql       access table (not enforced in this demo)
  truncate.sql            clear data, keep structure

dev_env/
  master.py               entry point - runs the pipeline and publishes
  app/app_pg.py           the Streamlit dashboard
  py_files/
    client_dataset.py     generates the synthetic retailer
    dataset.py            lags, logs, dummies, forward values
    forecast_core.py      per-model orchestration
    forecast_helpers.py   the Kalman filter and bootstrap
    forecast_downstream.py derived KPIs
    optimize_core.py      price grid and break-even
    charts_core.py        chart generation
    model_configs.py      model specifications and Q/R tuning
    run_configs.py        dates, run_id, publish flags
  data/shared/            FRED extract and forward assumptions
```

Data flows in one direction: `master.py` writes to Postgres, the dashboard reads from
it. The dashboard connects as a read-only role and cannot modify anything.

---

## Notes

**No authentication.** This demo has no login. The deployed version gates access behind
Google OAuth and a user table; that has been removed here so the dashboard runs
straight out of the box.

**Port 8501 is open to your network.** `docker-compose.yml` publishes it on all
interfaces, so anyone on your local network can reach the dashboard. To restrict it to
your own machine, change the port line under `streamlit` to `"127.0.0.1:8501:8501"`.
