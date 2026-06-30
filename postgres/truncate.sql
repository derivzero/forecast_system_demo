TRUNCATE raw.fred_long, 
         raw.client_long, 
         raw.forward_long;

TRUNCATE output.df_wide, 
         output.dial_values, 
         output.trees, 
         output.price_optimization_table, 
         output.chart_artifacts;
         --output.betas, 
         --output.forecast_distributions, 
         --output.holdout_results, 
         --output.mape_results;
         

TRUNCATE meta.forecast_registry, 
        meta.meta_registry;