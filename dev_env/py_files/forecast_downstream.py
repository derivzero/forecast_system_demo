def run_downstream(df, df_future, upstream_sims_dict, cfg_model):
    """
    df   = upstream df with all forecasted dependent variables
    """
    
    import pandas as pd
    from py_files.model_configs import MODEL_CONFIG as cfg_model
    from py_files.run_configs import RUN_CONFIG as cfg_run
    from py_files.charts_helpers import plot_ttm_trends
    from py_files.utils import build_ttm
           
    df = df.copy()

    start_train = cfg_run.start_train
    end_train = cfg_run.end_train
    start_fcst = cfg_run.start_fcst
    end_fcst = cfg_run.end_fcst

    # --------------------------------------------------------
    # 1a. Downstream Fit KPIs
    # --------------------------------------------------------
    df.loc[start_train:end_train, "sales_mkt_trend_fit"] = df.loc[start_train:end_train, "units_mkt_trend_fit"] * df.loc[start_train:end_train, "avg_price_trend"]
    df.loc[start_train:end_train, "sales_fit"] = df.loc[start_train:end_train, "units_fit"] * df.loc[start_train:end_train, "avg_price"]
    df.loc[start_train:end_train, "upv_fit"] = df.loc[start_train:end_train, "units_fit"] / df.loc[start_train:end_train, "visits_fit"]
    df.loc[start_train:end_train, "cogs_fit"] = df.loc[start_train:end_train, "units_fit"] * df.loc[start_train:end_train, "avg_cost_fit"]
    df.loc[start_train:end_train, "gm_fit"] = df.loc[start_train:end_train, "sales_fit"] - df.loc[start_train:end_train, "cogs_fit"]
    df.loc[start_train:end_train,"total_cost_fit"] = df.loc[start_train:end_train, "cogs_fit"] + df.loc[start_train:end_train, "fixed_cost"]
    df.loc[start_train:end_train,"net_income_fit"] = df.loc[start_train:end_train, "sales_fit"] - df.loc[start_train:end_train, "total_cost_fit"]
        
    # --------------------------------------------------------
    # 1b. Downstream Forecast KPIs
    # --------------------------------------------------------
    df.loc[start_fcst:end_fcst, "sales_mkt_trend_forecast"] = df.loc[start_fcst:end_fcst, "units_mkt_trend_forecast"] * df.loc[start_fcst:end_fcst, "avg_price_trend"]
    df.loc[start_fcst:end_fcst, "sales_forecast"] = df.loc[start_fcst:end_fcst, "units_forecast"] * df.loc[start_fcst:end_fcst, "avg_price"]
    df.loc[start_fcst:end_fcst, "upv_forecast"] = df.loc[start_fcst:end_fcst, "units_forecast"] / df.loc[start_fcst:end_fcst, "visits_forecast"]
    df.loc[start_fcst:end_fcst, "cogs_forecast"] = df.loc[start_fcst:end_fcst, "units_forecast"] * df.loc[start_fcst:end_fcst, "avg_cost_forecast"]
    df.loc[start_fcst:end_fcst, "gm_forecast"] = df.loc[start_fcst:end_fcst, "sales_forecast"] - df.loc[start_fcst:end_fcst, "cogs_forecast"]
    df.loc[start_fcst:end_fcst,"total_cost_forecast"] = df.loc[start_fcst:end_fcst, "cogs_forecast"] + df.loc[start_fcst:end_fcst, "fixed_cost"]
    df.loc[start_fcst:end_fcst,"net_income_forecast"] = df.loc[start_fcst:end_fcst, "sales_forecast"] - df.loc[start_fcst:end_fcst, "total_cost_forecast"]

    # --------------------------------------------------------
    # 1c. Insert forecast into original columns
    # --------------------------------------------------------
    df.loc[start_fcst:end_fcst, "sales_mkt_trend"] = df.loc[start_fcst:end_fcst, "units_mkt_trend_forecast"] * df.loc[start_fcst:end_fcst, "avg_price_trend"]
    df.loc[start_fcst:end_fcst, "sales"] = df.loc[start_fcst:end_fcst, "units_forecast"] * df.loc[start_fcst:end_fcst, "avg_price"]
    df.loc[start_fcst:end_fcst, "upv"] = df.loc[start_fcst:end_fcst, "units_forecast"] / df.loc[start_fcst:end_fcst, "visits_forecast"]
    df.loc[start_fcst:end_fcst, "cogs"] = df.loc[start_fcst:end_fcst, "units_forecast"] * df.loc[start_fcst:end_fcst, "avg_cost_forecast"]
    df.loc[start_fcst:end_fcst, "gm"] = df.loc[start_fcst:end_fcst, "sales_forecast"] - df.loc[start_fcst:end_fcst, "cogs_forecast"]
    df.loc[start_fcst:end_fcst,"total_cost"] = df.loc[start_fcst:end_fcst, "cogs_forecast"] + df.loc[start_fcst:end_fcst, "fixed_cost"]
    df.loc[start_fcst:end_fcst,"net_income"] = df.loc[start_fcst:end_fcst, "sales_forecast"] - df.loc[start_fcst:end_fcst, "total_cost_forecast"]

    # --------------------------------------------------------
    # 2. UNPACK UPSTREAM SIM PATHS - This is the uncertainty engine to help define forecast probabilities downstream
    # The shape is (1000, H).  1000 rows are the sims and H is the number of periods. So we have 1000 sims per period
    # --------------------------------------------------------
    units_mkt_trend_sims  = upstream_sims_dict["units_mkt_trend"]   # (1000, H)
    units_sims        = upstream_sims_dict["units"]                 # (1000, H)
    visits_sims       = upstream_sims_dict["visits"]                # (1000, H)
    avg_cost_sims     = upstream_sims_dict["avg_cost"]              # (1000, H)

    # --------------------------------------------------------
    # 3. Precompute downstream sims (once)
    # --------------------------------------------------------
    sales_mkt_trend_sims = cfg_model["sales_mkt_trend"].sim_func(units_mkt_trend_sims, df_future)
    sales_sims       = cfg_model["sales"].sim_func(units_sims, df_future)
    cogs_sims        = cfg_model["cogs"].sim_func(units_sims, avg_cost_sims)
    total_cost_sims  = cfg_model["total_cost"].sim_func(cogs_sims, df_future)
    gm_sims          = cfg_model["gm"].sim_func(sales_sims, cogs_sims)
    ni_sims          = cfg_model["net_income"].sim_func(sales_sims, total_cost_sims)
    upv_sims         = cfg_model["upv"].sim_func(units_sims, visits_sims)
    
    # --------------------------------------------------------
    # 4. Loop through derived KPIs for distributions
    # --------------------------------------------------------
    for kpi_name, cfg in cfg_model.items():

        if cfg.type != "derived":
            continue

        if kpi_name == "sales_mkt_trend":
            sim_paths = cfg.sim_func(units_mkt_trend_sims, df_future) # df_future['avg_price_trend']

        elif kpi_name == "cogs":
            sim_paths = cfg.sim_func(units_sims, avg_cost_sims)

        elif kpi_name == "total_cost":
            sim_paths = cfg.sim_func(cogs_sims, df_future) # df_future['fixed_cost']

        elif kpi_name in ("gm"):
            sim_paths = cfg.sim_func(sales_sims, cogs_sims)

        elif kpi_name in ("net_income"):
            sim_paths = cfg.sim_func(sales_sims, total_cost_sims)

        elif kpi_name == "upv":
            sim_paths = cfg.sim_func(units_sims, visits_sims)

        else:  # SALES
            sim_paths = cfg.sim_func(units_sims, df_future) # df_future['avg_price']

    # --------------------------------------------------------
    # 5. Build return dictionary AFTER loop
    # --------------------------------------------------------
    downstream_sims_dict = {
        "sales_mkt_trend" : sales_mkt_trend_sims,
        "sales"       : sales_sims,
        "cogs"        : cogs_sims,
        "total_cost"  : total_cost_sims,
        "gm"          : gm_sims,
        "net_income"  : ni_sims,
        "upv"         : upv_sims,
    }  

    # --------------------------------------------------------
    # 6. YOY CALCULATIONS FOR ALL DOWNSTREAM KPIs
    # --------------------------------------------------------

    kpis = ["cpi_fah", "oil_prices_lag7", "ppi_farm_products_lag4", "ppi_food_mfg_lag2", "ppi_food_mfg_lag3", "ppi_food_mfg_lag4", "ppi_grocery",
            "units_mkt_trend", "rdi", "home_price", "sales_mkt_trend", 
            "units", "avg_price", "avg_price_trend", "visits", "sales", "upv",
            "avg_cost", "cogs", "fixed_cost", 'total_cost', "gm", "net_income",
            "cpi_fah_fit", "units_mkt_trend_fit", "sales_mkt_trend_fit", "units_fit", "visits_fit", "sales_fit", "upv_fit", 
            "avg_cost_fit", "cogs_fit", 'total_cost_fit', "gm_fit", "net_income_fit"
            ]
    
    for kpi in kpis:
        df[f"{kpi}_yoy"] = df[kpi]/df[kpi].shift(12) - 1 

    # --------------------------------------------------------
    # 7. YOY CALCULATIONS FOR FORECAST KPIs
    # --------------------------------------------------------
    kpis_fcst = ["cpi_fah", "units_mkt_trend", "sales_mkt_trend", "units", "visits", "sales", "upv", 
            "avg_cost", "cogs", "total_cost", "gm", "net_income"]
    
    for kpi in kpis_fcst:
        df.loc[start_fcst:end_fcst, f"{kpi}_forecast_yoy"] = df[kpi]/df[kpi].shift(12) - 1 

    
    # ---------------------------------------------------------
    # 8. BUILD TTM CHARTS
    # ----------------------------------------------------------
    ttm_vars = ["units", "avg_price", "avg_cost", "sales", "cogs", "gm"]

    ttm_df = build_ttm(df, ttm_vars, cfg_run.end_train)

    ttm_charts = {}
    
    for var in ttm_vars:
        fig = plot_ttm_trends(
            ttm_df, 
            var_name=var, 
            title=f"{var.replace('_',' ').title()} — TTM",
            yaxis_title = var.replace("_"," ").title()
        )

        ttm_charts[f"{var}_ttm_comp"] = fig

    # -------------------------------------------------
    # 9. Built KPI Tree TTM
    # -------------------------------------------------
    tree_vars = ["units", "sales", "cogs", "gm"]

    end_ty = end_train
    start_ty = end_ty - pd.DateOffset(months=11)

    end_ly = end_ty - pd.DateOffset(months=12)
    start_ly = start_ty - pd.DateOffset(months=12)

    ty = df.loc[start_ty:end_ty, tree_vars]
    ly = df.loc[start_ly:end_ly, tree_vars]

    assert len(ty) == 12, f"TY has {len(ty)} rows"
    assert len(ly) == 12, f"LY has {len(ly)} rows"

    ty_sum = ty.sum()
    ly_sum = ly.sum()

    ty_sum["avg_price"] = ty_sum["sales"] / ty_sum["units"]
    ly_sum["avg_price"] = ly_sum["sales"] / ly_sum["units"]
    ty_sum["avg_cost"]  = ty_sum["cogs"]  / ty_sum["units"]
    ly_sum["avg_cost"]  = ly_sum["cogs"]  / ly_sum["units"]

    df_tree_ttm = pd.DataFrame({"ly": ly_sum, "ty": ty_sum})
    df_tree_ttm["yoy"]  = (df_tree_ttm["ty"] / df_tree_ttm["ly"] - 1) * 100
    df_tree_ttm["diff"] = df_tree_ttm["ty"] - df_tree_ttm["ly"]
    df_tree_ttm["value_type"] = "ttm"

    print("TTM Window:")
    print("TY dates:", ty.index.min(), "to", ty.index.max())
    print("LY dates:", ly.index.min(), "to", ly.index.max())

    # -------------------------------------------------
    # 10. Built KPI Tree Forecast
    # -------------------------------------------------
    tree_vars = ["units", "sales", "cogs", "gm"]  # additive only

    end_ty = end_fcst
    start_ty = end_ty - pd.DateOffset(months=5)

    end_ly = end_ty - pd.DateOffset(months=12)
    start_ly = start_ty - pd.DateOffset(months=12)

    ty = df.loc[start_ty:end_ty, tree_vars]
    ly = df.loc[start_ly:end_ly, tree_vars]

    assert len(ty) == 6, f"TY has {len(ty)} rows"
    assert len(ly) == 6, f"LY has {len(ly)} rows"
       
    ty_sum = ty[tree_vars].sum()
    ly_sum = ly[tree_vars].sum()

    # derive weighted averages
    ty_sum["avg_price"] = ty_sum["sales"] / ty_sum["units"] 
    ly_sum["avg_price"] = ly_sum["sales"] / ly_sum["units"]

    ty_sum["avg_cost"]  = ty_sum["cogs"] / ty_sum["units"]
    ly_sum["avg_cost"]  = ly_sum["cogs"] / ly_sum["units"]

    df_tree_fcst = pd.DataFrame({"ly": ly_sum, "ty": ty_sum})
    df_tree_fcst["yoy"]  = (df_tree_fcst["ty"] / df_tree_fcst["ly"] - 1) * 100
    df_tree_fcst["diff"] = df_tree_fcst["ty"] - df_tree_fcst["ly"]
    df_tree_fcst["value_type"] = "fcst"

    df_tree_ttm = df_tree_ttm.reset_index().rename(columns={"index": "var"})
    df_tree_fcst = df_tree_fcst.reset_index().rename(columns={"index": "var"})
    df_tree = pd.concat([df_tree_ttm, df_tree_fcst], axis=0)

    print("Forecast Window:")
    print("TY dates:", ty.index.min(), "to", ty.index.max())
    print("LY dates:", ly.index.min(), "to", ly.index.max())

    return df, downstream_sims_dict, ttm_charts, df_tree

     
            
