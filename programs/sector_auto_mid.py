import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import mplfinance as mpf
import matplotlib.pyplot as plt
import os
import glob
import concurrent.futures


# ================================
# SETTINGS
# ================================

INPUT_DIR = "../2B-10B_sector_tickers"
OUTPUT_CSV_DIR = "../chart_data_mid"
OUTPUT_IMG_DIR = "../chart_pics_mid"

os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)


# ================================
# FAST + RELIABLE MARKET CAP FETCH
# ================================

def get_market_cap(ticker, adj_close):

    try:

        tk = yf.Ticker(ticker)

        # fastest method
        mc = tk.fast_info.get("marketCap")
        if mc:
            return ticker, mc

        # fallback using shares
        shares = tk.fast_info.get("sharesOutstanding")
        if shares:
            price = adj_close[ticker].iloc[-1]
            return ticker, shares * price

        # final fallback (slow but reliable)
        info = tk.info
        mc = info.get("marketCap")
        if mc:
            return ticker, mc

    except Exception:
        pass

    return ticker, None


# ================================
# PROCESS FILES
# ================================

sector_files = glob.glob(f"{INPUT_DIR}/*.csv")

print(f"Found {len(sector_files)} sector files\n")


for csv_filename in sector_files:

    print("=" * 80)
    print(f"PROCESSING: {csv_filename}")
    print("=" * 80)

    try:

        # ----------------
        # LOAD TICKERS
        # ----------------

        base_filename = os.path.splitext(
            os.path.basename(csv_filename)
        )[0]

        sector_name = base_filename

        tickers = (
            pd.read_csv(csv_filename, header=None)
            .iloc[0]
            .dropna()
            .tolist()
        )

        if len(tickers) == 1:
            tickers = tickers[0].split(",")

        tickers = [t.strip().upper() for t in tickers]

        print(f"Tickers loaded: {len(tickers)}")


        # ----------------
        # DOWNLOAD PRICES
        # ----------------

        print("Downloading price data...")

        data = yf.download(
            tickers,
            period="180d",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True
        )


        # ----------------
        # FILTER VALID
        # ----------------

        valid = []
        missing = []

        for t in tickers:

            try:
                close = data[t]["Close"].dropna()

                if len(close) > 0:
                    valid.append(t)
                else:
                    missing.append(t)

            except:
                missing.append(t)


        print(f"Valid tickers: {len(valid)}")
        print(f"Missing: {missing}")


        if len(valid) == 0:
            print("Skipping sector\n")
            continue


        adj_close = pd.concat(
            [data[t]["Close"] for t in valid],
            axis=1
        )

        adj_close.columns = valid


        # ----------------
        # MARKET CAPS
        # ----------------

        print("Fetching market caps...")

        market_caps = {}
        failed = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=10
        ) as executor:

            futures = [
                executor.submit(
                    get_market_cap,
                    t,
                    adj_close
                )
                for t in valid
            ]

            for future in concurrent.futures.as_completed(
                futures
            ):

                ticker, mc = future.result()

                if mc:
                    market_caps[ticker] = mc
                else:
                    failed.append(ticker)


        print(f"Market caps found: {len(market_caps)}")
        print(f"Failed caps: {failed}")


        if len(market_caps) == 0:
            print("Skipping sector\n")
            continue


        # ----------------
        # BUILD INDEX
        # ----------------

        adj_close = adj_close[list(market_caps.keys())]

        weights = pd.Series(market_caps)
        weights = weights / weights.sum()

        returns = adj_close.pct_change().dropna()

        weighted_returns = returns.mul(
            weights,
            axis=1
        ).sum(axis=1)

        index = (1 + weighted_returns).cumprod() * 100


        # ----------------
        # CORRELATIONS
        # ----------------

        correlations = {}

        for ticker in adj_close.columns:

            r = adj_close[ticker].pct_change()

            aligned = pd.concat(
                [r, weighted_returns],
                axis=1
            ).dropna()

            if len(aligned) > 20:

                corr = aligned.iloc[:, 0].corr(
                    aligned.iloc[:, 1]
                )

                correlations[ticker] = corr


        high_corr = [
            t for t, c in correlations.items()
            if c >= 0.50
        ]

        print(
            f"Tickers ≥50% correlation: {len(high_corr)}"
        )


        # ----------------
        # PERFORMANCE
        # ----------------

        if len(high_corr) > 0:

            latest = adj_close[high_corr].iloc[-1]
            week = adj_close[high_corr].iloc[-5]
            month = adj_close[high_corr].iloc[-21]

            perf = pd.DataFrame({

                "Week %":
                    (latest / week - 1) * 100,

                "Month %":
                    (latest / month - 1) * 100,

                "Correlation":
                    [correlations[t]
                     for t in latest.index]

            })

            print("\nTop weekly performers:")
            print(
                perf[["Week %", "Correlation"]]
                .sort_values("Week %", ascending=False)
                .head(10)
                .to_string(float_format=lambda x: f"{x:.6f}")
            )

            # ---- Monthly report ----
            print("\nTop monthly performers:")
            print(
                perf[["Month %", "Correlation"]]
                .sort_values("Month %", ascending=False)
                .head(10)
                .to_string(float_format=lambda x: f"{x:.6f}")
            )



        # ----------------
        # BUILD REALISTIC OHLC
        # ----------------

        df = pd.DataFrame(index=index.index)

        df["Close"] = index
        df["Open"] = df["Close"].shift(1)

        move = (df["Close"] - df["Open"]).abs()

        wick = move * 0.4

        df["High"] = (
            df[["Open", "Close"]]
            .max(axis=1) + wick
        )

        df["Low"] = (
            df[["Open", "Close"]]
            .min(axis=1) - wick
        )

        df.dropna(inplace=True)


        # ----------------
        # SAVE CSV
        # ----------------

        csv_out = (
            f"{OUTPUT_CSV_DIR}/"
            f"{sector_name}.data.csv"
        )

        df.to_csv(csv_out)

        print(f"Saved CSV: {csv_out}")


        # ----------------
        # SAVE CHART
        # ----------------

        last_date = df.index[-1]

        title = (
            f"{sector_name} MA:25\n"
            f"Last candle: "
            f"{last_date.strftime('%Y-%m-%d')}"
        )

        img_out = (
            f"{OUTPUT_IMG_DIR}/"
            f"{sector_name}.png"
        )

        mpf.plot(
            df,
            type="candle",
            style="charles",
            title=title,
            ylabel="Index Value",
            mav=(25,),
            volume=False,
            figsize=(12, 6),
            tight_layout=True,
            savefig=dict(
                fname=img_out,
                dpi=100,
                bbox_inches="tight"
            )
        )

        plt.close("all")

        print(f"Saved chart: {img_out}")
        print("DONE\n")


    except Exception as e:

        print(f"ERROR: {e}\n")
        continue


print("=" * 80)
print("ALL SECTORS PROCESSED")