import pandas as pd
import numpy as np


def get_closest_entry(df, t, id):
    df_id = df[df.ID == id]
    return df_id.iloc[(df_id["Time"] - t).abs().argsort().iloc[:1]]


def get_analysis_df(df, timestep=60, orbital_period=1):

    t = np.round(np.linspace(0, df.Time.max(), int(df.Time.max() // timestep)))
    sats = df.ID.unique()
    df["known_actors"] = pd.Categorical(df.known_actors)
    df["comm_cat"] = df.known_actors.cat.codes
    standby = []
    processing = []
    comm_stat = []

    for idx, t_cur in enumerate(t):
        n_c = 0
        comm_status = 0
        for sat in sats:
            vals = get_closest_entry(df, t_cur, sat)
            n_c += vals.current_activity.values[0] == "Standby"
            comm_status += vals.known_actors.values[0] > 0
        comm_stat.append(comm_status)
        standby.append(n_c)
        processing.append(len(sats) - n_c)

    ana_df = pd.DataFrame(
        {
            "Time[s]": t,
            "# of Standby": standby,
            "# of Processing / Comms": processing,
            "# of Sats with commswindow": comm_stat,
        }
    )
    ana_df["Completed orbits"] = (ana_df["Time[s]"]) / orbital_period
    ana_df = ana_df.round({"Completed orbits": 2})
    ana_df["Share Processing"] = ana_df["# of Processing / Comms"] / len(sats)

    return ana_df
