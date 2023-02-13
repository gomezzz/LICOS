import matplotlib.pyplot as plt
import pandas as pd


def create_plots(paseos_instances):
    quantities = [
        "temperature",
        "state_of_charge",
        "current_activity",
        # "known_actors",
        "is_in_eclipse",
    ]

    # Setup data frame to collect all data
    df = pd.DataFrame(columns=("Time", "ID"))
    for idx, item in enumerate(quantities):
        names = []

        # Dataframe for this quantity
        small_df = pd.DataFrame(columns=("Time", "ID"))

        plt.figure(figsize=(8, 2), dpi=150)

        # Get dat afrom all satellites and plot it
        for instance in paseos_instances:

            # Get time of each data point
            timesteps = instance.monitor["timesteps"]

            # Get data
            values = instance.monitor[item]
            names.append(instance.local_actor.name)

            # Collect data from this sat into big dataframe
            smaller_df = pd.DataFrame(
                {
                    "Time": timesteps,
                    "ID": len(timesteps) * [instance.local_actor.name],
                    item: values,
                }
            )
            if item == "is_in_eclipse":  # pandas things...
                smaller_df["is_in_eclipse"] = smaller_df["is_in_eclipse"].astype(
                    "boolean"
                )
            small_df = pd.concat([small_df, smaller_df])

            # Plot it :)
            plt.plot(timesteps, values)
            plt.xlabel("Time [s]")
            plt.ylabel(item.replace("_", " "))
            plt.savefig("results/" + item + ".png", dpi=150)
        # Add a legend showing which satellite is which
        # plt.legend(
        #         names,
        #         fontsize = 8,
        #         bbox_to_anchor=(0.5, 1.4),
        #         ncol=10,
        #         loc="upper center",
        # )

        df = df.merge(small_df, how="right")
