import matplotlib.pyplot as plt
import pandas as pd


def get_known_actor_comms_status(values):
    """Helper function to track comms status

    Args:
        values (list of str): Names of known actors.

    Returns:
        list of int: Indices of known actors.
    """

    vals = []
    for item in values:
        if "Svalbard" in item:
            vals.append(1)
        elif "Matera" in item:
            vals.append(2)
        elif "Maspalomas" in item:
            vals.append(3)
        else:
            vals.append(0)
    return vals


def create_plots(paseos_instances, rank, figsize=(8, 2), dpi=150):
    """Creates plots from the instance

    Args:
        paseos_instances (PASEOS): instances to plot
        rank (int): Index of MPI rank.
        figsize (tuple): Size of plots.
        dpi (int): dpi of the plots.
    """
    quantities = [
        "temperature",
        "state_of_charge",
        "current_activity",
        "known_actors",
        "is_in_eclipse",
    ]

    # Setup data frame to collect all data
    df = pd.DataFrame(columns=("Time", "ID"))
    for idx, item in enumerate(quantities):
        names = []

        # Dataframe for this quantity
        small_df = pd.DataFrame(columns=("Time", "ID"))

        plt.figure(figsize=figsize, dpi=dpi)

        # Get dat afrom all satellites and plot it
        for instance in paseos_instances:

            # Get time of each data point
            timesteps = instance.monitor["timesteps"]

            # Get data
            if item == "known_actors":
                values = get_known_actor_comms_status(instance.monitor[item])
            else:
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
            if item == "known_actors":
                plt.yticks([0, 1, 2, 3], ["None", "Svalbard", "Matera", "Maspalomas"])
            plt.savefig("results/" + item + "_rank" + str(rank) + ".png", dpi=dpi)
        # Add a legend showing which satellite is which
        # plt.legend(
        #         names,
        #         fontsize = 8,
        #         bbox_to_anchor=(0.5, 1.4),
        #         ncol=10,
        #         loc="upper center",
        # )

        df = df.merge(small_df, how="right")
