import paseos
import pykep as pk


def decide_on_activity(
    paseos_instance: paseos.PASEOS,
    timestep,
    timestep_for_comms,
    time_in_standby,
    standby_period,
    time_since_last_update,
):
    """Heuristic to decide activitiy for . Initiates a standby period of passed
    length when going into standby.

    Args:
        paseos_instance (paseos): Local instance
        timestep (float): Timestep for standing by
        timestep_for_comms (float): Time we need to send model
        time_in_standby (float): Time we have been in standby
        standby_period (float): Time a standby period should last
        time_since_last_update (float): Time since we exchanged model last

    Returns:
        tuple: activity name, power consumption, time spent in standby
    """
    has_comm_window = False
    window_end = pk.epoch(
        paseos_instance.local_actor.local_time.mjd2000 + timestep_for_comms * pk.SEC2DAY
    )
    for actors in paseos_instance.known_actors.items():
        if paseos_instance.local_actor.is_in_line_of_sight(
            actors[1], epoch=paseos_instance.local_actor.local_time
        ) and paseos_instance.local_actor.is_in_line_of_sight(
            actors[1], epoch=window_end
        ):
            has_comm_window = True
            break
    if (
        has_comm_window
        and time_since_last_update > 900
        and len(paseos_instance.known_actors) > 0
        and paseos_instance.local_actor.state_of_charge > 0.1
        and paseos_instance.local_actor.temperature_in_K < 273.15 + 45
    ):
        return "Model_update", 10, 0
    elif (
        paseos_instance.local_actor.temperature_in_K > (273.15 + 40)
        or paseos_instance.local_actor.state_of_charge < 0.2
        or (time_in_standby > 0 and time_in_standby < standby_period)
    ):
        return "Standby", 5, time_in_standby + timestep
    else:
        # Wattage from 1605B https://www.amd.com/en/products/embedded-ryzen-v1000-series
        # https://unibap.com/wp-content/uploads/2021/06/spacecloud-ix5-100-product-overview_v23.pdf
        return "Training", 30, 0


def constraint_func(paseos_instance: paseos.PASEOS, actors_to_track):
    """Constraint function for activitiy

    Args:
        paseos_instance (paseos): Local instance
        actors_to_track (Actor): Actors we want to communicate with

    Returns:
        Whether constraint is still met
    """

    # Update known actors
    local_t = paseos_instance.local_actor.local_time
    paseos_instance.empty_known_actors()
    for actor in actors_to_track:
        if paseos_instance.local_actor.is_in_line_of_sight(actor, local_t):
            paseos_instance.add_known_actor(actor)

    # Check constraints
    if paseos_instance.local_actor.temperature_in_K > (273.15 + 65):
        return False
    if paseos_instance.local_actor.state_of_charge < 0.1:
        return False

    return True


def perform_activity(
    activity, power_consumption, paseos_instance, time_to_run, constraint_function
):
    """Advance time in paseos according to activity.

    Args:
        activity (str): Name of activity
        power_consumption (float): power consumed per s
        paseos_instance (paseos): instance of paseos to use
        time_to_run (float): time in seconds this is run for
        constraint_function (func): constraint function fo the activity
    """
    paseos_instance.local_actor._current_activity = activity
    return_code = paseos_instance.advance_time(
        time_to_run,
        current_power_consumption_in_W=power_consumption,
        constraint_function=constraint_function,
    )
    if return_code > 0:
        raise ("Activity was interrupted. Constraints no longer true?")
