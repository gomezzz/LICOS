import pykep as pk
import paseos
from paseos import ActorBuilder, SpacecraftActor, GroundstationActor

from get_constellation import get_constellation


def init_paseos(rank):
    # PASEOS setup
    altitude = 786 * 1000  # altitude above the Earth's ground [m]
    inclination = 98.62  # inclination of the orbit
    nPlanes = 1  # the number of orbital planes
    nSats = 1  # the number of satellites per orbital plane
    t0 = pk.epoch_from_string("2023-Dec-17 14:42:42")  # starting date of our simulation

    # Compute the orbit of each rank
    planet_list, sats_pos_and_v, _ = get_constellation(
        altitude, inclination, nSats, nPlanes, t0, verbose=False
    )
    print(
        f"Rank {rank} set up its orbit with altitude={altitude}m and inclination={inclination}deg"
    )

    earth = pk.planet.jpl_lp("earth")  # define our central body
    pos, v = sats_pos_and_v[0]  # get our position and velocity

    # Create the local actor, name will be the rank
    local_actor = ActorBuilder.get_actor_scaffold(
        name="Sat_" + str(rank), actor_type=SpacecraftActor, epoch=t0
    )
    ActorBuilder.set_orbit(
        actor=local_actor, position=pos, velocity=v, epoch=t0, central_body=earth
    )
    ActorBuilder.add_comm_device(
        actor=local_actor, device_name="Link1", bandwidth_in_kbps=1000
    )

    # Battery from https://sentinels.copernicus.eu/documents/247904/349490/S2_SP-1322_2.pdf
    # 87Ah * 28 Volt = 8.7696e9Ws
    ActorBuilder.set_power_devices(
        actor=local_actor,
        battery_level_in_Ws=8.7696e9 * 0.5,
        max_battery_level_in_Ws=8.7696e9,
        charging_rate_in_W=1700,
    )

    # TODO update and sanity check
    ActorBuilder.set_thermal_model(
        actor=local_actor,
        actor_mass=1200.0,
        actor_initial_temperature_in_K=293.15,
        actor_sun_absorptance=0.8,
        actor_infrared_absorptance=0.8,
        actor_sun_facing_area=4.0,
        actor_central_body_facing_area=4.0,
        actor_emissive_area=24.0,
        actor_thermal_capacity=1000,
    )

    paseos_instance = paseos.init_sim(local_actor=local_actor)
    print(f"Rank {rank} set up its PASEOS instance for its local actor {local_actor}")

    # Ground stations
    stations = [
        ["Maspalomas", 27.7629, -15.6338, 205.1],
        ["Matera", 40.6486, 16.7046, 536.9],
        ["Svalbard", 78.9067, 11.8883, 474.0],
    ]
    for station in stations:
        gs_actor = ActorBuilder.get_actor_scaffold(
            name=station[0], actor_type=GroundstationActor, epoch=t0
        )
        ActorBuilder.set_ground_station_location(
            gs_actor,
            latitude=station[1],
            longitude=station[2],
            elevation=station[3],
            minimum_altitude_angle=5,
        )
        paseos_instance.add_known_actor(gs_actor)

    return (
        paseos_instance,
        local_actor,
    )
