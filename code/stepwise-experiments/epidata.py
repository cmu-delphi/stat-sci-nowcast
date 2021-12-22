"""
Functions for querying data from the covidcast API and converting it to the necessary format.

For more features and up to date support for data retrieval, use the official covidcast API clients:
https://cmu-delphi.github.io/delphi-epidata/api/covidcast_clients.html
"""
import asyncio
from datetime import datetime, date
import os
from typing import Tuple, List, Dict
from itertools import product

from numpy import isnan
from pandas import date_range
from aiohttp import ClientSession, ContentTypeError
from json import JSONDecodeError
from delphi_epidata import Epidata
from tenacity import AsyncRetrying, RetryError, stop_after_attempt

from data_containers import LocationSeries, SensorConfig

EPIDATA_START_DATE = 20200101


async def get(params, session, sensor, location):
    """Helper function to make Epidata GET requests."""
    try:
        async for attempt in AsyncRetrying(stop=stop_after_attempt(4)):
            with attempt:
                async with session.get(Epidata.BASE_URL, params=params) as response:
                    return await response.json(), sensor, location
    except RetryError:
        pass

async def fetch_epidata(combos, as_of):
    """Helper function to asynchronously make and aggregate Epidata GET requests."""
    tasks = []
    async with ClientSession() as session:
        for sensor, location in combos:
            params = {
                    "endpoint": "covidcast",
                    "data_source": sensor.source,
                    "signals": sensor.signal,
                    "time_type": "day",
                    "geo_type": location.geo_type,
                    "time_values": f"{EPIDATA_START_DATE}-{as_of}",
                    "geo_value": location.geo_value,
                    "as_of": as_of,
                }
            task = asyncio.create_task(get(params, session, sensor, location))
            tasks.append(task)
        responses = await asyncio.gather(*tasks)
        return responses


def get_indicator_data(sensors: List[SensorConfig],
                       locations: List[LocationSeries],
                       as_of: date) -> Dict[Tuple, LocationSeries]:
    """
    Given a list of sensors and locations, asynchronously gets covidcast data for all combinations.

    Parameters
    ----------
    sensors
        list of SensorConfigs for sensors to retrieve.
    locations
        list of LocationSeries, one for each location desired. This is only used for the list of
        locations; none of the dates or values are used.
    as_of
        Date that the data should be retrieved as of.
    Returns
    -------
        Dictionary of LocationSeries with desired data.
    """
    output = {}
    all_combos = product(sensors, locations)
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(fetch_epidata(all_combos, as_of.strftime("%Y%m%d")))
    responses = loop.run_until_complete(future)
    responses = [i for i in responses if i]
    for response, sensor, location in responses:
        # -2 = no results, 1 = success. Truncated data or server errors may lead to this Exception.
        if response["result"] not in (-2, 1):
            raise Exception(f"Bad result from Epidata: {response['message']}")
        data = LocationSeries(
            geo_value=location.geo_value,
            geo_type=location.geo_type,
            data={datetime.strptime(str(i["time_value"]), "%Y%m%d").date(): i["value"]
                  for i in response.get("epidata", []) if not isnan(i["value"])}
        )
        if data.data:
            output[(sensor.source, sensor.signal, location.geo_type, location.geo_value)] = data
    return output
