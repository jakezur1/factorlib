import json
import os
from factorlib.utils.system import get_datetime_maps_dir


YF_INTERVALS_JSON = get_datetime_maps_dir() / 'yf_intervals.json'
TIMEDELTAS_JSON = get_datetime_maps_dir() / 'timedelta_intervals.json'

with open(YF_INTERVALS_JSON) as yf:
    yf_intervals = json.load(yf)

with open(TIMEDELTAS_JSON) as timedeltas:
    timedelta_intervals = json.load(timedeltas)
