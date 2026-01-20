from src.data_nodes.nodes import ImportValmer
from src.settings import BUCKET_NAME_HISTORICAL_VECTORS

first_time_update_loop = False
ts_all_files = ImportValmer(
    bucket_name=BUCKET_NAME_HISTORICAL_VECTORS,
)
try:
    us = ts_all_files.get_update_statistics()
except AttributeError:
    # statistcs will not exist, this will raise an error and
    first_time_update_loop = True
if first_time_update_loop == True:
    for i in range(360 // 5):
        ts_all_files = ImportValmer(
            bucket_name=BUCKET_NAME_HISTORICAL_VECTORS,
        )
        ts_all_files.run(force_update=True)


else:
    ts_all_files = ImportValmer(
        bucket_name=BUCKET_NAME_HISTORICAL_VECTORS,
    )
    ts_all_files.run(force_update=True)
