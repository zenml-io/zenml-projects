# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import time

from label_studio_sdk import Client

LABEL_STUDIO_URL = os.getenv(
    "LABEL_STUDIO_URL", default="http://localhost:8080"
)
# API_KEY = "xxxxx"
PROJECT_ID = int("1")
VIEW_ID = False  # or:int("18")

# connect to Label Studio
ls = Client(
    url=LABEL_STUDIO_URL, api_key="78afe31bd58c157723501a0351e3716e9c52a714"
)
ls.check_connection()

# get existing project
project = ls.get_project(PROJECT_ID)

# get the first tab
views = project.get_views()


project.export_tasks(
    export_type="YOLO",
    export_location="/tmp/whatever.zip",
    download_resources=True,
)


breakpoint()

for view in views:
    if VIEW_ID and VIEW_ID != view["id"]:
        continue

    task_filter_options = {"view": view["id"]} if views else {}
    view_name = view["data"]["title"]

    # create new export snapshot
    export_result = project.export_snapshot_create(
        title="Export SDK Snapshot", task_filter_options=task_filter_options
    )
    assert "id" in export_result
    export_id = export_result["id"]

    # wait until snapshot is ready
    while project.export_snapshot_status(export_id).is_in_progress():
        time.sleep(1.0)

    # download snapshot file
    status, file_name = project.export_snapshot_download(export_id)
    assert status == 200
    assert file_name is not None
    os.rename(file_name, view_name + ".json")
    print(f"Status of the export is {status}.\nFile name is {view_name}.json")
