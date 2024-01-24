#  Copyright (c) ZenML GmbH 2024. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

from typing import List
from datetime import datetime, timedelta, timezone

from zenml.steps import BaseParameters, step
from supabase import create_client, Client as SupabaseClient
from zenml.client import Client


class SupabaseReaderParams(BaseParameters):
    table_name: str = "analytics"
    filter_date_column: str = "created_at"  # assume supabase timestampz col
    filter_interval_hours: int = 24  # interval for hours
    summary_column: str = "video_title"  # the column to summarize
    limit: int = 300  # limit the number of rows to read


@step(enable_cache=False)
def supabase_reader(
    params: SupabaseReaderParams,
) -> List[str]:
    """Reads from supabase and returns a list of dicts."""
    supabase_secret = Client().get_secret("supabase")

    supabase: SupabaseClient = create_client(
        supabase_secret.secret_values["supabase_url"],
        supabase_secret.secret_values["supabase_key"],
    )

    interval = datetime.now(timezone.utc) - timedelta(
        hours=params.filter_interval_hours
    )

    # Create a supabase query to filter for the last 24 hours
    response = (
        supabase.table(params.table_name)
        .select(params.summary_column)
        .filter(params.filter_date_column, "gte", interval)
        .order(params.filter_date_column, desc=True)
        .limit(params.limit)
        .execute()
    )

    return [d[params.summary_column] for d in response.data]
