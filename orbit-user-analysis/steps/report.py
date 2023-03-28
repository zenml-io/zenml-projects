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
from datetime import datetime
from typing import Tuple, List, Any

from discord import SyncWebhook, Embed
from zenml.steps import step, BaseParameters

from constants import BOOMING_TAG, CHURNED_TAG
from steps.utils import list_members, get_orbit_secrets, get_discord_secret


class ReportParameters(BaseParameters):
    """Parameters for the report step.

    Attributes:
        check_days: the amount of days to go back while creating the link.
    """
    check_days: int = 180


def generate_link(tag: str, days: int) -> str:
    """Generates a link for a filtered orbit page.

    Args:
        tag: str, tag to filter with
        days: int, the number of days to filter with

    Returns:
        the generated link
    """

    workspace, _ = get_orbit_secrets()

    return f"https://app.orbit.love/{workspace}/members" \
           f"?affiliation_equals=false" \
           f"&tags_contains_any_of%5B%5D={tag}" \
           f"&activity_type_integer_is_set%5B0%5D=" \
           f"&timeframe_date_relative%5B0%5D=this_{days}_days"


@step
def report(params: ReportParameters) -> Tuple[List[Any], List[Any]]:
    """Step that sends Discord messages and saves the results."""

    discord_webhook = SyncWebhook.from_url(get_discord_secret())

    content = "Member analysis report generated at " \
              f"{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}"

    embeds = []
    for tag in [BOOMING_TAG, CHURNED_TAG]:
        embeds.append(
            Embed(
                title=f"{tag} users".upper(),
                description=f"Link to the Orbit members page, which is "
                            f"prefiltered to include members with the `{tag}` "
                            f"tag and who have been active at least once "
                            f"in the last {params.check_days} days!",
                url=generate_link(tag=tag, days=params.check_days),
                colour=12745742,
            )
        )
    discord_webhook.send(content=content, embeds=embeds)

    booming_users = list_members(tags=BOOMING_TAG)
    churning_users = list_members(tags=CHURNED_TAG)
    return booming_users, churning_users
