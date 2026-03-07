# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from string import Template
from typing import Any

from oasis.social_agent.agent_action import SocialAction
from oasis.social_platform.database import get_db_path


class Environment(ABC):

    @abstractmethod
    def to_text_prompt(self) -> str:
        r"""Convert the environment to text prompt."""
        raise NotImplementedError


@dataclass
class ObservationConfig:
    r"""Controls how much environment observation text is exposed to an LLM.

    All fields are optional and default to the original behavior (no explicit
    truncation) for backward compatibility.
    """

    max_posts: int | None = None
    max_post_text_chars: int | None = None
    max_groups: int | None = None
    max_group_messages: int | None = None
    max_group_message_chars: int | None = None
    max_total_prompt_chars: int | None = None


class SocialEnvironment(Environment):
    followers_env_template = Template("I have $num_followers followers.")
    follows_env_template = Template("I have $num_follows follows.")

    posts_env_template = Template(
        "After refreshing, you see some posts $posts")

    groups_env_template = Template(
        "And there are many group chat channels $all_groups\n"
        "And You are already in some groups $joined_groups\n"
        "You receive some messages from them $messages\n"
        "You can join the groups you are interested, "
        "leave the groups you already in, send messages to the group "
        "you already in.\n"
        "You must make sure you can only send messages to the group you "
        "are already in")
    env_template = Template(
        "$followers_env\n"
        "$follows_env\n"
        "$groups_env\n"
        "$posts_env\npick one you want to perform action that best "
        "reflects your current inclination based on your profile and "
        "posts content. Do not limit your action in just `like` to like posts")

    def __init__(self,
                 action: SocialAction,
                 observation_config: ObservationConfig | None = None):
        self.action = action
        self.observation_config = observation_config or ObservationConfig()

    @staticmethod
    def _truncate_text(content: Any, max_chars: int | None) -> Any:
        if not isinstance(content, str) or max_chars is None or max_chars < 0:
            return content
        if len(content) <= max_chars:
            return content
        return f"{content[:max_chars]}... [truncated]"

    def _truncate_nested_strings(self,
                                 payload: Any,
                                 max_chars: int | None) -> Any:
        if isinstance(payload, dict):
            return {
                key: self._truncate_nested_strings(value, max_chars)
                for key, value in payload.items()
            }
        if isinstance(payload, list):
            return [
                self._truncate_nested_strings(item, max_chars)
                for item in payload
            ]
        return self._truncate_text(payload, max_chars)

    def _apply_prompt_budget(self, text_prompt: str) -> str:
        max_chars = self.observation_config.max_total_prompt_chars
        if max_chars is None or max_chars < 0 or len(text_prompt) <= max_chars:
            return text_prompt

        suffix = "\n[Environment observation truncated to fit prompt budget.]"
        trimmed_limit = max(0, max_chars - len(suffix))
        return f"{text_prompt[:trimmed_limit]}{suffix}"

    async def get_posts_env(self) -> str:
        posts = await self.action.refresh()
        # TODO: Replace posts json format string to other formats
        if posts["success"]:
            post_list = posts["posts"]
            total_posts = len(post_list)

            max_posts = self.observation_config.max_posts
            if max_posts is not None and max_posts >= 0:
                post_list = post_list[:max_posts]

            post_list = self._truncate_nested_strings(
                post_list,
                self.observation_config.max_post_text_chars,
            )

            posts_env = json.dumps(post_list, indent=4)
            if len(post_list) < total_posts:
                posts_env += (
                    f"\n[Showing {len(post_list)} of {total_posts} posts in "
                    "this observation.]"
                )
            posts_env = self.posts_env_template.substitute(posts=posts_env)
        else:
            posts_env = "After refreshing, there are no existing posts."
        return posts_env

    async def get_followers_env(self) -> str:
        # TODO: Implement followers env
        agent_id = self.action.agent_id
        db_path = get_db_path()
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT num_followers FROM user WHERE agent_id = ?",
                           (agent_id, ))
            result = cursor.fetchone()
            num_followers = result[0] if result else 0
            conn.close()
        except Exception:
            num_followers = 0
        return self.followers_env_template.substitute(
            {"num_followers": num_followers})

    async def get_follows_env(self) -> str:
        # TODO: Implement follows env
        agent_id = self.action.agent_id
        try:
            db_path = get_db_path()
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT num_followings FROM user WHERE agent_id = ?",
                (agent_id, ))
            result = cursor.fetchone()
            num_followings = result[0] if result else 0
            conn.close()
        except Exception:
            num_followings = 0
        return self.follows_env_template.substitute(
            {"num_follows": num_followings})

    async def get_group_env(self) -> str:
        groups = await self.action.listen_from_group()
        if groups["success"]:
            all_groups = groups["all_groups"]
            joined_groups = groups["joined_groups"]
            messages = groups["messages"]

            max_groups = self.observation_config.max_groups
            if max_groups is not None and max_groups >= 0:
                all_groups = dict(list(all_groups.items())[:max_groups])
                joined_groups = joined_groups[:max_groups]

            max_group_messages = self.observation_config.max_group_messages
            if max_group_messages is not None and max_group_messages >= 0:
                messages = {
                    group_id: group_messages[:max_group_messages]
                    for group_id, group_messages in messages.items()
                }

            messages = self._truncate_nested_strings(
                messages,
                self.observation_config.max_group_message_chars,
            )

            all_groups = json.dumps(all_groups)
            joined_groups = json.dumps(joined_groups)
            messages = json.dumps(messages)
            groups_env = self.groups_env_template.substitute(
                all_groups=all_groups,
                joined_groups=joined_groups,
                messages=messages,
            )
        else:
            groups_env = "No groups."
        return groups_env

    async def to_text_prompt(
        self,
        include_posts: bool = True,
        include_followers: bool = True,
        include_follows: bool = True,
    ) -> str:
        followers_env = (await self.get_followers_env()
                         if include_followers else "No followers.")
        follows_env = (await self.get_follows_env()
                       if include_follows else "No follows.")
        posts_env = await self.get_posts_env() if include_posts else ""

        prompt = self.env_template.substitute(
            followers_env=followers_env,
            follows_env=follows_env,
            posts_env=posts_env,
            groups_env=await self.get_group_env(),
        )
        return self._apply_prompt_budget(prompt)
