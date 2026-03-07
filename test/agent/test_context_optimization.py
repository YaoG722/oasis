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

import os

import pytest
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelType
from camel.types import OpenAIBackendRole
from camel.types import ModelPlatformType

from oasis.social_agent.agent import SocialAgent
from oasis.social_agent.agent_environment import ObservationConfig, SocialEnvironment
from oasis.social_platform.channel import Channel
from oasis.social_platform.config import UserInfo

os.environ.setdefault("OPENAI_API_KEY", "test-key")

TEST_MODEL = ModelFactory.create(
    model_platform=ModelPlatformType.STUB,
    model_type=ModelType.STUB,
)


def _build_user_info() -> UserInfo:
    return UserInfo(
        name="tester",
        description="test profile",
        profile={"other_info": {"user_profile": "test user"}},
    )


def _build_agent(**kwargs) -> SocialAgent:
    kwargs.setdefault("model", TEST_MODEL)
    return SocialAgent(
        agent_id=0,
        user_info=_build_user_info(),
        channel=Channel(),
        **kwargs,
    )


def test_message_window_size_limits_context_history():
    agent = _build_agent(message_window_size=2)

    for idx in range(4):
        user_message = BaseMessage.make_user_message(
            role_name="User",
            content=f"message-{idx}",
        )
        agent.update_memory(user_message, OpenAIBackendRole.USER)

    context_messages, _ = agent.memory.get_context()
    context_text = [
        msg.get("content", "")
        for msg in context_messages
        if isinstance(msg.get("content"), str)
    ]

    assert "message-0" not in context_text
    assert "message-1" not in context_text
    assert "message-2" in context_text
    assert "message-3" in context_text


class _DummyAgentResponse:

    def __init__(self):
        self.info = {"tool_calls": []}


@pytest.mark.asyncio
async def test_environment_observation_can_be_transient(monkeypatch):
    agent = _build_agent(persist_environment_observation=False)

    async def fake_to_text_prompt(*_args, **_kwargs):
        return "ENV-SNAPSHOT-CONTENT"

    async def fake_astep(user_msg):
        agent.update_memory(user_msg, OpenAIBackendRole.USER)
        agent.update_memory(
            BaseMessage.make_assistant_message(
                role_name="Assistant",
                content="done",
            ),
            OpenAIBackendRole.ASSISTANT,
        )
        return _DummyAgentResponse()

    monkeypatch.setattr(agent.env, "to_text_prompt", fake_to_text_prompt)
    monkeypatch.setattr(agent, "astep", fake_astep)

    await agent.perform_action_by_llm()

    context_messages, _ = agent.memory.get_context()
    context_text = [
        msg.get("content", "")
        for msg in context_messages
        if isinstance(msg.get("content"), str)
    ]

    assert "ENV-SNAPSHOT-CONTENT" not in "\n".join(context_text)


@pytest.mark.asyncio
async def test_environment_observation_truncation_config():

    class _FakeAction:

        def __init__(self):
            self.agent_id = 0

        async def refresh(self):
            return {
                "success": True,
                "posts": [
                    {
                        "post_id": 1,
                        "content": "A" * 500,
                        "quote_content": "B" * 500,
                    },
                    {
                        "post_id": 2,
                        "content": "C" * 500,
                        "quote_content": "D" * 500,
                    },
                ],
            }

        async def listen_from_group(self):
            return {
                "success": True,
                "all_groups": {
                    1: "group-1",
                    2: "group-2",
                },
                "joined_groups": [1, 2],
                "messages": {
                    1: [{"content": "M" * 400}],
                    2: [{"content": "N" * 400}],
                },
            }

    env = SocialEnvironment(
        action=_FakeAction(),
        observation_config=ObservationConfig(
            max_posts=1,
            max_post_text_chars=30,
            max_groups=1,
            max_group_messages=1,
            max_group_message_chars=20,
            max_total_prompt_chars=500,
        ),
    )

    prompt = await env.to_text_prompt(
        include_followers=False,
        include_follows=False,
    )

    assert "Showing 1 of 2 posts" in prompt
    assert "group-2" not in prompt
    assert "[truncated]" in prompt
    assert len(prompt) <= 500
