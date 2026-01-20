# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.trinity_tool_parser import TrinityToolParser


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {
        "<tool_call>": 1,
        "</tool_call>": 2,
    }
    return tokenizer


@pytest.fixture
def trinity_tool_parser(mock_tokenizer):
    return TrinityToolParser(mock_tokenizer)


@pytest.fixture
def mock_request():
    request = MagicMock(spec=ChatCompletionRequest)
    request.tools = []
    request.tool_choice = "auto"
    return request


def test_extract_tool_calls_no_tools(trinity_tool_parser, mock_request):
    model_output = "Just some text."
    result = trinity_tool_parser.extract_tool_calls(model_output, mock_request)

    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == model_output


def test_extract_tool_calls_with_think_tags(trinity_tool_parser, mock_request):
    model_output = (
        "<think>thinking</think> prefix\n"
        "<tool_call>\n"
        '{"name": "get_weather", "arguments": {"location": "Paris"}}\n'
        "</tool_call>"
    )
    result = trinity_tool_parser.extract_tool_calls(model_output, mock_request)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "get_weather"
    assert json.loads(result.tool_calls[0].function.arguments) == {
        "location": "Paris"
    }
    assert result.content == "thinking prefix"


def test_extract_tool_calls_multiple(trinity_tool_parser, mock_request):
    model_output = (
        "<tool_call>\n"
        '{"name": "get_weather", "arguments": {"location": "Paris"}}\n'
        "</tool_call>\n"
        "<tool_call>\n"
        '{"name": "get_time", "arguments": {"timezone": "UTC"}}\n'
        "</tool_call>"
    )
    result = trinity_tool_parser.extract_tool_calls(model_output, mock_request)

    assert result.tools_called is True
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].function.name == "get_weather"
    assert result.tool_calls[1].function.name == "get_time"


def test_extract_tool_calls_streaming_basic(trinity_tool_parser, mock_request):
    deltas = [
        "hello <think>thought</think>\n",
        "<tool_call>\n"
        '{"name": "get_weather", "arguments": {"location": "Paris"}}\n'
        "</tool_call>",
    ]

    previous_text = ""
    accumulated_content = ""
    tool_calls_by_index = {}
    token_index = 0

    for delta_text in deltas:
        current_text = previous_text + delta_text
        delta_message = trinity_tool_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=list(range(token_index)),
            current_token_ids=list(range(token_index + 1)),
            delta_token_ids=[token_index],
            request=mock_request,
        )
        token_index += 1
        previous_text = current_text

        if not delta_message:
            continue
        if delta_message.content:
            accumulated_content += delta_message.content
        for tool_call_chunk in delta_message.tool_calls:
            tool_calls_by_index[tool_call_chunk.index] = tool_call_chunk

    assert accumulated_content == "hello thought\n"
    assert len(tool_calls_by_index) == 1
    tool_call = tool_calls_by_index[0]
    assert tool_call.function.name == "get_weather"
    assert json.loads(tool_call.function.arguments) == {"location": "Paris"}
