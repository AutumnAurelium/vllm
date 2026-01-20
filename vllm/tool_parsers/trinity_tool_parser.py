# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
)

logger = init_logger(__name__)


class TrinityToolParser(ToolParser):
    """
    Tool call parser for Trinity models using the Qwen-style format:

    <tool_call>
    {"name":"func1","arguments":{...}}
    </tool_call>

    Tool calls may appear inside <think> sections, so think tags are stripped
    before parsing.
    """

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        self.current_tool_name_sent = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id = -1
        self.streamed_args_for_tool: list[str] = []

        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        self.tool_calls_start_token = self.tool_call_start_token

        self.tool_call_regex = re.compile(
            r"<tool_call>\s*(?P<json>.*?)\s*</tool_call>", re.DOTALL
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        self._buffer = ""

    def _strip_think_tags(self, text: str) -> str:
        return text.replace("<think>", "").replace("</think>", "")

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        stripped_output = self._strip_think_tags(model_output)
        if self.tool_calls_start_token not in stripped_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=stripped_output
            )

        try:
            tool_call_json_list = self.tool_call_regex.findall(stripped_output)
            tool_calls = []
            for tool_call_json in tool_call_json_list:
                tool_call_dict = json.loads(tool_call_json)
                args_str = json.dumps(
                    tool_call_dict.get("arguments", {}), ensure_ascii=False
                )
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=tool_call_dict.get("name", ""),
                            arguments=args_str,
                        ),
                    )
                )

            content = stripped_output[
                : stripped_output.find(self.tool_calls_start_token)
            ].rstrip("\n")
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )
        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=stripped_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        delta_text = self._strip_think_tags(delta_text)
        self._buffer += delta_text
        cur_text = self._buffer

        start_idx = cur_text.find(self.tool_call_start_token)
        if start_idx == -1:
            self._buffer = ""
            if self.current_tool_id > 0:
                cur_text = ""
            return DeltaMessage(content=cur_text if cur_text else None)

        end_idx = cur_text.find(self.tool_call_end_token, start_idx)
        if end_idx != -1:
            if self.current_tool_id == -1:
                self.current_tool_id = 0
                self.prev_tool_call_arr = []
                self.streamed_args_for_tool = []
            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")

            extracted_tool_calls = self.extract_tool_calls(
                cur_text[: end_idx + len(self.tool_call_end_token)], request
            )

            if len(extracted_tool_calls.tool_calls) == 0:
                logger.warning("Failed to extract any tool calls.")
                return None
            tool_call = extracted_tool_calls.tool_calls[0]
            self.prev_tool_call_arr[self.current_tool_id] = {
                "name": tool_call.function.name,
                "arguments": json.loads(tool_call.function.arguments),
            }
            self.streamed_args_for_tool[self.current_tool_id] = (
                tool_call.function.arguments
            )
            delta = DeltaMessage(
                content=extracted_tool_calls.content,
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        id=tool_call.id,
                        type=tool_call.type,
                        function=DeltaFunctionCall(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        ),
                    )
                ],
            )
            self.current_tool_id += 1
            self._buffer = cur_text[end_idx + len(self.tool_call_end_token) :]
            return delta

        self._buffer = cur_text[start_idx:]
        content = cur_text[:start_idx].rstrip("\n")
        return DeltaMessage(content=content if content else None)
