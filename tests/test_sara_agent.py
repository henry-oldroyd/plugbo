import json

from sara.agent import _progress_line, run_campaign
from llm.base import ChatResponse, LLMClient, ToolCall


class ScriptedClient(LLMClient):
    """Replays a fixed sequence of responses, ignoring the actual message history."""

    def __init__(self, responses: list[ChatResponse]):
        super().__init__(model="scripted")
        self._responses = list(responses)
        self.calls = 0

    def chat(self, messages, tools, system):
        self.calls += 1
        return self._responses.pop(0)


def test_agent_executes_tool_call_then_reports(tmp_path):
    responses = [
        ChatResponse(
            content="checking status",
            tool_calls=[ToolCall(id="1", name="bash", arguments={"cmd": "echo hello"})],
            stop_reason="tool_use",
        ),
        ChatResponse(content="done: incumbent is x=1", tool_calls=[], stop_reason="end_turn"),
    ]
    client = ScriptedClient(responses)
    result = run_campaign(client, system_prompt="sys", user_prompt="go", sandbox=tmp_path)

    assert result.final_message == "done: incumbent is x=1"
    assert result.n_steps == 2
    tool_entries = [e for e in result.transcript if e.get("role") == "tool"]
    assert len(tool_entries) == 1
    assert "hello" in tool_entries[0]["content"]


def test_agent_sums_usage_across_turns_and_logs_it_per_turn(tmp_path):
    responses = [
        ChatResponse(
            content="checking status",
            tool_calls=[ToolCall(id="1", name="bash", arguments={"cmd": "echo hi"})],
            stop_reason="tool_use",
            usage={"input_tokens": 100, "output_tokens": 20, "cache_read_tokens": None, "cache_creation_tokens": None},
        ),
        ChatResponse(
            content="done",
            tool_calls=[],
            stop_reason="end_turn",
            usage={"input_tokens": 150, "output_tokens": 10, "cache_read_tokens": 5, "cache_creation_tokens": None},
        ),
    ]
    client = ScriptedClient(responses)
    trace_path = tmp_path / "trace.jsonl"
    result = run_campaign(client, system_prompt="sys", user_prompt="go", sandbox=tmp_path, trace_path=trace_path)

    assert result.usage_total == {
        "input_tokens": 250,
        "output_tokens": 30,
        "cache_read_tokens": 5,
        "cache_creation_tokens": 0,
    }
    assistant_entries = [json.loads(l) for l in trace_path.read_text().strip().splitlines() if json.loads(l).get("role") == "assistant"]
    assert assistant_entries[0]["usage"]["input_tokens"] == 100
    assert assistant_entries[1]["usage"]["input_tokens"] == 150


def test_agent_usage_total_is_none_when_provider_never_reports_it(tmp_path):
    responses = [ChatResponse(content="done", tool_calls=[], stop_reason="end_turn")]  # usage defaults to None
    client = ScriptedClient(responses)
    result = run_campaign(client, system_prompt="sys", user_prompt="go", sandbox=tmp_path)
    assert result.usage_total is None


def test_agent_writes_trace_file(tmp_path):
    responses = [ChatResponse(content="ok", tool_calls=[], stop_reason="end_turn")]
    client = ScriptedClient(responses)
    trace_path = tmp_path / "trace.jsonl"
    run_campaign(client, system_prompt="sys", user_prompt="go", sandbox=tmp_path, trace_path=trace_path)

    lines = trace_path.read_text().strip().splitlines()
    assert len(lines) >= 2
    entries = [json.loads(line) for line in lines]  # each line is valid JSON
    assert entries[0]["role"] == "user" and entries[0]["content"] == "go"  # initial prompt is actually on disk
    assert isinstance(entries[0]["ts"], (int, float))
    assert entries[1]["role"] == "assistant"


def test_agent_trace_file_is_fresh_per_run(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text('{"role": "user", "content": "stale from a previous run"}\n')

    client = ScriptedClient([ChatResponse(content="ok", tool_calls=[], stop_reason="end_turn")])
    run_campaign(client, system_prompt="sys", user_prompt="fresh prompt", sandbox=tmp_path, trace_path=trace_path)

    entries = [json.loads(line) for line in trace_path.read_text().strip().splitlines()]
    assert entries[0]["role"] == "user" and entries[0]["content"] == "fresh prompt"
    assert not any(e.get("content") == "stale from a previous run" for e in entries)


def _write_state(path, objectives, trials):
    state = {"shelf": {"objectives": objectives}, "trials": trials}
    path.write_text(json.dumps(state))


def test_progress_line_single_objective_minimize(tmp_path):
    trials = [
        {"status": "observed", "metrics": {"y": 5.0}, "observed_at": 1},
        {"status": "observed", "metrics": {"y": 2.0}, "observed_at": 2},
        {"status": "observed", "metrics": {"y": 3.0}, "observed_at": 3},
    ]
    _write_state(tmp_path / "state.json", [{"metric": "y", "minimize": True}], trials)
    line = _progress_line(tmp_path, budget=10)
    assert line == "eval 3/10  y=3  best=2"


def test_progress_line_maximize_uses_max_as_best(tmp_path):
    trials = [
        {"status": "observed", "metrics": {"y": 1.0}, "observed_at": 1},
        {"status": "observed", "metrics": {"y": 4.0}, "observed_at": 2},
    ]
    _write_state(tmp_path / "state.json", [{"metric": "y", "minimize": False}], trials)
    line = _progress_line(tmp_path, budget=None)
    assert line == "eval 2  y=4  best=4"


def test_progress_line_multi_objective(tmp_path):
    trials = [{"status": "observed", "metrics": {"y1": 1.0, "y2": 2.0}, "observed_at": 1}]
    _write_state(tmp_path / "state.json", [{"metric": "y1", "minimize": True}, {"metric": "y2", "minimize": False}], trials)
    line = _progress_line(tmp_path, budget=20)
    assert line == "eval 1/20  (2 objectives)"


def test_progress_line_no_observations_returns_none(tmp_path):
    _write_state(tmp_path / "state.json", [{"metric": "y", "minimize": True}], [])
    assert _progress_line(tmp_path, budget=10) is None


def test_progress_line_missing_state_returns_none(tmp_path):
    assert _progress_line(tmp_path, budget=10) is None


def test_agent_prints_progress_line_on_each_new_observation(tmp_path, capsys):
    def write_trial(y):
        state_path = tmp_path / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            "shelf": {"objectives": [{"metric": "y", "minimize": True}]},
            "trials": [],
        }
        state["trials"].append({"status": "observed", "metrics": {"y": y}, "observed_at": len(state["trials"]) + 1})
        state_path.write_text(json.dumps(state))

    class RecordingClient(LLMClient):
        def __init__(self):
            super().__init__(model="scripted")
            self.calls = 0

        def chat(self, messages, tools, system):
            self.calls += 1
            if self.calls <= 2:
                write_trial(10.0 - self.calls)
                return ChatResponse(
                    content="",
                    tool_calls=[ToolCall(id=str(self.calls), name="bash", arguments={"cmd": "echo ok"})],
                    stop_reason="tool_use",
                )
            return ChatResponse(content="done", tool_calls=[], stop_reason="end_turn")

    run_campaign(RecordingClient(), system_prompt="sys", user_prompt="go", sandbox=tmp_path, budget=5)
    err = capsys.readouterr().err
    assert "eval 1/5" in err
    assert "eval 2/5" in err


def test_agent_nudges_when_over_budget(tmp_path):
    (tmp_path / "state.json").write_text(
        json.dumps({"trials": [{"status": "observed"} for _ in range(10)]})
    )
    responses = [
        ChatResponse(
            content="",
            tool_calls=[ToolCall(id="1", name="bash", arguments={"cmd": "echo x"})],
            stop_reason="tool_use",
        ),
        ChatResponse(content="wrapping up", tool_calls=[], stop_reason="end_turn"),
    ]
    client = ScriptedClient(responses)
    result = run_campaign(client, system_prompt="sys", user_prompt="go", sandbox=tmp_path, budget=5)

    nudges = [e for e in result.transcript if e.get("role") == "user" and "budget of 5" in e.get("content", "")]
    assert len(nudges) == 1


def test_agent_stops_at_max_steps(tmp_path):
    looping = ChatResponse(
        content="",
        tool_calls=[ToolCall(id="1", name="bash", arguments={"cmd": "echo x"})],
        stop_reason="tool_use",
    )

    class InfiniteClient(LLMClient):
        def __init__(self):
            super().__init__(model="scripted")

        def chat(self, messages, tools, system):
            return looping

    result = run_campaign(
        InfiniteClient(), system_prompt="sys", user_prompt="go", sandbox=tmp_path, max_steps=3
    )
    assert result.n_steps == 3
    assert "max computational steps" in result.final_message


def test_read_tool_blocks_escape(tmp_path):
    from sara.tools import build_tools

    (tmp_path / "context.md").write_text("hello world")
    _, handlers = build_tools(tmp_path)
    assert handlers["read"]("context.md") == "hello world"
    assert "outside the sandbox" in handlers["read"]("../../etc/passwd")
    assert "not found" in handlers["read"]("missing.txt")


def test_bash_tool_runs_in_sandbox(tmp_path):
    from sara.tools import build_tools

    _, handlers = build_tools(tmp_path)
    out = handlers["bash"]("pwd")
    assert str(tmp_path.resolve()) in out


# --- early-stopping override (SARA_ENFORCE_BUDGET / enforce_budget) -------------------------------------


def _write_observed(sandbox, n):
    (sandbox / "state.json").write_text(json.dumps({"trials": [{"status": "observed"} for _ in range(n)]}))


def _final(text="final report"):
    return ChatResponse(content=text, tool_calls=[], stop_reason="end_turn")


def test_budget_not_enforced_by_default_agent_may_stop_early(tmp_path, monkeypatch):
    import sara.agent as agent

    monkeypatch.setattr(agent, "ENFORCE_BUDGET", False)
    _write_observed(tmp_path, 3)
    client = ScriptedClient([_final()])
    result = run_campaign(client, system_prompt="sys", user_prompt="go", sandbox=tmp_path, budget=10)
    assert client.calls == 1
    assert result.final_message == "final report"
    assert not any("forced_continuation" in e for e in result.transcript)


def test_enforce_budget_pushes_agent_on_until_cap(tmp_path, monkeypatch):
    import sara.agent as agent

    monkeypatch.setattr(agent, "MAX_REPROMPTS", 2)
    _write_observed(tmp_path, 3)  # 3 of 10 used, and the scripted model never evaluates anything
    client = ScriptedClient([_final("early 1"), _final("early 2"), _final("early 3")])
    result = run_campaign(client, system_prompt="sys", user_prompt="go", sandbox=tmp_path, budget=10, enforce_budget=True)

    forced = [e for e in result.transcript if "forced_continuation" in e]
    assert len(forced) == 2  # capped, so a stuck model cannot loop forever
    assert "3 of 10 evaluations" in forced[0]["content"]
    assert client.calls == 3
    assert result.final_message == "early 3"  # after the cap the run is allowed to end


def test_enforce_budget_run_completes_once_budget_is_used(tmp_path):
    state = tmp_path / "state.json"

    class Client(LLMClient):
        def __init__(self):
            super().__init__(model="scripted")
            self.calls = 0

        def chat(self, messages, tools, system):
            self.calls += 1
            if self.calls == 1:  # model gives up with 1/3 evaluations done
                return _final("I am done")
            if self.calls == 2:  # after being told to continue it evaluates the rest
                state.write_text(json.dumps({"trials": [{"status": "observed"} for _ in range(3)]}))
                return ChatResponse(
                    content="", tool_calls=[ToolCall(id="1", name="bash", arguments={"cmd": "echo ok"})], stop_reason="tool_use"
                )
            return _final("all done")

    _write_observed(tmp_path, 1)
    client = Client()
    result = run_campaign(client, system_prompt="sys", user_prompt="go", sandbox=tmp_path, budget=3, enforce_budget=True)
    assert client.calls == 3
    assert result.final_message == "all done"
    assert len([e for e in result.transcript if "forced_continuation" in e]) == 1


def test_enforce_budget_lets_agent_finish_when_budget_is_spent(tmp_path):
    _write_observed(tmp_path, 10)
    client = ScriptedClient([_final()])
    result = run_campaign(client, system_prompt="sys", user_prompt="go", sandbox=tmp_path, budget=10, enforce_budget=True)
    assert client.calls == 1
    assert not any("forced_continuation" in e for e in result.transcript)


def test_explicit_enforce_budget_argument_overrides_environment_default(tmp_path, monkeypatch):
    import sara.agent as agent

    monkeypatch.setattr(agent, "ENFORCE_BUDGET", True)  # what SARA_ENFORCE_BUDGET=1 sets
    monkeypatch.setattr(agent, "MAX_REPROMPTS", 1)
    _write_observed(tmp_path, 2)

    on = run_campaign(ScriptedClient([_final(), _final()]), system_prompt="s", user_prompt="g", sandbox=tmp_path, budget=5)
    assert len([e for e in on.transcript if "forced_continuation" in e]) == 1  # env default applied

    off = ScriptedClient([_final()])
    run_campaign(off, system_prompt="s", user_prompt="g", sandbox=tmp_path, budget=5, enforce_budget=False)
    assert off.calls == 1  # explicit False beats the env default


# --- text-mode tool calls (SARA_TEXT_TOOLS / text_tools) --------------------------------------------------


def test_extract_bash_block():
    from sara.agent import extract_bash_block

    assert extract_bash_block("run:\n```bash\nlenz status --state ./state.json\n```\nthen wait") == "lenz status --state ./state.json"
    assert extract_bash_block("```\nlenz suggest --state ./state.json\n```") == "lenz suggest --state ./state.json"  # unlabelled, command-like
    assert extract_bash_block("```\nsome output 1 2 3\n```") is None  # unlabelled prose/output
    assert extract_bash_block("<think>try ```bash\nrm -rf x\n```</think>no command here") is None  # reasoning ignored
    assert extract_bash_block("```bash\nfirst\n```\n```bash\nsecond\n```") == "first"  # only the first block
    assert extract_bash_block("") is None and extract_bash_block("plain text") is None


def test_text_tools_runs_fenced_command_as_bash_call(tmp_path):
    responses = [
        ChatResponse(content="Let me check.\n```bash\necho hello-from-text\n```", tool_calls=[], stop_reason="end_turn"),
        ChatResponse(content="done", tool_calls=[], stop_reason="end_turn"),
    ]
    client = ScriptedClient(responses)
    result = run_campaign(client, system_prompt="s", user_prompt="g", sandbox=tmp_path, text_tools=True)
    tool_entries = [e for e in result.transcript if e.get("role") == "tool"]
    assert len(tool_entries) == 1 and "hello-from-text" in tool_entries[0]["content"]
    assert result.final_message == "done" and client.calls == 2


def test_text_tools_off_by_default_so_fenced_command_is_just_text(tmp_path, monkeypatch):
    import sara.agent as agent

    monkeypatch.setattr(agent, "TEXT_TOOLS", False)
    client = ScriptedClient([ChatResponse(content="```bash\necho hi\n```", tool_calls=[], stop_reason="end_turn")])
    result = run_campaign(client, system_prompt="s", user_prompt="g", sandbox=tmp_path)
    assert client.calls == 1 and not [e for e in result.transcript if e.get("role") == "tool"]


def test_text_tools_combines_with_budget_enforcement(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"trials": [{"status": "observed"} for _ in range(1)]}))
    cmd = "python3 -c \"import json;json.dump({'trials':[{'status':'observed'}]*3},open('state.json','w'))\""
    responses = [
        ChatResponse(content="I will start:\n```bash\n" + cmd + "\n```", tool_calls=[], stop_reason="end_turn"),
        ChatResponse(content="all done", tool_calls=[], stop_reason="end_turn"),
    ]
    client = ScriptedClient(responses)
    result = run_campaign(client, system_prompt="s", user_prompt="g", sandbox=tmp_path, budget=3,
                          enforce_budget=True, text_tools=True)
    assert client.calls == 2 and result.final_message == "all done"
    assert not [e for e in result.transcript if "forced_continuation" in e]  # the fenced command counted as action
