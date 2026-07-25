"""Tests for the durable background-task core (vNext §9.3, Stage 9 D)."""
from __future__ import annotations

import app.tasks as T


def _rec(id="t1", state=T.PENDING, schedule=None, last_run=None, created_at=0.0):
    return T.TaskRecord(spec=T.TaskSpec(id=id, schedule=schedule or T.Schedule()),
                        state=state, last_run=last_run, created_at=created_at)


# ---- state machine --------------------------------------------------------
def test_legal_transitions():
    assert T.can_transition(T.PENDING, T.RUNNING)
    assert T.can_transition(T.RUNNING, T.NEEDS_INPUT)
    assert T.can_transition(T.RUNNING, T.COMPLETED)
    assert T.can_transition(T.NEEDS_INPUT, T.RUNNING)
    assert T.can_transition(T.PAUSED, T.RUNNING)


def test_illegal_transitions():
    assert not T.can_transition(T.COMPLETED, T.RUNNING)
    assert not T.can_transition(T.PENDING, T.COMPLETED)  # must run first
    assert not T.can_transition(T.CANCELLED, T.RUNNING)


def test_terminal_and_runnable_predicates():
    for s in (T.COMPLETED, T.FAILED, T.CANCELLED):
        assert T.is_terminal(s) and not T.is_runnable(s)
    for s in (T.PENDING, T.PAUSED):
        assert T.is_runnable(s) and not T.is_terminal(s)
    assert not T.is_runnable(T.NEEDS_INPUT)   # parked, not runnable


def test_transition_mutates_or_rejects():
    r = _rec()
    assert T.transition(r, T.RUNNING) and r.state == T.RUNNING
    assert not T.transition(r, T.PENDING) and r.state == T.RUNNING


# ---- schedule -------------------------------------------------------------
def test_once_runs_then_never():
    s = T.Schedule("once")
    assert T.next_run(s, now=1000, created_at=900) == 900
    assert T.next_run(s, now=1000, last_run=950) is None


def test_interval_schedule():
    s = T.Schedule("interval", interval_s=60)
    assert T.next_run(s, now=1000) == 1000            # never run → now
    assert T.next_run(s, now=1000, last_run=980) == 1040


def test_daily_schedule_next_boundary():
    s = T.Schedule("daily", at_second_of_day=100)
    # now=1000 is past today's 100 → tomorrow's 100.
    assert T.next_run(s, now=1000) == 86500
    # now=50 is before today's 100 → today's 100.
    assert T.next_run(s, now=50) == 100


def test_next_run_never_raises():
    assert T.next_run(T.Schedule("bogus"), now=1000) is None


def test_due_tasks_filters_by_time_and_runnable():
    recs = [
        _rec("due", schedule=T.Schedule("interval", interval_s=10), last_run=980),
        _rec("notyet", schedule=T.Schedule("interval", interval_s=10), last_run=995),
        _rec("parked", state=T.NEEDS_INPUT,
             schedule=T.Schedule("interval", interval_s=10), last_run=900),
    ]
    due = T.due_tasks(recs, now=1000)
    assert [r.spec.id for r in due] == ["due"]    # notyet future; parked not runnable


# ---- runner scheduling ----------------------------------------------------
def test_pick_runnable_fills_free_slots():
    pool = [_rec(str(i)) for i in range(5)]
    assert [r.spec.id for r in T.pick_runnable(pool, running=1, max_concurrency=2)] == ["0"]
    assert [r.spec.id for r in T.pick_runnable(pool, running=0, max_concurrency=2)] == ["0", "1"]


def test_pick_runnable_no_slots():
    pool = [_rec(str(i)) for i in range(3)]
    assert T.pick_runnable(pool, running=2, max_concurrency=2) == []


def test_pick_runnable_skips_non_runnable():
    pool = [_rec("a", state=T.RUNNING), _rec("b", state=T.NEEDS_INPUT), _rec("c")]
    assert [r.spec.id for r in T.pick_runnable(pool, running=0, max_concurrency=5)] == ["c"]


# ---- checkpoint + rehydrate ----------------------------------------------
def test_advance_checkpoint():
    cp = T.Checkpoint(step=0, total=3,
                      todos=[{"text": "a", "done": False}, {"text": "b", "done": False}])
    T.advance_checkpoint(cp, done_index=0, artifact="out.md", data={"k": 1})
    assert cp.step == 1
    assert cp.todos[0]["done"] is True
    assert cp.artifacts == ["out.md"]
    assert cp.data == {"k": 1}
    assert round(cp.progress(), 2) == 0.33


def test_checkpoint_progress_zero_total():
    assert T.Checkpoint(step=2, total=0).progress() == 0.0


def test_rehydrate_resumes_paused_and_running():
    for s in (T.PAUSED, T.RUNNING):
        r = _rec(state=s)
        T.rehydrate(r)
        assert r.state == T.RUNNING            # resume from checkpoint


def test_rehydrate_keeps_parked_and_terminal():
    r = _rec(state=T.NEEDS_INPUT)
    T.rehydrate(r)
    assert r.state == T.NEEDS_INPUT            # stays parked on human input
    r2 = _rec(state=T.COMPLETED)
    T.rehydrate(r2)
    assert r2.state == T.COMPLETED


# ---- side-effect step gate (reuses §9.9) ---------------------------------
def test_side_effect_step_needs_approval():
    assert T.step_needs_approval("file_write")
    assert T.step_needs_approval("git_push")
    assert T.step_needs_approval("create_task")


def test_read_only_step_no_approval():
    assert not T.step_needs_approval("web_search")
    assert not T.step_needs_approval("conversation_search")


# ---- end-to-end lifecycle -------------------------------------------------
def test_task_survives_restart_and_resumes_from_checkpoint():
    # A task mid-run gets checkpointed, "restart" rehydrates it to RUNNING with
    # its progress intact — the §9.3 acceptance criterion.
    r = _rec("fix-codebase", state=T.RUNNING)
    r.checkpoint = T.Checkpoint(step=2, total=5, artifacts=["patch1.diff"])
    T.transition(r, T.PAUSED)                  # drain checkpoints it
    assert r.state == T.PAUSED
    T.rehydrate(r)                             # pod restart
    assert r.state == T.RUNNING
    assert r.checkpoint.step == 2 and r.checkpoint.artifacts == ["patch1.diff"]
