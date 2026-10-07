import numpy as np
import pytest
from fastapi.testclient import TestClient
from pusht_diffusion.live import LiveSession, PolicyRegistry, create_app


class MockPolicy:
    name = 'act'
    provenance = {'kind': 'mock'}

    def predict_action_chunk(self, history, **kwargs):
        return np.repeat(np.array([[200.0, 200.0]], dtype=np.float32), 16, axis=0)


def test_continuous_interventions_reset_switch_and_pending():
    registry = PolicyRegistry({'act': MockPolicy(), 'unet': MockPolicy()})
    s = LiveSession(registry, seed=100)
    try:
        initial = s.initial_scene
        s.tick({'paused': False})
        assert s.steps == 1 and len(s.actions) == 3
        s.tick({'paused': True})
        assert s.steps == 1
        s.move_block(280, 280)
        assert len(s.actions) == 0 and s.history[0] is s.history[1]
        pose = tuple(s.env.block.position)
        s.switch('unet')
        assert tuple(s.env.block.position) == pose and len(s.actions) == 0
        with pytest.raises(ValueError):
            s.switch('dit')
        s.tick({'reset': 'same'})
        assert s.initial_scene == initial and s.steps == 0
        # Continuous env has no TimeLimit; success signal is intentionally ignored.
        s.steps = 300
        s.tick({'paused': False})
        assert s.steps == 301
        s.actions.clear()
        step = s.steps
        s.tick({}, revision_valid=lambda: False)
        assert s.steps == step and not s.actions
        s.tick({'reset': 'new', 'paused': True})
        assert s.initial_scene != initial
    finally:
        s.close()


def test_app_routes_and_latest_frame_socket():
    with TestClient(create_app(PolicyRegistry({'act': MockPolicy()}))) as c:
        assert c.get('/live').status_code == 200
        status = c.get('/status').json()
        assert status['mode'] == 'exploratory_only' and not status['models'][1]['available']
        with c.websocket_connect('/ws/live') as ws:
            first = ws.receive_json()
            assert first['type'] == 'frame' and first['paused']
            ws.send_json({'type': 'pause', 'paused': False})
            for _ in range(20):
                if ws.receive_json().get('steps', 0) > 0:
                    break
            else:
                raise AssertionError('Session did not run')


def test_origin_messages_and_debug_ledger(tmp_path):
    import json

    ledger = tmp_path / 'interactive-debug.jsonl'
    registry = PolicyRegistry({'act': MockPolicy(), 'unet': MockPolicy()}, debug_ledger=ledger)
    s = LiveSession(registry)
    try:
        s.move_block(280, 270)
        s.switch('unet')
        s.reset()
    finally:
        s.close()
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [r['event'] for r in records] == ['reset', 'drag', 'switch', 'reset']
    assert all(r['evidence_kind'] == 'interactive_debug' for r in records)
    with TestClient(create_app(registry)) as c:
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect('/ws/live', headers={'origin': 'https://evil.example'}):
                pass
        with c.websocket_connect('/ws/live', headers={'origin': 'http://testserver'}) as ws:
            ws.send_json(['not', 'an', 'object'])
            ws.send_json(None)
            assert ws.receive_json()['type'] == 'frame'


def test_disconnect_joins_inflight_native_tick(monkeypatch):
    import threading
    import time
    import pusht_diffusion.live as live

    entered = threading.Event()
    finished = threading.Event()
    closed = threading.Event()
    original_tick = live.LiveSession.tick
    original_close = live.LiveSession.close

    def slow_tick(self, *args, **kwargs):
        entered.set()
        time.sleep(0.08)
        try:
            return original_tick(self, *args, **kwargs)
        finally:
            finished.set()

    def checked_close(self):
        assert finished.is_set(), 'physics closed before worker finished'
        original_close(self)
        closed.set()

    monkeypatch.setattr(live.LiveSession, 'tick', slow_tick)
    monkeypatch.setattr(live.LiveSession, 'close', checked_close)
    with TestClient(create_app(PolicyRegistry({'act': MockPolicy()}))) as c:
        with c.websocket_connect('/ws/live'):
            assert entered.wait(2)
    assert finished.is_set() and closed.is_set()
