"""Continuous exploratory sessions; no writes to formal evaluation results."""

from __future__ import annotations
import asyncio
import anyio
import base64
from collections import deque
import contextlib
import io
import math
import json
from urllib.parse import urlsplit
from pathlib import Path
import threading
import time
from typing import Callable
import numpy as np
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from .environment import make_env, legal_scene
from .utils import append_event


class PolicyRegistry:
    def __init__(self, policies: dict, *, debug_ledger: str | Path | None = None):
        self.policies = policies
        self.lock = threading.RLock()
        self.debug_ledger = debug_ledger

    def record_debug(self, record: dict):
        if self.debug_ledger is not None:
            with self.lock:
                append_event(self.debug_ledger, record)

    def descriptions(self) -> list[dict]:
        return [
            {
                'name': name,
                'available': name in self.policies,
                'status': '已加载' if name in self.policies else '待实现 / 未加载兼容权重',
            }
            for name in ('act', 'unet', 'dit')
        ]

    def get(self, name):
        if name not in self.policies:
            raise ValueError(f'{name}: 待实现 / 未加载兼容权重')
        return self.policies[name]

    def predict(self, name, history, **kwargs):
        with self.lock:
            return self.get(name).predict_action_chunk(history, **kwargs)


class LiveSession:
    def __init__(self, registry: PolicyRegistry, *, seed: int = 0, model: str = 'act'):
        self.registry = registry
        registry.get(model)
        self.model_name = model
        self.env = make_env(continuous=True)
        self.history = deque(maxlen=2)
        self.actions = deque()
        self.paused = True
        self.held = False
        self.drag_target = None
        self.latency_ms = 0.0
        self.reset(seed)

    def clear_context(self):
        obs = self.env.get_obs()
        self.history.clear()
        self.history.extend([obs, obs])
        self.actions.clear()
        self.replans = 0

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.seed, self.initial_scene = legal_scene(seed)
        obs, _ = self.env.reset(seed=self.seed, options={'reset_to_state': self.initial_scene.as_array()})
        self.history.clear()
        self.history.extend([obs, obs])
        self.actions.clear()
        self.steps = 0
        self.replans = 0
        self.held = False
        self.drag_target = None
        self.latency_ms = 0.0
        self.registry.record_debug(
            {
                'event': 'reset',
                'environment_seed': self.seed,
                'state': self.initial_scene.to_dict(),
                'state_semantics': 'reset_origin',
                'evidence_kind': 'interactive_debug',
            }
        )

    def move_block(self, x: float, y: float, *, record: bool = True):
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError('拖动坐标必须为有限值')
        body = self.env.block
        body.position = (float(np.clip(x, 100, 400)), float(np.clip(y, 100, 400)))
        body.velocity = (0, 0)
        body.angular_velocity = 0
        body.force = (0, 0)
        body.torque = 0
        body.activate()
        self.env.space.reindex_shapes_for_body(body)
        self.clear_context()
        if record:
            self.registry.record_debug(
                {
                    'event': 'drag',
                    'environment_seed': self.seed,
                    'state': {
                        'agent_x': float(self.env.agent.position.x),
                        'agent_y': float(self.env.agent.position.y),
                        'block_x': float(body.position.x),
                        'block_y': float(body.position.y),
                        'block_angle': float(body.angle),
                    },
                    'state_semantics': 'physical_body_pose_after_intervention',
                    'evidence_kind': 'interactive_debug',
                }
            )

    def switch(self, name: str):
        self.registry.get(name)
        self.model_name = name
        self.clear_context()
        body = self.env.block
        self.registry.record_debug(
            {
                'event': 'switch',
                'environment_seed': self.seed,
                'model': name,
                'state': {
                    'agent_x': float(self.env.agent.position.x),
                    'agent_y': float(self.env.agent.position.y),
                    'block_x': float(body.position.x),
                    'block_y': float(body.position.y),
                    'block_angle': float(body.angle),
                },
                'state_semantics': 'physical_body_pose_after_context_rebuild',
                'evidence_kind': 'interactive_debug',
            }
        )

    def tick(self, commands: dict, *, revision_valid: Callable[[], bool] = lambda: True) -> dict:
        # 先处理暂停、重置、切换策略和拖动指令。
        if 'paused' in commands:
            self.paused = commands['paused']
        if commands.get('reset') == 'same':
            self.reset()
        elif commands.get('reset') == 'new':
            self.reset(self.seed + 1)
        if 'model' in commands:
            self.switch(commands['model'])
        if 'drag' in commands:
            drag = commands['drag']
            self.drag_target = (drag['x'], drag['y'])
            self.held = drag['held']
            self.move_block(*self.drag_target)
        if self.held and self.drag_target:
            self.move_block(*self.drag_target, record=False)

        # 动作队列用完后重新规划，本次 tick 只执行一个环境步。
        if not self.paused and not self.held:
            if not self.actions:
                begin = time.perf_counter()
                actions = self.registry.predict(
                    self.model_name, list(self.history), scene_seed=self.seed, replan_index=self.replans
                )
                self.latency_ms = (time.perf_counter() - begin) * 1000
                # Any newer control invalidates this prediction before physics executes it.
                if not revision_valid():
                    return self.frame()
                actions = np.asarray(actions)
                if actions.shape != (16, 2) or not np.isfinite(actions).all():
                    raise ValueError('Invalid prediction')
                self.actions.extend(actions[:4])
                self.replans += 1
            obs, _, _, _, _ = self.env.step(self.actions.popleft())
            self.history.append(obs)
            self.steps += 1
        return self.frame()

    def frame(self) -> dict:
        buffer = io.BytesIO()
        Image.fromarray(np.asarray(self.env.render())).save(buffer, format='JPEG', quality=85)
        body = self.env.block
        return {
            'type': 'frame',
            'image': base64.b64encode(buffer.getvalue()).decode(),
            'steps': self.steps,
            'paused': self.paused,
            'seed': self.seed,
            'model': self.model_name,
            'coverage': float(self.env._get_coverage()),
            'latency_ms': self.latency_ms,
            'replans': self.replans,
            'block': [float(body.position.x), float(body.position.y)],
            'models': self.registry.descriptions(),
        }

    def close(self):
        self.env.close()


def create_app(registry: PolicyRegistry) -> FastAPI:
    app = FastAPI(title='Push-T Policy Lab')

    @app.get('/')
    def index():
        return RedirectResponse('/live')

    @app.get('/live', response_class=HTMLResponse)
    def live():
        return (Path(__file__).parent / 'web/live.html').read_text()

    @app.get('/status')
    def status():
        return {'mode': 'exploratory_only', 'models': registry.descriptions()}

    @app.websocket('/ws/live')
    async def socket(ws: WebSocket):
        origin = ws.headers.get('origin')
        if origin:
            parsed = urlsplit(origin)
            if parsed.scheme not in ('http', 'https') or parsed.netloc != ws.headers.get('host'):
                await ws.close(code=1008)
                return
        await ws.accept()
        pending = {}
        revision = 0
        outgoing = asyncio.Queue(maxsize=1)
        session = None

        async def receive():
            nonlocal revision
            # 新指令覆盖同类旧指令，并使尚未执行的旧预测失效。
            while True:
                try:
                    msg = await ws.receive_json()
                except json.JSONDecodeError:
                    continue
                if not isinstance(msg, dict):
                    continue
                kind = msg.get('type')
                if kind == 'pause' and isinstance(msg.get('paused'), bool):
                    pending['paused'] = msg['paused']
                elif kind == 'reset' and msg.get('mode') in ('same', 'new'):
                    pending['reset'] = msg['mode']
                elif kind == 'model' and msg.get('name') in ('act', 'unet', 'dit'):
                    pending['model'] = msg['name']
                elif kind == 'drag':
                    try:
                        x = float(msg['x'])
                        y = float(msg['y'])
                        if not math.isfinite(x) or not math.isfinite(y):
                            continue
                        pending['drag'] = {'x': x, 'y': y, 'held': msg.get('held') is True}
                    except (ValueError, KeyError, TypeError):
                        continue
                else:
                    continue
                revision += 1

        async def send():
            while True:
                await ws.send_json(await outgoing.get())

        async def simulate():
            while True:
                # 取得本轮控制指令，将物理仿真放到独立线程执行。
                snapshot = dict(pending)
                pending.clear()
                observed = revision
                worker = asyncio.create_task(
                    asyncio.to_thread(session.tick, snapshot, revision_valid=lambda: revision == observed)
                )
                try:
                    frame = await asyncio.shield(worker)
                except asyncio.CancelledError:
                    # Cancelling to_thread never stops its native worker. Join before closing physics.
                    with anyio.CancelScope(shield=True):
                        with contextlib.suppress(Exception):
                            await asyncio.shield(worker)
                    raise
                except (ValueError, RuntimeError, NotImplementedError) as exc:
                    session.paused = True
                    frame = {'type': 'error', 'message': str(exc)}
                if outgoing.full():
                    outgoing.get_nowait()
                outgoing.put_nowait(frame)
                await asyncio.sleep(0.05 if session.paused else 0.001)

        tasks = []
        try:
            session = await asyncio.to_thread(LiveSession, registry)
            tasks = [asyncio.create_task(fn()) for fn in (receive, send, simulate)]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            # ASGI disconnect/shutdown may cancel the surrounding AnyIO scope too.
            # Shield cleanup so native work and environment close complete in order.
            with anyio.CancelScope(shield=True):
                for task in tasks:
                    task.cancel()
                for task in tasks:
                    with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                        await task
                if session:
                    await asyncio.to_thread(session.close)

    return app
