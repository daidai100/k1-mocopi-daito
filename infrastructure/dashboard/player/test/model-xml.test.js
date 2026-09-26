import test from "node:test";
import assert from "node:assert/strict";

import { PLAYBACK_ARENA_MEMORY, boundPlaybackArena } from "../src/model-xml.js";


test("replaces legacy robosuite contact limits with a bounded arena", () => {
  const xml = '<mujoco model="x"><size nconmax="5000" njmax="5000"></size><worldbody/></mujoco>';
  const bounded = boundPlaybackArena(xml);
  assert.match(bounded, new RegExp(`<size memory="${PLAYBACK_ARENA_MEMORY}"/>`));
  assert.doesNotMatch(bounded, /nconmax|njmax/);
  assert.match(bounded, /<worldbody\/>/);
});


test("replaces a self-closing size element", () => {
  assert.equal(
    boundPlaybackArena('<mujoco><size memory="2G"/><worldbody/></mujoco>'),
    `<mujoco><size memory="${PLAYBACK_ARENA_MEMORY}"/><worldbody/></mujoco>`,
  );
});


test("adds a bounded arena when size is absent", () => {
  assert.equal(
    boundPlaybackArena('<mujoco model="x"><worldbody/></mujoco>'),
    `<mujoco model="x">\n  <size memory="${PLAYBACK_ARENA_MEMORY}"/><worldbody/></mujoco>`,
  );
});
