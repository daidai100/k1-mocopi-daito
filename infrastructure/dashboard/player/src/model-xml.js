export const PLAYBACK_ARENA_MEMORY = "128M";

/**
 * Bound MuJoCo's transient arena for state-only browser playback.
 *
 * Legacy robosuite XML uses nconmax/njmax, which MuJoCo 3.12 translates into
 * an arena larger than 1 GiB for these scenes. The viewer only restores qpos,
 * runs mj_forward, and renders; a fixed 128 MiB arena avoids exhausting the
 * WebAssembly heap without changing bodies, joints, geometry, or recorded
 * states.
 */
export function boundPlaybackArena(modelXml) {
  const bounded = `<size memory="${PLAYBACK_ARENA_MEMORY}"/>`;
  const size = /<size\b[^>]*(?:\/>|>\s*<\/size>)/i;
  if (size.test(modelXml)) return modelXml.replace(size, bounded);
  return modelXml.replace(/<mujoco\b[^>]*>/i, match => `${match}\n  ${bounded}`);
}
