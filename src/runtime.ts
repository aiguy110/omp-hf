import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { type CatalogEntry, hfHeaders } from "./catalog";

/** PrismML's llama.cpp fork: upstream plus the PQ2_0/PTQ1_0 kernels Bonsai 2 needs. */
const LLAMA_REPO = "PrismML-Eng/llama.cpp";
const LLAMA_TAG = process.env.OMP_HF_LLAMA_TAG || "prism-b10787-73bfe6c";

export const HOME = process.env.OMP_HF_HOME || path.join(os.homedir(), ".omp", "hf");
export const PORT = Number(process.env.OMP_HF_PORT || 18080);
export const BASE_URL = `http://127.0.0.1:${PORT}`;
const STATE_FILE = path.join(HOME, "server.json");
export const LOG_FILE = path.join(HOME, "llama-server.log");

export type Progress = (message: string) => void;

interface ServerState {
	pid: number;
	modelId: string;
	port: number;
}

// ── llama.cpp binaries ─────────────────────────────────────────────────────

interface Variant {
	asset: string;
	gpu: boolean;
}

function pickVariant(): Variant {
	const forced = process.env.OMP_HF_BACKEND;
	const asset = (suffix: string) => `llama-${LLAMA_TAG}-bin-${suffix}.tar.gz`;
	const { platform, arch } = process;
	if (platform === "darwin") return { asset: asset(arch === "arm64" ? "macos-arm64" : "macos-x64"), gpu: arch === "arm64" };
	if (platform !== "linux") throw new Error(`omp-hf supports Linux and macOS; ${platform} is not supported`);
	if (arch === "arm64") {
		return forced === "vulkan" ? { asset: asset("ubuntu-vulkan-arm64"), gpu: true } : { asset: asset("ubuntu-arm64"), gpu: false };
	}
	// Integrated GPUs stay on CPU unless asked: on an Arc iGPU, Vulkan prefilled ~1.5x faster but
	// decoded ~0.7x as fast as CPU in llama-server, and thinking models are decode-bound.
	const backend = forced || (Bun.which("nvidia-smi") ? "cuda" : Bun.which("rocminfo") ? "rocm" : "cpu");
	switch (backend) {
		case "cuda":
			return { asset: asset("linux-cuda-12.4-x64"), gpu: true };
		case "cuda-12.8":
		case "cuda-13.3":
			return { asset: asset(`linux-${backend}-x64`), gpu: true };
		case "rocm":
			return { asset: asset("ubuntu-rocm-7.2-x64"), gpu: true };
		case "vulkan":
			return { asset: asset("ubuntu-vulkan-x64"), gpu: true };
		case "cpu":
			return { asset: asset("ubuntu-x64"), gpu: false };
		default:
			throw new Error(`Unknown OMP_HF_BACKEND "${backend}" (cpu, cuda, cuda-12.8, cuda-13.3, rocm, vulkan)`);
	}
}

async function findFile(dir: string, name: string): Promise<string | undefined> {
	for (const entry of await fs.readdir(dir, { recursive: true, withFileTypes: true })) {
		if (entry.isFile() && entry.name === name) return path.join(entry.parentPath, entry.name);
	}
	return undefined;
}

/** Resolve `llama-server`, downloading the pinned PrismML build on first use. */
export async function ensureLlamaServer(progress: Progress, signal?: AbortSignal): Promise<{ bin: string; gpu: boolean }> {
	const variant = pickVariant();
	const override = process.env.OMP_HF_LLAMA_SERVER;
	if (override) return { bin: override, gpu: variant.gpu };

	const dir = path.join(HOME, "llama", variant.asset.replace(/\.tar\.gz$/, ""));
	const existing = await findFile(dir, "llama-server").catch(() => undefined);
	if (existing) return { bin: existing, gpu: variant.gpu };

	const archive = `${dir}.tar.gz`;
	await download(`https://github.com/${LLAMA_REPO}/releases/download/${LLAMA_TAG}/${variant.asset}`, archive, "llama.cpp runtime", progress, signal);
	const staging = `${dir}.partial`;
	await fs.rm(staging, { recursive: true, force: true });
	await fs.mkdir(staging, { recursive: true });
	const tar = Bun.spawn(["tar", "-xzf", archive, "-C", staging], { stderr: "pipe" });
	if ((await tar.exited) !== 0) throw new Error(`Extracting ${archive} failed: ${await new Response(tar.stderr).text()}`);
	await fs.rename(staging, dir);
	await fs.rm(archive, { force: true });
	const bin = await findFile(dir, "llama-server");
	if (!bin) throw new Error(`${variant.asset} contains no llama-server`);
	return { bin, gpu: variant.gpu };
}

// ── Downloads ──────────────────────────────────────────────────────────────

const MB = 1024 * 1024;

/** Stream `url` to `dest`, resuming a previous `.part` file when the server honors Range. */
export async function download(url: string, dest: string, label: string, progress: Progress, signal?: AbortSignal): Promise<void> {
	await fs.mkdir(path.dirname(dest), { recursive: true });
	const part = `${dest}.part`;
	const have = await fs.stat(part).then(s => s.size, () => 0);
	const headers: Record<string, string> = url.includes("huggingface.co") ? hfHeaders() : {};
	if (have > 0) headers.Range = `bytes=${have}-`;

	const res = await fetch(url, { headers, signal });
	if (res.status === 416) {
		// The partial file is already complete.
		await fs.rename(part, dest);
		return;
	}
	if (!res.ok || !res.body) throw new Error(`Download of ${label} failed: HTTP ${res.status} from ${url}`);
	const resumed = res.status === 206;
	let done = resumed ? have : 0;
	const total = done + Number(res.headers.get("content-length") || 0);

	const file = await fs.open(part, resumed ? "a" : "w");
	let lastReport = 0;
	try {
		for await (const chunk of res.body) {
			await file.write(chunk);
			done += chunk.byteLength;
			const now = Date.now();
			if (now - lastReport > 500) {
				lastReport = now;
				const pct = total ? ` ${Math.floor((done / total) * 100)}%` : "";
				progress(`Downloading ${label}${pct} (${Math.round(done / MB)}/${Math.round(total / MB)} MB)`);
			}
		}
	} finally {
		await file.close();
	}
	if (total && done !== total) throw new Error(`Download of ${label} truncated at ${done}/${total} bytes; retry to resume`);
	await fs.rename(part, dest);
}

export function modelPath(entry: CatalogEntry, file: string): string {
	return path.join(HOME, "models", entry.repo, file);
}

export async function isDownloaded(entry: CatalogEntry): Promise<boolean> {
	const files = [entry.file, entry.mmproj].filter((f): f is string => !!f);
	const present = await Promise.all(files.map(f => fs.exists(modelPath(entry, f))));
	return present.every(Boolean);
}

async function ensureWeights(entry: CatalogEntry, progress: Progress, signal?: AbortSignal): Promise<void> {
	for (const file of [entry.file, entry.mmproj]) {
		if (!file) continue;
		const dest = modelPath(entry, file);
		if (await fs.exists(dest)) continue;
		await download(`https://huggingface.co/${entry.repo}/resolve/main/${file}`, dest, file, progress, signal);
	}
}

// ── Server lifecycle ───────────────────────────────────────────────────────

async function readState(): Promise<ServerState | undefined> {
	try {
		return JSON.parse(await fs.readFile(STATE_FILE, "utf8")) as ServerState;
	} catch {
		return undefined;
	}
}

function alive(pid: number): boolean {
	try {
		process.kill(pid, 0);
		return true;
	} catch {
		return false;
	}
}

/** `"ready"`, `"loading"`, or `"down"` for whatever listens on our port. */
async function health(): Promise<"ready" | "loading" | "down"> {
	try {
		const res = await fetch(`${BASE_URL}/health`, { signal: AbortSignal.timeout(2000) });
		return res.ok ? "ready" : "loading";
	} catch {
		return "down";
	}
}

async function servedModelId(): Promise<string | undefined> {
	try {
		const res = await fetch(`${BASE_URL}/v1/models`, { signal: AbortSignal.timeout(2000) });
		const body = (await res.json()) as { data?: { id: string }[] };
		return body.data?.[0]?.id;
	} catch {
		return undefined;
	}
}

export async function status(): Promise<{ state: "ready" | "loading" | "down"; modelId?: string; pid?: number }> {
	const state = await readState();
	const h = await health();
	return { state: h, modelId: h === "ready" ? await servedModelId() : state?.modelId, pid: state?.pid };
}

export async function stopServer(): Promise<boolean> {
	const state = await readState();
	await fs.rm(STATE_FILE, { force: true });
	if (!state || !alive(state.pid)) return false;
	process.kill(state.pid, "SIGTERM");
	for (let i = 0; i < 50 && alive(state.pid); i++) await Bun.sleep(100);
	if (alive(state.pid)) process.kill(state.pid, "SIGKILL");
	return true;
}

async function startServer(entry: CatalogEntry, progress: Progress, signal?: AbortSignal): Promise<void> {
	const { bin, gpu } = await ensureLlamaServer(progress, signal);
	await ensureWeights(entry, progress, signal);

	const ctx = Number(process.env.OMP_HF_CTX) || entry.contextWindow;
	const args = [
		"-m", modelPath(entry, entry.file),
		...(entry.mmproj ? ["--mmproj", modelPath(entry, entry.mmproj)] : []),
		"--alias", entry.id,
		"--host", "127.0.0.1",
		"--port", String(PORT),
		"-c", String(ctx),
		"-ngl", gpu ? "99" : "0",
		"-fa", "on",
		"--jinja",
		...entry.serverArgs,
		...(process.env.OMP_HF_SERVER_ARGS?.split(/\s+/).filter(Boolean) ?? []),
	];

	const logStart = await fs.stat(LOG_FILE).then(s => s.size, () => 0);
	// Detach through `sh` so the server outlives this omp process and is shared by later sessions.
	const launcher = Bun.spawn(["sh", "-c", 'nohup "$@" >>"$OMP_HF_LOG" 2>&1 </dev/null & echo $!', "sh", bin, ...args], {
		env: { ...process.env, OMP_HF_LOG: LOG_FILE },
		stdout: "pipe",
	});
	const pid = Number((await new Response(launcher.stdout).text()).trim());
	await launcher.exited;
	await fs.writeFile(STATE_FILE, JSON.stringify({ pid, modelId: entry.id, port: PORT } satisfies ServerState));

	progress(`Loading ${entry.name}…`);
	while ((await health()) !== "ready") {
		if (!alive(pid)) {
			const log = await Bun.file(LOG_FILE).slice(logStart).text();
			await fs.rm(STATE_FILE, { force: true });
			throw new Error(`llama-server exited while loading ${entry.name}:\n${log.trim().split("\n").slice(-15).join("\n")}`);
		}
		if (signal?.aborted) throw signal.reason;
		await Bun.sleep(250);
	}
}

let queue: Promise<unknown> = Promise.resolve();

/**
 * Make `entry` the model served on {@link PORT}: install llama.cpp, download
 * weights, and (re)start the server as needed. Calls are serialized.
 */
export function ensureServing(entry: CatalogEntry, progress: Progress, signal?: AbortSignal): Promise<void> {
	const run = queue.then(async () => {
		const h = await health();
		if (h === "ready" && (await servedModelId()) === entry.id) return;
		const state = await readState();
		if (h !== "down" && !(state && alive(state.pid))) {
			throw new Error(`Port ${PORT} is in use by a server omp-hf did not start; set OMP_HF_PORT to another port`);
		}
		await stopServer();
		await startServer(entry, progress, signal);
	});
	queue = run.catch(() => {});
	return run;
}
