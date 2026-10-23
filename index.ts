import * as fs from "node:fs/promises";
import * as path from "node:path";
import {
	type Api,
	type AssistantMessage,
	AssistantMessageEventStream,
	type Context,
	type Model,
	type SimpleStreamOptions,
	streamSimple,
} from "@oh-my-pi/pi-ai";
import type {
	ExtensionAPI,
	ExtensionCommandContext,
	ExtensionUIContext,
	ProviderModelConfig,
} from "@oh-my-pi/pi-coding-agent";
import { CATALOG, type CatalogEntry, resolveHfRef } from "./src/catalog";
import { BASE_URL, ensureServing, HOME, isDownloaded, LOG_FILE, PORT, status, stopServer } from "./src/runtime";

const PROVIDER = "hf";
const CUSTOM_FILE = path.join(HOME, "custom.json");
const STATUS_KEY = "omp-hf";
const CUSTOM_OPTION = "Custom Hugging Face repo…";
const STOP_OPTION = "Stop llama-server";

async function loadCustom(): Promise<CatalogEntry[]> {
	try {
		return JSON.parse(await fs.readFile(CUSTOM_FILE, "utf8")) as CatalogEntry[];
	} catch {
		return [];
	}
}

function toProviderModel(entry: CatalogEntry): ProviderModelConfig {
	const ctx = Number(process.env.OMP_HF_CTX) || entry.contextWindow;
	return {
		id: entry.id,
		name: `${entry.name} (local)`,
		reasoning: entry.reasoning,
		input: entry.mmproj ? ["text", "image"] : ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: ctx,
		maxTokens: Math.min(entry.maxTokens, Math.floor(ctx / 2)),
		// Mirrors omp's llama.cpp provider rules: llama-server rejects these OpenAI extras.
		compat: {
			supportsStore: false,
			supportsDeveloperRole: false,
			supportsReasoningEffort: false,
			supportsNamedToolChoice: false,
			...(entry.family === "qwen" ? { thinkingFormat: "qwen" } : {}),
		},
	};
}

function errorMessage(model: Model<Api>, err: unknown, aborted: boolean): AssistantMessage {
	return {
		role: "assistant",
		content: [],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: aborted ? "aborted" : "error",
		errorMessage: err instanceof Error ? err.message : String(err),
		timestamp: Date.now(),
	} as AssistantMessage;
}

function formatSize(bytes: number): string {
	return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.round(bytes / 1e6)} MB`;
}

export default function ompHf(pi: ExtensionAPI) {
	let entries: CatalogEntry[] = [...CATALOG];
	// Main-session UI, used to surface download/load progress from the provider stream.
	let ui: ExtensionUIContext | undefined;

	const report = (message: string | undefined) => {
		ui?.setStatus(STATUS_KEY, message);
		ui?.setWorkingMessage(message);
	};

	const stream = (model: Model<Api>, context: Context, options?: SimpleStreamOptions) => {
		const out = new AssistantMessageEventStream();
		void (async () => {
			try {
				const entry = entries.find(e => e.id === model.id);
				if (!entry) throw new Error(`Unknown omp-hf model "${model.id}"; run /hf to pick one`);
				await out.trackLocalWork(ensureServing(entry, report, options?.signal));
			} catch (err) {
				const error = errorMessage(model, err, options?.signal?.aborted === true);
				out.push({ type: "error", reason: error.stopReason as "aborted" | "error", error });
				out.end();
				return;
			} finally {
				report(undefined);
			}
			try {
				const inner = streamSimple({ ...model, api: "openai-completions" } as Model<"openai-completions">, context, options);
				out.forwardLocalWorkFrom(inner);
				for await (const event of inner) out.push(event);
				out.end();
			} catch (err) {
				out.fail(err);
			}
		})();
		return out;
	};

	const register = () =>
		pi.registerProvider(PROVIDER, {
			baseUrl: `${BASE_URL}/v1`,
			apiKey: "omp-hf-local",
			api: "omp-hf",
			streamSimple: stream,
			models: entries.map(toProviderModel),
		});

	register();
	void loadCustom().then(custom => {
		if (custom.length === 0) return;
		entries = [...CATALOG, ...custom.filter(c => !CATALOG.some(e => e.id === c.id))];
		register();
	});

	pi.on("session_start", (_event, ctx) => {
		if (ctx.agent.kind === "main" && ctx.hasUI) ui = ctx.ui;
	});

	async function activate(entry: CatalogEntry, ctx: ExtensionCommandContext): Promise<void> {
		const model = ctx.models.resolve(`${PROVIDER}/${entry.id}`);
		if (!model || !(await pi.setModel(model))) {
			ctx.ui.notify(`Could not select ${PROVIDER}/${entry.id}`, "error");
			return;
		}
		ui = ctx.ui;
		try {
			await ensureServing(entry, report);
			ctx.ui.notify(`${entry.name} is serving on ${BASE_URL} and selected as the model`, "info");
		} catch (err) {
			ctx.ui.notify(err instanceof Error ? err.message : String(err), "error");
		} finally {
			report(undefined);
		}
	}

	async function addCustom(ref: string, ctx: ExtensionCommandContext): Promise<void> {
		let entry: CatalogEntry;
		try {
			entry = await resolveHfRef(ref);
		} catch (err) {
			ctx.ui.notify(err instanceof Error ? err.message : String(err), "error");
			return;
		}
		const ok = await ctx.ui.confirm(
			`Add ${entry.name}?`,
			`${entry.repo} → ${entry.file}${entry.mmproj ? ` + ${entry.mmproj}` : ""} (${formatSize(entry.bytes)})`,
		);
		if (!ok) return;
		const custom = (await loadCustom()).filter(c => c.id !== entry.id);
		custom.push(entry);
		await fs.mkdir(HOME, { recursive: true });
		await fs.writeFile(CUSTOM_FILE, `${JSON.stringify(custom, null, 2)}\n`);
		entries = [...entries.filter(e => e.id !== entry.id), entry];
		register();
		await activate(entry, ctx);
	}

	pi.registerCommand("hf", {
		description: "Pick a small Hugging Face GGUF model to download and serve locally (/hf, /hf user/repo[:QUANT], /hf stop, /hf status)",
		getArgumentCompletions: prefix =>
			["status", "stop", ...entries.map(e => e.id)]
				.filter(v => v.startsWith(prefix))
				.map(value => ({ value, label: value })),
		handler: async (args, ctx) => {
			const arg = args.trim();
			if (arg === "stop") {
				ctx.ui.notify((await stopServer()) ? "llama-server stopped" : "llama-server was not running", "info");
				return;
			}
			if (arg === "status") {
				const s = await status();
				ctx.ui.notify(
					s.state === "down"
						? `llama-server is not running (port ${PORT})`
						: `llama-server ${s.state} on ${BASE_URL}: ${s.modelId ?? "unknown model"} (pid ${s.pid ?? "?"}). Log: ${LOG_FILE}`,
					"info",
				);
				return;
			}
			if (arg) {
				const known = entries.find(e => e.id === arg);
				await (known ? activate(known, ctx) : addCustom(arg, ctx));
				return;
			}

			const served = (await status()).modelId;
			const options = await Promise.all(
				entries.map(async e => ({
					label: e.name,
					description: [
						formatSize(e.bytes),
						e.note,
						(await isDownloaded(e)) ? "downloaded" : undefined,
						e.id === served ? "serving" : undefined,
					]
						.filter(Boolean)
						.join(" · "),
				})),
			);
			const choice = await ctx.ui.select("Local Hugging Face models", [
				...options,
				{ label: CUSTOM_OPTION, description: "user/repo or user/repo:QUANT" },
				{ label: STOP_OPTION, description: `Free the memory held by the server on port ${PORT}` },
			]);
			if (!choice) return;
			if (choice === STOP_OPTION) {
				ctx.ui.notify((await stopServer()) ? "llama-server stopped" : "llama-server was not running", "info");
			} else if (choice === CUSTOM_OPTION) {
				const ref = await ctx.ui.input("Hugging Face GGUF repo", "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M");
				if (ref?.trim()) await addCustom(ref, ctx);
			} else {
				const entry = entries.find(e => e.name === choice);
				if (entry) await activate(entry, ctx);
			}
		},
	});
}
